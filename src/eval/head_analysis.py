"""Attention head analysis: syntactic/semantic probing, redundancy detection, and head importance."""  # noqa: E501

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadAnalysisConfig:
    """Configuration for attention head analysis."""

    n_layers: int = 2
    n_heads: int = 2
    redundancy_threshold: float = 0.9  # cosine similarity above this = redundant
    importance_method: str = "gradient"  # "gradient" | "ablation" | "entropy"


def compute_head_entropy(attn_weights: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute entropy per attention head.

    Args:
        attn_weights: Attention probabilities of shape (B, n_heads, T, T).
        eps: Small constant to avoid log(0).

    Returns:
        Tensor of shape (B, n_heads) -- entropy per head, averaged over query positions.
        Low entropy = focused head, high entropy = diffuse/uniform head.
    """
    p = attn_weights.clamp(min=eps)
    # entropy over key dimension for each query position: (B, n_heads, T)
    per_query_entropy = -(p * p.log()).sum(dim=-1)
    # mean over query positions: (B, n_heads)
    return per_query_entropy.mean(dim=-1)


def compute_head_agreement(attn_a: torch.Tensor, attn_b: torch.Tensor) -> float:
    """Compute cosine similarity between two attention distributions.

    Args:
        attn_a: Attention weights of shape (B, ...).
        attn_b: Attention weights of the same shape as attn_a.

    Returns:
        Float in [-1, 1] -- cosine similarity averaged over batch.
    """
    B = attn_a.shape[0]
    a_flat = attn_a.reshape(B, -1).float()
    b_flat = attn_b.reshape(B, -1).float()
    cos_sim = F.cosine_similarity(a_flat, b_flat, dim=1)  # (B,)
    return cos_sim.mean().item()


def find_redundant_heads(
    attn_weights_per_layer: list[torch.Tensor],
    threshold: float = 0.9,
) -> list[tuple[int, int, int, float]]:
    """Find redundant head pairs across all layers.

    Args:
        attn_weights_per_layer: List of (B, n_heads, T, T) tensors, one per layer.
        threshold: Cosine similarity above which two heads are considered redundant.

    Returns:
        List of (layer_idx, head_i, head_j, similarity) tuples where similarity > threshold,
        sorted by similarity descending.
    """
    redundant: list[tuple[int, int, int, float]] = []

    for layer_idx, attn in enumerate(attn_weights_per_layer):
        B, n_heads, T, _ = attn.shape
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                head_i = attn[:, i, :, :]  # (B, T, T)
                head_j = attn[:, j, :, :]  # (B, T, T)
                sim = compute_head_agreement(head_i, head_j)
                if sim > threshold:
                    redundant.append((layer_idx, i, j, sim))

    redundant.sort(key=lambda x: x[3], reverse=True)
    return redundant


def ablation_head_importance(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    layer_idx: int,
    head_idx: int,
) -> float:
    """Measure head importance by ablating (zeroing) one head's output.

    Registers a forward hook on the target attention layer that zeros out
    the output contribution of the specified head, then computes the loss
    difference versus baseline.

    Args:
        model: AureliusTransformer (or compatible) model.
        input_ids: Input token ids of shape (B, T).
        labels: Target token ids of shape (B, T), used with cross-entropy loss.
        layer_idx: Index of the transformer layer to ablate.
        head_idx: Index of the attention head within the layer to ablate.

    Returns:
        loss_diff = ablated_loss - baseline_loss. Positive means the head was important.
    """
    model.eval()
    with torch.no_grad():
        # Baseline forward pass
        logits_baseline = model(input_ids)
        if isinstance(logits_baseline, tuple):
            logits_baseline = logits_baseline[0]
        baseline_loss = F.cross_entropy(
            logits_baseline[:, :-1, :].reshape(-1, logits_baseline.shape[-1]),
            labels[:, 1:].reshape(-1),
        )

        attn_module = model.layers[layer_idx].attn
        head_dim = getattr(attn_module, "head_dim", None)
        if head_dim is None:
            n_heads = getattr(attn_module, "n_heads", 1)
            head_dim = attn_module.q_proj.out_features // n_heads

        def ablation_hook(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: object,
        ) -> object:
            if isinstance(output, tuple):
                out = output[0]
                rest = output[1:]
            else:
                out = output
                rest = None

            B, T, d_model = out.shape
            out = out.clone()
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            if end <= d_model:
                out[:, :, start:end] = 0.0

            if rest is not None:
                return (out,) + rest
            return out

        hook_handle = attn_module.register_forward_hook(ablation_hook)

        try:
            logits_ablated = model(input_ids)
            if isinstance(logits_ablated, tuple):
                logits_ablated = logits_ablated[0]
            ablated_loss = F.cross_entropy(
                logits_ablated[:, :-1, :].reshape(-1, logits_ablated.shape[-1]),
                labels[:, 1:].reshape(-1),
            )
        finally:
            hook_handle.remove()

    return (ablated_loss - baseline_loss).item()


class HeadImportanceAnalyzer:
    """Analyzes attention head importance and properties for an AureliusTransformer."""

    def __init__(self, model: nn.Module, config: HeadAnalysisConfig) -> None:
        self.model = model
        self.config = config

    def collect_attention_weights(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        """Collect attention weights from all layers via forward hooks.

        Since AureliusTransformer uses flash attention and does not expose raw
        attention weights, this method recomputes them from Q and K projections
        using the same approach as AttentionExtractor in attention_patterns.py.
        Falls back to synthetic uniform attention if the model structure is not recognized.

        Args:
            input_ids: Token ids of shape (B, T).

        Returns:
            List of n_layers tensors, each of shape (B, n_heads, T, T).
        """
        from src.model.attention import GroupedQueryAttention, apply_rope

        B = input_ids.shape[0]
        T = input_ids.shape[1]
        captured: list[torch.Tensor | None] = [None] * self.config.n_layers
        hooks: list[torch.utils.hooks.RemovableHook] = []

        def make_hook(layer_idx: int):
            def hook_fn(
                module: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                output: object,
            ) -> None:
                try:
                    x = inputs[0]
                    freqs_cis = inputs[1]
                    _B, S, _ = x.shape

                    q = module.q_proj(x).view(_B, S, module.n_heads, module.head_dim)
                    k = module.k_proj(x).view(_B, S, module.n_kv_heads, module.head_dim)

                    q = apply_rope(q, freqs_cis)
                    k = apply_rope(k, freqs_cis)

                    q = q.transpose(1, 2)  # (B, n_heads, S, head_dim)
                    k = k.transpose(1, 2)  # (B, n_kv_heads, S, head_dim)

                    if module.n_rep > 1:
                        k = (
                            k.unsqueeze(2)
                            .expand(_B, module.n_kv_heads, module.n_rep, S, module.head_dim)
                            .reshape(_B, module.n_heads, S, module.head_dim)
                        )

                    scale = math.sqrt(module.head_dim)
                    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

                    causal_mask = torch.triu(
                        torch.full((S, S), float("-inf"), device=attn_weights.device),
                        diagonal=1,
                    )
                    attn_weights = attn_weights + causal_mask
                    attn_weights = F.softmax(attn_weights, dim=-1)

                    captured[layer_idx] = attn_weights.detach().cpu()
                except Exception:
                    captured[layer_idx] = torch.ones(_B, self.config.n_heads, S, S) / S

            return hook_fn

        layer_idx = 0
        for _name, module in self.model.named_modules():
            if isinstance(module, GroupedQueryAttention) and layer_idx < self.config.n_layers:
                hook = module.register_forward_hook(make_hook(layer_idx))
                hooks.append(hook)
                layer_idx += 1

        try:
            self.model.eval()
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        result = []
        for i in range(self.config.n_layers):
            if captured[i] is not None:
                result.append(captured[i])
            else:
                result.append(torch.ones(B, self.config.n_heads, T, T) / T)

        return result

    def compute_all_entropies(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute entropy for every head in every layer.

        Args:
            input_ids: Token ids of shape (B, T).

        Returns:
            Tensor of shape (n_layers, B, n_heads) -- entropy values.
        """
        weights_per_layer = self.collect_attention_weights(input_ids)
        entropies = []
        for attn in weights_per_layer:
            ent = compute_head_entropy(attn)  # (B, n_heads)
            entropies.append(ent)
        return torch.stack(entropies, dim=0)  # (n_layers, B, n_heads)

    def rank_heads_by_entropy(self, input_ids: torch.Tensor) -> list[tuple[int, int, float]]:
        """Rank all heads by entropy (ascending = most focused first).

        Args:
            input_ids: Token ids of shape (B, T).

        Returns:
            List of (layer_idx, head_idx, mean_entropy) tuples sorted by entropy ascending.
        """
        all_entropies = self.compute_all_entropies(input_ids)  # (n_layers, B, n_heads)
        mean_ent = all_entropies.mean(dim=1)  # (n_layers, n_heads)

        ranked: list[tuple[int, int, float]] = []
        for layer_idx in range(self.config.n_layers):
            for head_idx in range(self.config.n_heads):
                ranked.append((layer_idx, head_idx, mean_ent[layer_idx, head_idx].item()))

        ranked.sort(key=lambda x: x[2])
        return ranked


def head_specialization_score(attn_weights: torch.Tensor) -> dict[str, float]:
    """Compute specialization metrics for a single head's attention matrix.

    Args:
        attn_weights: Attention matrix of shape (T, T) -- one head's distribution.

    Returns:
        Dict with three keys:
        - "diagonal_focus": mean of main diagonal (attends to self)
        - "local_focus": mean of +-1 off-diagonals (attends to neighbors)
        - "uniform_score": max_entropy / actual_entropy (how close to uniform)
    """
    T = attn_weights.shape[0]
    eps = 1e-9

    diagonal_focus = attn_weights.diagonal().mean().item()

    if T > 1:
        upper = attn_weights.diagonal(offset=1).tolist()
        lower = attn_weights.diagonal(offset=-1).tolist()
        local_vals = upper + lower
        local_focus = sum(local_vals) / len(local_vals)
    else:
        local_focus = 0.0

    max_entropy = math.log(T + eps)
    p = attn_weights.clamp(min=eps)
    actual_entropy = -(p * p.log()).sum(dim=-1).mean().item()
    uniform_score = max_entropy / (actual_entropy + eps)

    return {
        "diagonal_focus": diagonal_focus,
        "local_focus": local_focus,
        "uniform_score": uniform_score,
    }
