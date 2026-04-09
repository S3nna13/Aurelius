"""Attention head importance scoring and structured pruning of transformer attention heads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HeadPruningConfig:
    """Configuration for attention head pruning."""

    importance_metric: str = "gradient"  # "gradient" | "entropy" | "sensitivity"
    prune_fraction: float = 0.3
    min_heads_per_layer: int = 1
    global_pruning: bool = True  # prune globally (not per-layer)


def compute_head_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """Compute entropy of each attention head's distribution over keys.

    Args:
        attention_weights: (B, n_heads, T, T) -- attention probability matrix.

    Returns:
        (n_heads,) tensor -- lower entropy = more focused = more important.
    """
    p = attention_weights
    entropy = -(p * (p + 1e-9).log()).sum(dim=-1)  # (B, n_heads, T)
    return entropy.mean(dim=(0, -1))  # (n_heads,)


def compute_head_sensitivity(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Compute proxy sensitivity scores for each attention head via gradient of scaling gates.

    For each head, a scaling gate (initialized to 1) is applied to the head's portion
    of the attention output. The gradient magnitude of the loss w.r.t. this gate
    is used as the importance score.

    Args:
        model: AureliusTransformer.
        input_ids: (B, T) input token ids.
        layer_idx: Index of the transformer layer to analyze.

    Returns:
        (n_heads,) importance scores -- higher = more important.
    """
    attn = model.layers[layer_idx].attn
    n_heads = attn.n_heads
    head_dim = attn.head_dim

    gates = torch.ones(n_heads, requires_grad=True, device=input_ids.device)

    def hook_fn(module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        B, S, d_model = output.shape
        out_reshaped = output.view(B, S, n_heads, head_dim)
        gated = out_reshaped * gates.view(1, 1, n_heads, 1)
        return gated.view(B, S, d_model)

    handle = attn.register_forward_hook(hook_fn)

    was_training = model.training
    model.train()

    try:
        model.zero_grad()
        labels = input_ids.clone()
        loss, _, _ = model(input_ids=input_ids, labels=labels)
        loss.backward()
    finally:
        handle.remove()
        if not was_training:
            model.eval()

    if gates.grad is not None:
        importance = gates.grad.abs().detach()
    else:
        importance = torch.zeros(n_heads, device=input_ids.device)

    return importance


def score_heads_globally(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: HeadPruningConfig,
) -> dict[int, torch.Tensor]:
    """Score every attention head in every layer.

    Args:
        model: AureliusTransformer.
        input_ids: (B, T) input token ids.
        config: HeadPruningConfig specifying the importance metric.

    Returns:
        Dict mapping layer_idx -> (n_heads,) importance scores.
        Higher score = more important head.
    """
    n_layers = len(model.layers)
    scores: dict[int, torch.Tensor] = {}

    if config.importance_metric == "entropy":
        from src.eval.attention_patterns import AttentionExtractor

        with torch.no_grad():
            with AttentionExtractor(model) as extractor:
                model(input_ids)
            patterns = extractor.patterns

        for pattern in patterns:
            lidx = pattern.layer_idx
            entropy = compute_head_entropy(pattern.weights.to(input_ids.device))
            # Invert: lower entropy = more important, so negate for "higher = more important"
            scores[lidx] = -entropy

    elif config.importance_metric == "sensitivity":
        for lidx in range(n_layers):
            scores[lidx] = compute_head_sensitivity(model, input_ids, lidx)

    else:
        # gradient-based importance
        was_training = model.training
        model.train()
        try:
            model.zero_grad()
            labels = input_ids.clone()
            loss, _, _ = model(input_ids=input_ids, labels=labels)
            loss.backward()

            for lidx, layer in enumerate(model.layers):
                attn = layer.attn
                n_heads = attn.n_heads
                head_dim = attn.head_dim

                if attn.o_proj.weight.grad is not None:
                    grad = attn.o_proj.weight.grad  # (d_model, n_heads * head_dim)
                    grad_reshaped = grad.view(grad.shape[0], n_heads, head_dim)
                    head_scores = grad_reshaped.abs().mean(dim=(0, 2))  # (n_heads,)
                else:
                    head_scores = torch.zeros(n_heads, device=input_ids.device)

                scores[lidx] = head_scores.detach()
        finally:
            model.zero_grad()
            if not was_training:
                model.eval()

    for lidx in range(n_layers):
        if lidx not in scores:
            n_heads = model.layers[lidx].attn.n_heads
            scores[lidx] = torch.zeros(n_heads, device=input_ids.device)

    return scores


def select_heads_to_prune(
    scores: dict[int, torch.Tensor],
    config: HeadPruningConfig,
) -> dict[int, list[int]]:
    """Select the lowest-scoring heads for pruning.

    Args:
        scores: Dict mapping layer_idx -> (n_heads,) importance scores.
        config: HeadPruningConfig with pruning parameters.

    Returns:
        Dict mapping layer_idx -> list of head indices to prune.
    """
    heads_to_prune: dict[int, list[int]] = {lidx: [] for lidx in scores}

    if config.global_pruning:
        all_heads: list[tuple[int, int, float]] = []
        for lidx, layer_scores in scores.items():
            for head_idx, score in enumerate(layer_scores.tolist()):
                all_heads.append((lidx, head_idx, score))

        total_heads = len(all_heads)
        n_prune = int(total_heads * config.prune_fraction)

        if n_prune == 0:
            return heads_to_prune

        all_heads_sorted = sorted(all_heads, key=lambda x: x[2])

        layer_head_counts: dict[int, int] = {
            lidx: len(layer_scores) for lidx, layer_scores in scores.items()
        }

        pruned = 0
        for lidx, head_idx, _score in all_heads_sorted:
            if pruned >= n_prune:
                break
            remaining = layer_head_counts[lidx]
            if remaining > config.min_heads_per_layer:
                heads_to_prune[lidx].append(head_idx)
                layer_head_counts[lidx] -= 1
                pruned += 1

    else:
        for lidx, layer_scores in scores.items():
            n_heads = len(layer_scores)
            n_prune_layer = int(n_heads * config.prune_fraction)
            max_prune = n_heads - config.min_heads_per_layer
            n_prune_layer = min(n_prune_layer, max_prune)

            if n_prune_layer <= 0:
                continue

            _, indices = torch.sort(layer_scores)
            heads_to_prune[lidx] = indices[:n_prune_layer].tolist()

    return heads_to_prune


class HeadMask(nn.Module):
    """Learnable binary mask over attention heads (straight-through estimator)."""

    def __init__(self, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.masks = nn.Parameter(torch.ones(n_layers, n_heads))

    def apply_mask(self, attention_output: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply head mask to attention output.

        Args:
            attention_output: (B, n_heads, T, head_dim)
            layer_idx: Index of the transformer layer.

        Returns:
            Masked attention output of the same shape.
        """
        mask = self.masks[layer_idx].view(1, self.n_heads, 1, 1)
        return attention_output * mask


def apply_head_pruning(
    model: nn.Module,
    heads_to_prune: dict[int, list[int]],
) -> nn.Module:
    """Zero out output projection weights for pruned attention heads.

    For each pruned head, the corresponding columns in o_proj (which maps
    concatenated head outputs back to d_model) are zeroed out.

    Args:
        model: AureliusTransformer (modified in-place).
        heads_to_prune: Dict mapping layer_idx -> list of head indices to prune.

    Returns:
        The model with pruned heads (in-place modification).
    """
    with torch.no_grad():
        for lidx, head_indices in heads_to_prune.items():
            if not head_indices:
                continue

            attn = model.layers[lidx].attn
            head_dim = attn.head_dim

            for head_idx in head_indices:
                col_start = head_idx * head_dim
                col_end = col_start + head_dim
                attn.o_proj.weight[:, col_start:col_end] = 0.0

    return model


def evaluate_pruning_impact(
    model: nn.Module,
    input_ids: torch.Tensor,
    heads_to_prune: dict[int, list[int]],
) -> dict[str, float]:
    """Evaluate the impact of head pruning on perplexity.

    Runs the model before and after pruning on input_ids.

    Args:
        model: AureliusTransformer.
        input_ids: (B, T) input token ids.
        heads_to_prune: Dict mapping layer_idx -> list of head indices to prune.

    Returns:
        Dict with keys:
            "perplexity_before": float
            "perplexity_after": float
            "n_heads_pruned": int (as float for dict uniformity)
    """
    was_training = model.training
    model.eval()

    labels = input_ids.clone()

    with torch.no_grad():
        loss_before, _, _ = model(input_ids=input_ids, labels=labels)
        perplexity_before = math.exp(loss_before.item())

    apply_head_pruning(model, heads_to_prune)

    with torch.no_grad():
        loss_after, _, _ = model(input_ids=input_ids, labels=labels)
        perplexity_after = math.exp(loss_after.item())

    if was_training:
        model.train()

    n_heads_pruned = sum(len(heads) for heads in heads_to_prune.values())

    return {
        "perplexity_before": perplexity_before,
        "perplexity_after": perplexity_after,
        "n_heads_pruned": float(n_heads_pruned),
    }
