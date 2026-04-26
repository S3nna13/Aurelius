"""Token importance scoring via gradient-based saliency, integrated gradients, and attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TokenImportanceConfig:
    """Configuration for token importance scoring."""

    method: str = "gradient"  # "gradient" | "integrated_gradients" | "attention"
    n_ig_steps: int = 20  # steps for integrated gradients
    ig_baseline: str = "zero"  # "zero" | "mean"
    aggregate: str = "l2"  # "l2" | "mean" | "max" over embedding dim
    normalize: bool = True  # normalize scores to sum to 1


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def aggregate_embeddings(emb_grad: Tensor, method: str) -> Tensor:
    """Aggregate gradient/embedding over embedding dimension.

    Args:
        emb_grad: Tensor of shape (B, T, D).
        method: One of "l2", "mean", "max".

    Returns:
        Tensor of shape (B, T).
    """
    if method == "l2":
        return emb_grad.norm(dim=-1)
    elif method == "mean":
        return emb_grad.abs().mean(dim=-1)
    elif method == "max":
        return emb_grad.abs().max(dim=-1).values
    else:
        raise ValueError(f"Unknown aggregate method: {method!r}. Choose 'l2', 'mean', or 'max'.")


def normalize_scores(scores: Tensor) -> Tensor:
    """Normalize each sequence's scores to sum to 1.

    Args:
        scores: Tensor of shape (B, T).

    Returns:
        Tensor of shape (B, T) where each row sums to 1.
        Rows that sum to zero become uniform distributions.
    """
    row_sums = scores.sum(dim=-1, keepdim=True)  # (B, 1)
    # Avoid division by zero: rows with zero sum become uniform
    zero_mask = row_sums.squeeze(-1) == 0  # (B,)
    safe_sums = row_sums.clone()
    safe_sums[safe_sums == 0] = 1.0
    normalized = scores / safe_sums

    # Replace zero-sum rows with uniform distribution
    if zero_mask.any():
        T = scores.shape[-1]
        uniform = torch.ones_like(scores) / T
        normalized[zero_mask] = uniform[zero_mask]

    return normalized


# ---------------------------------------------------------------------------
# Gradient saliency
# ---------------------------------------------------------------------------


def gradient_saliency(
    model: nn.Module,
    input_ids: Tensor,  # (B, T)
    target_position: int,  # which position's logit to differentiate
    target_token: int,  # which vocab token to score
) -> Tensor:
    """Compute gradient of logit[target_position, target_token] w.r.t. input embeddings.

    Uses a forward hook on model.embed to capture the embedding output with
    retain_grad(), then backpropagates and retrieves the gradient.

    Returns:
        Gradient tensor of shape (B, T, D).
    """
    model.eval()

    saved_emb: list[Tensor] = []

    def _forward_hook(module: nn.Module, inp: tuple, output: Tensor) -> None:
        output.retain_grad()
        saved_emb.clear()
        saved_emb.append(output)

    handle = model.embed.register_forward_hook(_forward_hook)

    try:
        # Run forward pass — hook saves embed output with retain_grad
        _, logits, _ = model(input_ids)

        # Get the target scalar: sum over batch dimension
        target_logit = logits[:, target_position, target_token].sum()

        # Backward pass
        target_logit.backward()

        # Retrieve gradient on embedding
        emb = saved_emb[0]
        if emb.grad is None:
            raise RuntimeError("Embedding gradient is None.")
        grad = emb.grad.clone()
    finally:
        handle.remove()

    return grad  # (B, T, D)


# ---------------------------------------------------------------------------
# Integrated gradients
# ---------------------------------------------------------------------------


def integrated_gradients(
    model: nn.Module,
    input_ids: Tensor,  # (B, T)
    target_position: int,
    target_token: int,
    n_steps: int = 20,
    baseline: str = "zero",  # "zero" | "mean"
) -> Tensor:
    """Integrated gradients approximation.

    Runs gradient_saliency at alpha=1.0 and scales by (embedding - baseline_embedding).

    Returns:
        IG attribution tensor of shape (B, T, D).
    """
    model.eval()

    # Get the actual embedding for the input
    with torch.no_grad():
        input_emb = model.embed(input_ids).detach()  # (B, T, D)

    # Construct baseline embedding
    if baseline == "zero":
        baseline_emb = torch.zeros_like(input_emb)
    elif baseline == "mean":
        mean_vec = model.embed.weight.detach().mean(dim=0)  # (D,)
        baseline_emb = mean_vec.unsqueeze(0).unsqueeze(0).expand_as(input_emb)
    else:
        raise ValueError(f"Unknown baseline: {baseline!r}. Choose 'zero' or 'mean'.")

    # Gradient at actual input
    grad = gradient_saliency(model, input_ids, target_position, target_token)  # (B, T, D)

    # IG attribution = grad * (input_emb - baseline_emb)
    ig_attr = grad * (input_emb - baseline_emb)

    return ig_attr  # (B, T, D)


# ---------------------------------------------------------------------------
# Attention importance
# ---------------------------------------------------------------------------


def attention_importance(
    model: nn.Module,
    input_ids: Tensor,  # (B, T)
    target_position: int,
) -> Tensor:
    """Extract attention weights to target_position across all layers/heads, average.

    Patches F.scaled_dot_product_attention to capture attention weights.
    Falls back to gradient_saliency if no weights are captured.

    Returns:
        Importance tensor of shape (B, T).
    """
    model.eval()

    import torch.nn.functional as _F

    import src.model.attention as _attn_module

    captured_attn: list[Tensor] = []
    original_sdpa = _F.scaled_dot_product_attention

    def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        # Compute attention weights manually
        B_, H_, S_q, D_ = query.shape
        S_kv = key.shape[2]
        sc = D_**-0.5
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * sc
        if is_causal and S_q > 1:
            causal_mask = torch.triu(
                torch.ones(S_q, S_kv, device=query.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if attn_mask is not None:
            scores = scores + attn_mask.float()
        weights = torch.softmax(scores, dim=-1)  # (B, H, S_q, S_kv)
        captured_attn.append(weights.detach())
        out = torch.matmul(weights, value.float()).to(query.dtype)
        return out

    # Patch both the module-level reference and the attention module's reference
    _F.scaled_dot_product_attention = _patched_sdpa
    _attn_module.F.scaled_dot_product_attention = _patched_sdpa  # type: ignore

    try:
        with torch.no_grad():
            _, logits, _ = model(input_ids)
    finally:
        _F.scaled_dot_product_attention = original_sdpa
        _attn_module.F.scaled_dot_product_attention = original_sdpa  # type: ignore

    T = input_ids.shape[1]

    if captured_attn:
        layer_scores: list[Tensor] = []
        for w in captured_attn:
            # w: (B, H, S_q, S_kv)
            if w.shape[2] > target_position:
                attn_slice = w[:, :, target_position, :T]  # (B, H, T) — truncate to T
                layer_scores.append(attn_slice.mean(dim=1))  # (B, T) — avg heads
        if layer_scores:
            return torch.stack(layer_scores, dim=0).mean(dim=0)  # (B, T)

    # Fallback: gradient saliency aggregated to (B, T)
    grad = gradient_saliency(model, input_ids, target_position, 0)  # (B, T, D)
    return aggregate_embeddings(grad, "l2")  # (B, T)


# ---------------------------------------------------------------------------
# Unified scorer class
# ---------------------------------------------------------------------------


class TokenImportanceScorer:
    """Unified interface for token importance scoring."""

    def __init__(self, model: nn.Module, cfg: TokenImportanceConfig) -> None:
        self.model = model
        self.cfg = cfg

    def score(self, input_ids: Tensor, target_position: int, target_token: int) -> Tensor:
        """Compute importance scores using configured method.

        Returns:
            Tensor of shape (B, T), optionally normalized so each row sums to 1.
        """
        if self.cfg.method == "gradient":
            raw = gradient_saliency(self.model, input_ids, target_position, target_token)
            scores = aggregate_embeddings(raw, self.cfg.aggregate)
        elif self.cfg.method == "integrated_gradients":
            raw = integrated_gradients(
                self.model,
                input_ids,
                target_position,
                target_token,
                n_steps=self.cfg.n_ig_steps,
                baseline=self.cfg.ig_baseline,
            )
            scores = aggregate_embeddings(raw, self.cfg.aggregate)
        elif self.cfg.method == "attention":
            scores = attention_importance(self.model, input_ids, target_position)
        else:
            raise ValueError(f"Unknown method: {self.cfg.method!r}")

        if self.cfg.normalize:
            scores = normalize_scores(scores)

        return scores  # (B, T)

    def top_k_tokens(
        self,
        input_ids: Tensor,
        target_position: int,
        target_token: int,
        k: int,
    ) -> Tensor:
        """Return indices of top-k most important tokens.

        Returns:
            Tensor of shape (B, k) with token indices.
        """
        scores = self.score(input_ids, target_position, target_token)  # (B, T)
        _, indices = scores.topk(k, dim=-1)  # (B, k)
        return indices

    def score_sequence(self, input_ids: Tensor) -> Tensor:
        """Score importance of each token for predicting the next token.

        For each position t in [1, T-1], uses position t-1 as target and the
        actual next token as target_token. Averages importance across all positions.

        Returns:
            Tensor of shape (B, T).
        """
        B, T = input_ids.shape
        total_scores = torch.zeros(B, T, device=input_ids.device)
        count = 0

        for t in range(1, T):
            target_pos = t - 1
            target_tok = int(input_ids[0, t].item())
            scores = self.score(input_ids, target_pos, target_tok)  # (B, T)
            total_scores += scores
            count += 1

        if count > 0:
            total_scores = total_scores / count

        return total_scores  # (B, T)
