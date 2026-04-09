"""Advanced attention analysis: rollout, flow, entropy decomposition, and head redundancy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionAnalysisConfig:
    """Configuration for attention analysis."""

    n_heads: int = 8
    n_layers: int = 12
    rollout_discard_ratio: float = 0.0  # discard lowest attention weights per head before rollout


def compute_attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """Compute per-head per-token entropy of attention distributions.

    Args:
        attn_weights: (B, H, T, T) — softmax attention weights.

    Returns:
        (B, H, T) entropy per head per query token.
    """
    # Entropy: -sum(w * log(w + eps), dim=-1)
    entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1)
    return entropy


def attention_rollout(
    attn_matrices: list[torch.Tensor],
    discard_ratio: float = 0.0,
) -> torch.Tensor:
    """Compute attention rollout across layers.

    Attention rollout propagates attention through layers by matrix multiplication,
    accounting for residual connections.

    Args:
        attn_matrices: List of L tensors, each (B, H, T, T) — one per layer.
        discard_ratio: Fraction of lowest attention weights to zero out per row
            before averaging heads.

    Returns:
        (B, T, T) rollout matrix.
    """
    result = None

    for attn in attn_matrices:
        # attn: (B, H, T, T)
        if discard_ratio > 0.0:
            # Zero out lowest discard_ratio fraction per row per head
            B, H, T, T2 = attn.shape
            flat = attn.reshape(B * H * T, T2)
            k = int(discard_ratio * T2)
            if k > 0:
                # Find the k-th smallest threshold per row
                threshold, _ = flat.kthvalue(k, dim=-1, keepdim=True)
                mask = flat < threshold
                flat = flat.masked_fill(mask, 0.0)
                # Re-normalize rows that still have weight
                row_sum = flat.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                flat = flat / row_sum
            attn = flat.reshape(B, H, T, T2)

        # Average over heads: (B, T, T)
        avg = attn.mean(dim=1)

        # Add identity for residual: A_i = 0.5 * A_i + 0.5 * I
        B, T, _ = avg.shape
        eye = torch.eye(T, device=avg.device, dtype=avg.dtype).unsqueeze(0).expand(B, -1, -1)
        avg = 0.5 * avg + 0.5 * eye

        if result is None:
            result = avg
        else:
            result = torch.bmm(result, avg)

    if result is None:
        raise ValueError("attn_matrices must not be empty")

    return result


def compute_head_importance(
    attn_matrices: list[torch.Tensor],
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-head importance scores based on entropy variance.

    Heads with higher entropy variance across tokens are more focused
    (attend sharply to some tokens but not others) and thus more important.

    Args:
        attn_matrices: List of L tensors, each (B, H, T, T).
        labels: Unused; reserved for gradient-based importance methods.

    Returns:
        (L, H) importance scores — mean entropy std per layer-head.
    """
    importance_rows = []
    for attn in attn_matrices:
        # attn: (B, H, T, T)
        entropy = compute_attention_entropy(attn)  # (B, H, T)
        # Std over token dimension, then mean over batch
        std_over_tokens = entropy.std(dim=-1)  # (B, H)
        mean_std = std_over_tokens.mean(dim=0)  # (H,)
        importance_rows.append(mean_std)

    return torch.stack(importance_rows, dim=0)  # (L, H)


def detect_redundant_heads(
    importance: torch.Tensor,
    threshold: float = 0.1,
) -> list[tuple[int, int]]:
    """Identify heads with importance below the threshold.

    Args:
        importance: (L, H) importance scores from compute_head_importance.
        threshold: Heads with importance < threshold are considered redundant.

    Returns:
        List of (layer_idx, head_idx) tuples for redundant heads.
    """
    redundant: list[tuple[int, int]] = []
    L, H = importance.shape
    for layer in range(L):
        for head in range(H):
            if importance[layer, head].item() < threshold:
                redundant.append((layer, head))
    return redundant


def attention_flow(attn_matrices: list[torch.Tensor]) -> torch.Tensor:
    """Compute cumulative information flow across all layers.

    For each layer, average over heads and accumulate the attention matrices
    to show total flow from source to target tokens.

    Args:
        attn_matrices: List of L tensors, each (B, H, T, T).

    Returns:
        (B, T, T) total flow matrix.
    """
    if not attn_matrices:
        raise ValueError("attn_matrices must not be empty")

    flow = None
    for attn in attn_matrices:
        # Average heads: (B, T, T)
        avg = attn.mean(dim=1)
        if flow is None:
            flow = avg
        else:
            flow = flow + avg

    return flow


def extract_attention_maps(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    """Extract attention weights from specified layers via forward hooks.

    Registers hooks on model.layers[i].attn to recompute attention weights
    from Q and K projections (since SDPA doesn't return weights directly).

    Args:
        model: AureliusTransformer instance.
        input_ids: (B, T) token IDs.
        layer_indices: List of layer indices to capture attention from.

    Returns:
        Dict mapping layer_idx -> (B, H, T, T) attention weight tensor.
    """
    from src.model.attention import apply_rope

    captured: dict[int, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHook] = []

    def make_hook(layer_idx: int):
        def hook_fn(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: object,
        ) -> None:
            x = inputs[0]
            freqs_cis = inputs[1]
            B, S, _ = x.shape

            # Re-project Q and K to recompute attention weights
            q = module.q_proj(x).view(B, S, module.n_heads, module.head_dim)
            k = module.k_proj(x).view(B, S, module.n_kv_heads, module.head_dim)

            # Apply RoPE
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)

            # Transpose to (B, heads, S, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            # Expand KV heads for GQA if needed
            if module.n_rep > 1:
                k = (
                    k.unsqueeze(2)
                    .expand(B, module.n_kv_heads, module.n_rep, S, module.head_dim)
                    .reshape(B, module.n_heads, S, module.head_dim)
                )

            # Compute scaled dot-product attention weights
            scale = math.sqrt(module.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, S, S)

            # Apply causal mask
            causal_mask = torch.triu(
                torch.full((S, S), float("-inf"), device=attn_weights.device),
                diagonal=1,
            )
            attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1)

            captured[layer_idx] = attn_weights.detach()

        return hook_fn

    # Register hooks for requested layers
    for idx in layer_indices:
        attn_module = model.layers[idx].attn
        hook = attn_module.register_forward_hook(make_hook(idx))
        hooks.append(hook)

    try:
        with torch.no_grad():
            _, _, _ = model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return captured
