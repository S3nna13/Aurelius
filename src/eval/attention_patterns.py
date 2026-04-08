"""Extract and analyze attention patterns from AureliusTransformer.

Provides hooks into GroupedQueryAttention layers to recompute attention
weights (which are not returned by F.scaled_dot_product_attention in the
flash attention path).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionPattern:
    """Captured attention weights for a single layer."""

    layer_idx: int
    weights: torch.Tensor  # (B, n_heads, S, S) attention weights (post-softmax)


class AttentionExtractor:
    """Extracts attention weights from all layers via forward hooks.

    Usage::

        with AttentionExtractor(model) as extractor:
            model(input_ids)
        patterns = extractor.patterns  # list[AttentionPattern]
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._patterns: list[AttentionPattern] = []
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def __enter__(self) -> AttentionExtractor:
        self._patterns.clear()
        self._register_hooks()
        return self

    def __exit__(self, *args: object) -> None:
        self._remove_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on all GroupedQueryAttention modules."""
        from src.model.attention import GroupedQueryAttention

        layer_idx = 0
        for _name, module in self.model.named_modules():
            if isinstance(module, GroupedQueryAttention):
                hook = module.register_forward_hook(self._make_hook(layer_idx))
                self._hooks.append(hook)
                layer_idx += 1

    def _make_hook(self, layer_idx: int):
        """Create a hook that recomputes attention weights from Q, K."""

        def hook_fn(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: object,
        ) -> None:
            from src.model.attention import apply_rope

            # GroupedQueryAttention.forward(x, freqs_cis, mask=None, past_kv=None)
            x = inputs[0]
            freqs_cis = inputs[1]
            B, S, _ = x.shape

            # Re-project Q and K (same projections as forward)
            q = module.q_proj(x).view(B, S, module.n_heads, module.head_dim)
            k = module.k_proj(x).view(B, S, module.n_kv_heads, module.head_dim)

            # Apply RoPE (critical for matching actual attention patterns)
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)

            # Transpose to (B, heads, S, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            # Expand KV heads for GQA
            if module.n_rep > 1:
                k = (
                    k.unsqueeze(2)
                    .expand(B, module.n_kv_heads, module.n_rep, S, module.head_dim)
                    .reshape(B, module.n_heads, S, module.head_dim)
                )

            # Compute attention weights manually
            scale = math.sqrt(module.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, S, S)

            # Apply causal mask: positions can only attend to earlier positions
            causal_mask = torch.triu(
                torch.full((S, S), float("-inf"), device=attn_weights.device),
                diagonal=1,
            )
            attn_weights = attn_weights + causal_mask

            attn_weights = F.softmax(attn_weights, dim=-1)

            self._patterns.append(
                AttentionPattern(
                    layer_idx=layer_idx,
                    weights=attn_weights.detach().cpu(),
                )
            )

        return hook_fn

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @property
    def patterns(self) -> list[AttentionPattern]:
        """Return captured attention patterns."""
        return self._patterns


def entropy_per_head(pattern: AttentionPattern, eps: float = 1e-9) -> torch.Tensor:
    """Compute entropy of each attention head's distribution.

    Args:
        pattern: AttentionPattern with weights of shape (B, H, S, S).
        eps: Small constant to avoid log(0).

    Returns:
        Tensor of shape (B, H, S) -- entropy per head per query position.
    """
    w = pattern.weights.clamp(min=eps)
    return -(w * w.log()).sum(dim=-1)  # (B, H, S)


def top_attended_positions(
    pattern: AttentionPattern,
    top_k: int = 3,
) -> torch.Tensor:
    """Return indices of top-k most attended positions per head per query.

    Args:
        pattern: AttentionPattern with weights of shape (B, H, S, S).
        top_k: Number of top positions to return.

    Returns:
        Tensor of shape (B, H, S, top_k) -- indices into the key dimension.
    """
    k = min(top_k, pattern.weights.shape[-1])
    return pattern.weights.topk(k, dim=-1).indices
