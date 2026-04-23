"""Streaming attention: sliding window + sink tokens (StreamingLLM, Xiao et al. 2309.17453)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class StreamingConfig:
    window_size: int = 512
    n_sink_tokens: int = 4
    max_cache_size: int = 520


class StreamingAttentionCache:
    """KV cache implementing the StreamingLLM eviction policy.

    The first n_sink_tokens are kept as attention sinks; when the cache
    exceeds max_cache_size the oldest non-sink tokens are evicted.
    """

    def __init__(self, config: StreamingConfig | None = None) -> None:
        self.config = config if config is not None else StreamingConfig()
        self._keys: list[Tensor] = []
        self._values: list[Tensor] = []

    # ------------------------------------------------------------------

    def add_token(self, key: Tensor, value: Tensor) -> None:
        """Append one token's key/value; evict oldest non-sink tokens if needed."""
        self._keys.append(key)
        self._values.append(value)

        cfg = self.config
        while len(self._keys) > cfg.max_cache_size:
            # Evict the oldest non-sink token (index = n_sink_tokens)
            evict_idx = cfg.n_sink_tokens
            if evict_idx >= len(self._keys):
                # All tokens are sinks; evict the oldest one (shouldn't happen in
                # normal usage but guard anyway)
                evict_idx = 0
            del self._keys[evict_idx]
            del self._values[evict_idx]

    def get_cache(self) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Return (stacked keys, stacked values), or (None, None) if empty."""
        if not self._keys:
            return None, None
        return torch.stack(self._keys, dim=0), torch.stack(self._values, dim=0)

    def __len__(self) -> int:
        return len(self._keys)

    def sink_count(self) -> int:
        """Number of sink tokens currently in the cache."""
        return min(self.config.n_sink_tokens, len(self._keys))


# ---------------------------------------------------------------------------
# Attention computation
# ---------------------------------------------------------------------------


def compute_streaming_attention(
    query: Tensor,
    cache: StreamingAttentionCache,
    scale: float | None = None,
) -> Tensor:
    """Compute scaled dot-product attention against the streaming KV cache.

    Args:
        query: (batch, heads, head_dim) or (batch, heads, 1, head_dim)
        cache: StreamingAttentionCache holding K/V history
        scale: optional scale factor; defaults to 1/sqrt(head_dim)

    Returns:
        Tensor with same shape as query.
    """
    K, V = cache.get_cache()

    if K is None:
        return torch.zeros_like(query)

    original_shape = query.shape
    # Normalise to 4-D: (batch, heads, q_len, head_dim)
    if query.dim() == 3:
        q = query.unsqueeze(2)  # (B, H, 1, D)
    else:
        q = query  # (B, H, L, D)

    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # K/V stacked along dim=0 → (cache_len, *token_shape)
    # Reshape so K is (cache_len, head_dim) for simplicity then broadcast.
    # Handle arbitrary token shapes by flattening non-head dims.
    # K shape: (cache_len, ...) — we need (cache_len, head_dim)
    cache_len = K.shape[0]
    k_flat = K.reshape(cache_len, -1)   # (cache_len, head_dim)
    v_flat = V.reshape(cache_len, -1)   # (cache_len, head_dim)

    # q: (B, H, L, D) — compute for each (B, H, L) independently
    B, H, L, D = q.shape

    # scores: (B, H, L, cache_len)
    scores = torch.einsum("bhld,cd->bhlc", q, k_flat) * scale
    attn = torch.softmax(scores, dim=-1)   # (B, H, L, cache_len)

    # out: (B, H, L, D)
    out = torch.einsum("bhlc,cd->bhld", attn, v_flat)

    # Restore original shape
    if query.dim() == 3:
        out = out.squeeze(2)  # (B, H, D)

    return out
