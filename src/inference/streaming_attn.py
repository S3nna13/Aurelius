"""StreamingLLM: attention sink tokens + sliding window KV cache for infinite-length generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class StreamingConfig:
    """Configuration for StreamingLLM attention sink + sliding window KV cache."""

    n_sink_tokens: int = 4
    window_size: int = 256
    max_cache_size: int = 260  # n_sink + window_size
    eviction_strategy: str = "sliding"  # "sliding" | "random"


class SinkTokenCache:
    """Rolling KV cache with fixed sink tokens + sliding window.

    Maintains a per-layer cache of shape (B, n_heads, T_cache, head_dim).
    The first n_sink_tokens positions in the cache are always the initial
    sink tokens; the remaining slots hold the most recent window_size tokens.
    """

    def __init__(
        self,
        config: StreamingConfig,
        n_layers: int,
        n_heads: int,
        head_dim: int,
    ) -> None:
        self.config = config
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim

        # Per-layer storage: list of (keys_buf, values_buf) or None when empty
        self._keys: list[Tensor | None] = [None] * n_layers
        self._values: list[Tensor | None] = [None] * n_layers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, new_keys: Tensor, new_values: Tensor, layer_idx: int
    ) -> tuple[Tensor, Tensor]:
        """Append new_keys / new_values to the cache and return the full cached tensors.

        Args:
            new_keys:   (B, n_heads, T_new, head_dim)
            new_values: (B, n_heads, T_new, head_dim)
            layer_idx:  which transformer layer this cache belongs to

        Returns:
            (cached_keys, cached_values): (B, n_heads, T_cache, head_dim)
        """
        n_sink = self.config.n_sink_tokens
        window = self.config.window_size
        max_size = self.config.max_cache_size

        existing_k = self._keys[layer_idx]
        existing_v = self._values[layer_idx]

        if existing_k is None:
            combined_k = new_keys
            combined_v = new_values
        else:
            combined_k = torch.cat([existing_k, new_keys], dim=2)
            combined_v = torch.cat([existing_v, new_values], dim=2)

        T_combined = combined_k.size(2)

        if T_combined <= max_size:
            self._keys[layer_idx] = combined_k
            self._values[layer_idx] = combined_v
        else:
            # Keep sink tokens + most recent window_size tokens
            sink_k = combined_k[:, :, :n_sink, :]
            sink_v = combined_v[:, :, :n_sink, :]
            recent_k = combined_k[:, :, -window:, :]
            recent_v = combined_v[:, :, -window:, :]
            self._keys[layer_idx] = torch.cat([sink_k, recent_k], dim=2)
            self._values[layer_idx] = torch.cat([sink_v, recent_v], dim=2)

        return self._keys[layer_idx], self._values[layer_idx]

    def get_cache_size(self) -> int:
        """Return the current number of cached tokens (from layer 0)."""
        if self._keys[0] is None:
            return 0
        return self._keys[0].size(2)

    def clear(self) -> None:
        """Reset all cached state."""
        self._keys = [None] * self.n_layers
        self._values = [None] * self.n_layers


# ---------------------------------------------------------------------------
# Position IDs
# ---------------------------------------------------------------------------


def compute_position_ids_for_cache(seq_len: int, cache_size: int, n_sink: int) -> Tensor:
    """Compute StreamingLLM-style position ids for the cached sequence.

    Sink tokens keep positions 0 .. n_sink-1.
    Recent tokens get consecutive positions immediately after the sink block,
    starting at n_sink.

    Args:
        seq_len:    total number of tokens seen so far (informational)
        cache_size: number of tokens currently in the cache (n_sink + recent)
        n_sink:     number of sink tokens

    Returns:
        Tensor of shape (cache_size,) with dtype torch.long
    """
    if cache_size <= n_sink:
        return torch.arange(cache_size, dtype=torch.long)

    sink_ids = torch.arange(n_sink, dtype=torch.long)
    n_recent = cache_size - n_sink
    recent_ids = torch.arange(n_sink, n_sink + n_recent, dtype=torch.long)
    return torch.cat([sink_ids, recent_ids])


# ---------------------------------------------------------------------------
# SinkAttention module
# ---------------------------------------------------------------------------


class SinkAttention(nn.Module):
    """Multi-head attention that uses a SinkTokenCache for incremental generation."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        config: StreamingConfig,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.config = config

        inner_dim = n_heads * head_dim

        self.q_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, d_model, bias=False)

        self.cache = SinkTokenCache(config, n_layers=1, n_heads=n_heads, head_dim=head_dim)

    def forward(self, x: Tensor, layer_idx: int = 0) -> Tensor:
        """Forward pass with sink attention cache.

        Args:
            x:         (B, T, d_model)
            layer_idx: ignored for standalone module (always uses cache slot 0)

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        def _split_heads(t: Tensor) -> Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = _split_heads(self.q_proj(x))
        k = _split_heads(self.k_proj(x))
        v = _split_heads(self.v_proj(x))

        cached_k, cached_v = self.cache.update(k, v, layer_idx=0)
        T_cache = cached_k.size(2)

        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, cached_k.transpose(-2, -1)) / scale

        if T > 1:
            q_idx = torch.arange(T, device=x.device)
            k_idx = torch.arange(T_cache, device=x.device)
            causal_mask = k_idx.unsqueeze(0) <= (T_cache - T + q_idx).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, cached_v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# Rolling generation
# ---------------------------------------------------------------------------


def rolling_generate(
    model: nn.Module,
    tokenizer_encode: Callable[[str], list[int]],
    tokenizer_decode: Callable[[list[int]], str],
    prompt: str,
    max_new: int,
    config: StreamingConfig,
) -> str:
    """Generate tokens autoregressively with sink attention cache.

    Uses greedy (argmax) decoding.  The model should accept (1, T) int tensor
    and return logits of shape (1, T, vocab_size) or a tuple whose second
    element is logits of that shape.

    Returns:
        Decoded string of newly generated tokens.
    """
    model.train(False)
    prompt_ids = tokenizer_encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    generated: list[int] = []

    with torch.no_grad():
        out = model(input_ids)
        logits = out[1] if isinstance(out, (tuple, list)) else out
        next_token = int(logits[0, -1].argmax().item())
        generated.append(next_token)

        for _ in range(max_new - 1):
            cur = torch.tensor([[next_token]], dtype=torch.long)
            out = model(cur)
            logits = out[1] if isinstance(out, (tuple, list)) else out
            next_token = int(logits[0, -1].argmax().item())
            generated.append(next_token)

    return tokenizer_decode(generated)


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------


def compute_cache_efficiency(config: StreamingConfig, seq_len: int) -> dict[str, float]:
    """Return efficiency statistics for a given sequence length.

    Keys:
        cache_size        -- tokens actually kept in cache
        full_cache_size   -- tokens that would be kept without compression (seq_len)
        compression_ratio -- cache_size / full_cache_size  (< 1 when compressing)
        sink_fraction     -- n_sink_tokens / max_cache_size
    """
    cache_size = min(seq_len, config.max_cache_size)
    full_cache_size = seq_len
    compression_ratio = cache_size / full_cache_size if full_cache_size > 0 else 1.0
    sink_fraction = config.n_sink_tokens / config.max_cache_size if config.max_cache_size > 0 else 0.0
    return {
        "cache_size": float(cache_size),
        "full_cache_size": float(full_cache_size),
        "compression_ratio": compression_ratio,
        "sink_fraction": sink_fraction,
    }
