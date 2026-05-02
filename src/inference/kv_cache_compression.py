"""KV cache compression with token eviction for long-context efficiency.

Implements attention-sink, recent-only, heavy-hitter, and random eviction
strategies following StreamingLLM and H2O findings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class KVCacheConfig:
    """Configuration for KV cache compression."""

    max_cache_size: int = 512
    eviction_strategy: str = (
        "attention_sink"  # "attention_sink" | "recent" | "random" | "heavy_hitter"
    )
    sink_tokens: int = 4
    recent_tokens: int = 256
    heavy_hitter_ratio: float = 0.2


# ---------------------------------------------------------------------------
# Eviction functions
# ---------------------------------------------------------------------------


def evict_attention_sink(
    keys: Tensor,
    values: Tensor,
    config: KVCacheConfig,
) -> tuple[Tensor, Tensor]:
    """Keep first sink_tokens + last (max_cache_size - sink_tokens) tokens.

    Args:
        keys:   (B, n_heads, T, head_dim)
        values: (B, n_heads, T, head_dim)
        config: KVCacheConfig

    Returns:
        (evicted_keys, evicted_values)
    """
    T = keys.shape[2]
    sink = config.sink_tokens
    recent_budget = config.max_cache_size - sink

    if T <= config.max_cache_size:
        return keys, values

    # Always keep first sink_tokens; fill the rest with the most recent tokens
    recent_start = max(sink, T - recent_budget)
    sink_idx = torch.arange(sink, device=keys.device)
    recent_idx = torch.arange(recent_start, T, device=keys.device)
    keep = torch.cat([sink_idx, recent_idx])
    keep = torch.unique(keep, sorted=True)

    return keys[:, :, keep, :], values[:, :, keep, :]


def evict_recent_only(
    keys: Tensor,
    values: Tensor,
    config: KVCacheConfig,
) -> tuple[Tensor, Tensor]:
    """Keep only last max_cache_size tokens.

    Args:
        keys:   (B, n_heads, T, head_dim)
        values: (B, n_heads, T, head_dim)
        config: KVCacheConfig

    Returns:
        (evicted_keys, evicted_values)
    """
    T = keys.shape[2]
    if T <= config.max_cache_size:
        return keys, values

    start = T - config.max_cache_size
    return keys[:, :, start:, :], values[:, :, start:, :]


def evict_heavy_hitter(
    keys: Tensor,
    values: Tensor,
    attention_weights: Tensor | None,
    config: KVCacheConfig,
) -> tuple[Tensor, Tensor]:
    """Keep top-k tokens by mean attention score plus sink tokens.

    Args:
        keys:              (B, n_heads, T, head_dim)
        values:            (B, n_heads, T, head_dim)
        attention_weights: (B, n_heads, T) or None
        config:            KVCacheConfig

    Returns:
        (evicted_keys, evicted_values)
    """
    T = keys.shape[2]
    if T <= config.max_cache_size:
        return keys, values

    if attention_weights is None:
        return evict_recent_only(keys, values, config)

    sink = config.sink_tokens
    # Mean attention score across batch and heads -> (T,)
    # attention_weights shape: (B, n_heads, T)
    token_scores = attention_weights.mean(dim=(0, 1))  # (T,)

    n_to_keep = config.max_cache_size - sink
    non_sink_scores = token_scores[sink:]  # (T - sink,)
    n_available = non_sink_scores.shape[0]
    n_to_keep = min(n_to_keep, n_available)

    sink_idx = torch.arange(sink, device=keys.device)

    if n_to_keep > 0:
        _, top_local = torch.topk(non_sink_scores, k=n_to_keep, largest=True, sorted=False)
        top_global = top_local + sink
        top_global_sorted, _ = torch.sort(top_global)
        keep = torch.cat([sink_idx, top_global_sorted])
    else:
        keep = sink_idx

    keep = torch.unique(keep, sorted=True)
    return keys[:, :, keep, :], values[:, :, keep, :]


def _evict_random(
    keys: Tensor,
    values: Tensor,
    config: KVCacheConfig,
) -> tuple[Tensor, Tensor]:
    """Keep sink_tokens + random sample of remaining up to max_cache_size."""
    T = keys.shape[2]
    if T <= config.max_cache_size:
        return keys, values

    sink = config.sink_tokens
    n_to_keep = config.max_cache_size - sink
    non_sink_pool = torch.arange(sink, T, device=keys.device)
    perm = torch.randperm(non_sink_pool.shape[0], device=keys.device)
    chosen = non_sink_pool[perm[:n_to_keep]]
    sink_idx = torch.arange(sink, device=keys.device)
    keep, _ = torch.sort(torch.cat([sink_idx, chosen]))
    keep = torch.unique(keep, sorted=True)
    return keys[:, :, keep, :], values[:, :, keep, :]


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------


class KVCache:
    """Stores key-value pairs per transformer layer with token eviction."""

    def __init__(
        self,
        config: KVCacheConfig,
        n_layers: int,
        n_heads: int,
        head_dim: int,
    ) -> None:
        self.config = config
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.keys: list[Tensor | None] = [None] * n_layers
        self.values: list[Tensor | None] = [None] * n_layers

    def update(
        self,
        layer_idx: int,
        new_keys: Tensor,
        new_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Append new_keys/new_values and evict if over max_cache_size.

        Args:
            layer_idx: which layer to update
            new_keys:  (B, n_heads, T_new, head_dim)
            new_values:(B, n_heads, T_new, head_dim)

        Returns:
            (full_keys, full_values) after potential eviction
        """
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = new_keys
            self.values[layer_idx] = new_values
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], new_keys], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], new_values], dim=2)

        k = self.keys[layer_idx]
        v = self.values[layer_idx]

        if k.shape[2] > self.config.max_cache_size:
            k, v = self._evict(k, v)
            self.keys[layer_idx] = k
            self.values[layer_idx] = v

        return k, v

    def _evict(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        strategy = self.config.eviction_strategy
        if strategy == "attention_sink":
            return evict_attention_sink(keys, values, self.config)
        elif strategy == "recent":
            return evict_recent_only(keys, values, self.config)
        elif strategy == "heavy_hitter":
            return evict_heavy_hitter(keys, values, None, self.config)
        elif strategy == "random":
            return _evict_random(keys, values, self.config)
        else:
            raise ValueError(f"Unknown eviction_strategy: {strategy!r}")

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        """Return (keys, values) for layer, or None if not yet populated."""
        k = self.keys[layer_idx]
        v = self.values[layer_idx]
        if k is None:
            return None
        return k, v

    def clear(self) -> None:
        """Reset all cached keys/values to empty."""
        self.keys = [None] * self.n_layers
        self.values = [None] * self.n_layers

    def __len__(self) -> int:
        """Current sequence length (from layer 0), or 0 if empty."""
        if self.keys[0] is None:
            return 0
        return self.keys[0].shape[2]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class CacheCompressionStats:
    """Statistics for cache compression/eviction."""

    n_evictions: int = 0
    total_tokens_evicted: int = 0
    compression_ratio: float = 0.0


# ---------------------------------------------------------------------------
# CompressedKVCache — extends KVCache with eviction tracking
# ---------------------------------------------------------------------------


class CompressedKVCache(KVCache):
    """KVCache subclass that tracks eviction statistics."""

    def __init__(
        self,
        config: KVCacheConfig,
        n_layers: int,
        n_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__(config, n_layers, n_heads, head_dim)
        self._stats = CacheCompressionStats()
        self._total_tokens_seen: int = 0

    def update(
        self,
        layer_idx: int,
        new_keys: Tensor,
        new_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Update cache and track eviction statistics."""
        # Size before append
        before_size = 0 if self.keys[layer_idx] is None else self.keys[layer_idx].shape[2]
        tokens_added = new_keys.shape[2]

        # Use parent update
        k, v = super().update(layer_idx, new_keys, new_values)

        after_size = k.shape[2]
        expected_size = before_size + tokens_added

        if after_size < expected_size:
            tokens_evicted = expected_size - after_size
            self._stats.n_evictions += 1
            self._stats.total_tokens_evicted += tokens_evicted

        # Track total tokens seen (only from layer 0 to avoid double-counting)
        if layer_idx == 0:
            self._total_tokens_seen += tokens_added
            if self._total_tokens_seen > 0:
                self._stats.compression_ratio = after_size / self._total_tokens_seen

        return k, v

    def get_stats(self) -> CacheCompressionStats:
        """Return a copy of the current compression statistics."""
        return CacheCompressionStats(
            n_evictions=self._stats.n_evictions,
            total_tokens_evicted=self._stats.total_tokens_evicted,
            compression_ratio=self._stats.compression_ratio,
        )
