"""
KV-cache eviction strategies for bounded-memory inference.

Implements StreamingLLM (sink + recent window), Heavy-Hitter Oracle (H2O),
a unified KVCacheManager, and utility functions for cache analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvictionConfig:
    """Configuration for KV-cache eviction policies."""

    max_cache_size: int = 512
    n_sink_tokens: int = 4
    n_recent_tokens: int = 64
    eviction_strategy: str = "heavy_hitter"  # "streaming_llm" | "heavy_hitter"
    score_decay: float = 0.9


# ---------------------------------------------------------------------------
# StreamingLLM eviction
# ---------------------------------------------------------------------------


class StreamingLLMEviction:
    """Always keep the first *n_sink* tokens and the last *n_recent* tokens.

    This follows the StreamingLLM paper: attention sinks (early tokens)
    receive disproportionately high attention and must be retained so that
    the model remains stable; recent tokens are retained for local context.
    """

    def __init__(self, config: EvictionConfig) -> None:
        self.config = config

    def __call__(
        self,
        keys: Tensor,
        values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evict tokens outside the sink + recent window.

        Args:
            keys:   (B, H, T, d_head)
            values: (B, H, T, d_head)

        Returns:
            Tuple of (evicted_keys, evicted_values) with shape
            (B, H, T', d_head) where T' <= max_cache_size.
        """
        T = keys.size(2)
        if T <= self.config.max_cache_size:
            return keys, values

        n_sink = self.config.n_sink_tokens
        n_recent = self.config.n_recent_tokens

        sink_keys = keys[:, :, :n_sink, :]
        sink_vals = values[:, :, :n_sink, :]

        recent_keys = keys[:, :, -n_recent:, :]
        recent_vals = values[:, :, -n_recent:, :]

        evicted_keys = torch.cat([sink_keys, recent_keys], dim=2)
        evicted_vals = torch.cat([sink_vals, recent_vals], dim=2)

        return evicted_keys, evicted_vals


# ---------------------------------------------------------------------------
# Heavy-Hitter Oracle (H2O) eviction
# ---------------------------------------------------------------------------


class HeavyHitterEviction:
    """Eviction based on cumulative attention scores (H2O policy).

    Tokens that receive the most attention across all heads/steps are
    considered "heavy hitters" and are kept.  Sink tokens are always
    preserved regardless of their score.
    """

    def __init__(self, config: EvictionConfig) -> None:
        self.config = config
        self.scores: Tensor | None = None  # (T,)

    # ------------------------------------------------------------------
    def update_scores(self, attn_weights: Tensor) -> None:
        """Accumulate attention received by each key position.

        Args:
            attn_weights: (B, H, T_q, T_k) — raw or softmax attention weights.
        """
        # Sum attention received by each key position across batch, heads, queries
        # Result shape: (T_k,)
        received = attn_weights.mean(dim=(0, 1)).sum(dim=0)  # (T_k,)

        if self.scores is None:
            self.scores = received.detach().clone()
        else:
            T_k = received.size(0)
            T_s = self.scores.size(0)

            if T_k == T_s:
                self.scores = self.scores * self.config.score_decay + received.detach()
            elif T_k > T_s:
                # Cache grew — extend scores with zeros for new positions
                pad = torch.zeros(T_k - T_s, device=received.device, dtype=received.dtype)
                self.scores = torch.cat([self.scores * self.config.score_decay, pad], dim=0)
                self.scores = self.scores + received.detach()
            else:
                # Cache shrank (after previous eviction) — truncate
                self.scores = self.scores[:T_k] * self.config.score_decay + received.detach()

    # ------------------------------------------------------------------
    def evict(
        self,
        keys: Tensor,
        values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return keys/values keeping top-scored positions + sink tokens.

        Args:
            keys:   (B, H, T, d_head)
            values: (B, H, T, d_head)

        Returns:
            Tuple of (evicted_keys, evicted_values).
        """
        T = keys.size(2)
        budget = self.config.max_cache_size
        n_sink = self.config.n_sink_tokens

        if T <= budget:
            return keys, values

        if self.scores is not None and self.scores.size(0) >= T:
            scores = self.scores[:T]
        else:
            # Fall back to uniform scores when no history is available
            scores = torch.ones(T, device=keys.device)

        # Force-keep sink tokens by setting their scores to +inf
        boosted = scores.clone().float()
        boosted[:n_sink] = float("inf")

        # Pick top-budget indices and sort them to preserve order
        keep_indices = torch.topk(boosted, k=min(budget, T), largest=True).indices
        keep_indices, _ = keep_indices.sort()

        evicted_keys = keys[:, :, keep_indices, :]
        evicted_vals = values[:, :, keep_indices, :]

        # Trim accumulated scores to match the new cache layout
        if self.scores is not None:
            self.scores = self.scores[keep_indices]

        return evicted_keys, evicted_vals

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear accumulated attention scores."""
        self.scores = None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compute_cache_hit_rate(requested: list[int], cached: list[int]) -> float:
    """Fraction of requested token positions found in the cache.

    Args:
        requested: List of token position ids that were requested.
        cached:    List of token position ids currently in the cache.

    Returns:
        Hit rate in [0.0, 1.0].  Returns 0.0 if *requested* is empty.
    """
    if not requested:
        return 0.0
    cached_set = set(cached)
    hits = sum(1 for pos in requested if pos in cached_set)
    return hits / len(requested)


def estimate_cache_memory(
    max_size: int,
    n_layers: int,
    n_heads: int,
    d_head: int,
    dtype_bytes: int = 2,
) -> int:
    """Estimate total KV-cache memory in bytes.

    Formula: 2 * max_size * n_layers * n_heads * d_head * dtype_bytes
    The factor 2 accounts for both K and V tensors.

    Args:
        max_size:    Maximum number of cached tokens.
        n_layers:    Number of transformer layers.
        n_heads:     Number of attention heads per layer.
        d_head:      Dimension of each attention head.
        dtype_bytes: Bytes per element (default 2 for fp16/bf16).

    Returns:
        Total memory in bytes as an integer.
    """
    return 2 * max_size * n_layers * n_heads * d_head * dtype_bytes


# ---------------------------------------------------------------------------
# KVCacheManager
# ---------------------------------------------------------------------------


class KVCacheManager:
    """Manages a rolling KV-cache using the StreamingLLM eviction policy.

    Usage::

        manager = KVCacheManager(config)
        keys, values = manager.update(new_keys, new_values)
        print(manager.size())
        manager.reset()
    """

    def __init__(self, config: EvictionConfig) -> None:
        self.config = config
        self._eviction = StreamingLLMEviction(config)
        self._keys: Tensor | None = None
        self._values: Tensor | None = None

    # ------------------------------------------------------------------
    def update(
        self,
        new_keys: Tensor,
        new_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Append new K/V pairs and evict if over budget.

        Args:
            new_keys:   (B, H, T_new, d_head)
            new_values: (B, H, T_new, d_head)

        Returns:
            Current (keys, values) after potential eviction.
        """
        if self._keys is None:
            self._keys = new_keys
            self._values = new_values
        else:
            self._keys = torch.cat([self._keys, new_keys], dim=2)
            self._values = torch.cat([self._values, new_values], dim=2)

        # Evict if over budget
        self._keys, self._values = self._eviction(self._keys, self._values)

        return self._keys, self._values

    # ------------------------------------------------------------------
    def size(self) -> int:
        """Return the current number of cached tokens."""
        if self._keys is None:
            return 0
        return self._keys.size(2)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the entire cache."""
        self._keys = None
        self._values = None
