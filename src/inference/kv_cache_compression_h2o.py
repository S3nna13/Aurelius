"""KV cache compression via H2O (Heavy Hitter Oracle) eviction policy.

Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of Large Language Models", NeurIPS 2023 (arXiv:2306.14048).

Pure native PyTorch — no external dependencies beyond the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class KVCacheConfig:
    """Configuration for the H2O KV cache."""

    max_cache_size: int = 512
    """Maximum total number of cached KV positions."""

    n_heavy_hitters: int = 256
    """Number of heavy-hitter positions to keep after eviction."""

    n_recent: int = 64
    """Number of most-recent positions always kept (never evicted)."""

    eviction_interval: int = 16
    """How frequently (in tokens) eviction is triggered."""


# ---------------------------------------------------------------------------
# H2O KV cache
# ---------------------------------------------------------------------------


class H2OKVCache:
    """KV cache that evicts unimportant positions using accumulated attention
    scores, retaining only the top-*n_heavy_hitters* heavy hitters and the
    *n_recent* most recent positions.

    Tensor layout: ``(n_heads, seq_len, head_dim)`` — no batch dimension; one
    cache instance is expected per sequence.
    """

    def __init__(
        self,
        config: KVCacheConfig,
        n_heads: int,
        head_dim: int,
    ) -> None:
        self.config = config
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        new_keys: Tensor,
        new_values: Tensor,
        attn_weights: Tensor,
    ) -> None:
        """Append new KV pairs and accumulate attention scores.

        Args:
            new_keys:    ``(n_heads, t, head_dim)``
            new_values:  ``(n_heads, t, head_dim)``
            attn_weights: ``(n_heads, q, t)`` — attention weights *for the
                          new positions only* (i.e. the right-most *t* columns
                          of the full attention matrix for this step).
        """
        # Validate shapes.
        assert new_keys.shape == new_values.shape, (
            f"key/value shape mismatch: {new_keys.shape} vs {new_values.shape}"
        )
        assert new_keys.shape[0] == self.n_heads, (
            f"Expected {self.n_heads} heads, got {new_keys.shape[0]}"
        )

        # Accumulate importance: sum over the query dimension → (n_heads, t)
        new_scores: Tensor = attn_weights.sum(dim=-2)  # (n_heads, t)

        # Append to cache buffers.
        self.keys = torch.cat([self.keys, new_keys], dim=1)
        self.values = torch.cat([self.values, new_values], dim=1)
        self.attn_scores = torch.cat([self.attn_scores, new_scores], dim=1)

        # Evict if over budget.
        if self.keys.shape[1] > self.config.max_cache_size:
            self.evict()

    def evict(self) -> None:
        """Evict unimportant positions, keeping heavy hitters + recent tokens."""
        S = self.keys.shape[1]
        cfg = self.config

        n_recent = min(cfg.n_recent, S)
        n_hh = min(cfg.n_heavy_hitters, S - n_recent)

        if n_hh <= 0:
            # Nothing to evict beyond the recent window.
            return

        # Positions eligible to become heavy hitters (exclude the recent tail).
        eligible_end = S - n_recent
        # Average importance across heads for ranking.
        scores_avg = self.attn_scores[:, :eligible_end].mean(dim=0)  # (eligible_end,)

        _, top_indices = torch.topk(scores_avg, k=n_hh, largest=True, sorted=True)
        top_indices, _ = torch.sort(top_indices)  # restore temporal order

        recent_indices = torch.arange(
            S - n_recent, S, device=self.keys.device, dtype=torch.long
        )
        keep = torch.cat([top_indices, recent_indices])  # (n_hh + n_recent,)

        self.keys = self.keys[:, keep, :]
        self.values = self.values[:, keep, :]
        self.attn_scores = self.attn_scores[:, keep]

    def get(self) -> Tuple[Tensor, Tensor]:
        """Return ``(keys, values)`` tensors of shape ``(n_heads, S, head_dim)``."""
        return self.keys, self.values

    def size(self) -> int:
        """Current number of cached positions."""
        return self.keys.shape[1]

    def reset(self) -> None:
        """Clear all cached state."""
        self.keys: Tensor = torch.empty(self.n_heads, 0, self.head_dim)
        self.values: Tensor = torch.empty(self.n_heads, 0, self.head_dim)
        self.attn_scores: Tensor = torch.empty(self.n_heads, 0)


# ---------------------------------------------------------------------------
# Sliding-window baseline
# ---------------------------------------------------------------------------


class SlidingWindowKVCache:
    """Simple sliding-window KV cache: always keeps the *window_size* most
    recently seen positions.

    Tensor layout: ``(n_heads, seq_len, head_dim)``.
    """

    def __init__(self, window_size: int, n_heads: int, head_dim: int) -> None:
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self._keys: Tensor = torch.empty(n_heads, 0, head_dim)
        self._values: Tensor = torch.empty(n_heads, 0, head_dim)

    def update(self, new_keys: Tensor, new_values: Tensor) -> None:
        """Append and truncate to ``window_size`` (keeps most recent tokens).

        Args:
            new_keys:   ``(n_heads, t, head_dim)``
            new_values: ``(n_heads, t, head_dim)``
        """
        self._keys = torch.cat([self._keys, new_keys], dim=1)
        self._values = torch.cat([self._values, new_values], dim=1)

        if self._keys.shape[1] > self.window_size:
            self._keys = self._keys[:, -self.window_size:, :]
            self._values = self._values[:, -self.window_size:, :]

    def get(self) -> Tuple[Tensor, Tensor]:
        """Return ``(keys, values)`` of shape ``(n_heads, S, head_dim)``."""
        return self._keys, self._values

    def size(self) -> int:
        """Current number of cached positions."""
        return self._keys.shape[1]


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Tracks basic eviction statistics for a KV cache."""

    total_tokens_seen: int = 0
    tokens_evicted: int = 0
    current_size: int = 0

    @property
    def eviction_rate(self) -> float:
        """Fraction of seen tokens that were evicted (0.0 if none seen yet)."""
        if self.total_tokens_seen == 0:
            return 0.0
        return self.tokens_evicted / self.total_tokens_seen
