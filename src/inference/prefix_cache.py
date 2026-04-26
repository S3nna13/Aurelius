"""Prefix caching: reuse KV cache across requests sharing common prefixes (system prompt caching)."""  # noqa: E501

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PrefixCacheConfig:
    """Configuration for prefix KV cache."""

    max_entries: int = 64
    max_prefix_len: int = 512
    min_prefix_len: int = 8
    eviction_policy: str = "lru"  # "lru" | "lfu"


@dataclass
class CacheEntry:
    """A single cached prefix entry."""

    prefix_ids: list[int]
    kv_cache: list[tuple[Tensor, Tensor]]  # one (K, V) per layer, shape (1, n_heads, T, head_dim)
    hit_count: int
    last_accessed: float
    created_at: float


def compute_prefix_hash(token_ids: list[int]) -> str:
    """SHA256 hash of token id sequence for cache key lookup."""
    data = b"".join(i.to_bytes(4, "little") for i in token_ids)
    return hashlib.sha256(data).hexdigest()


def find_longest_prefix_match(
    token_ids: list[int],
    cache: dict[str, CacheEntry],
) -> tuple[str | None, int]:
    """Find the longest cached prefix that matches the start of token_ids.

    Returns (cache_key, match_length) or (None, 0) if no match.
    """
    best_key: str | None = None
    best_len: int = 0

    for key, entry in cache.items():
        plen = len(entry.prefix_ids)
        if plen <= len(token_ids) and token_ids[:plen] == entry.prefix_ids:
            if plen > best_len:
                best_len = plen
                best_key = key

    return best_key, best_len


def truncate_kv_cache(
    kv_cache: list[tuple[Tensor, Tensor]],
    length: int,
) -> list[tuple[Tensor, Tensor]]:
    """Truncate KV cache to first `length` tokens.

    Each tensor has shape (1, H, T, D).
    Returns new list with tensors sliced to (1, H, length, D).
    """
    return [(k[:, :, :length, :], v[:, :, :length, :]) for k, v in kv_cache]


def merge_kv_caches(
    prefix_kv: list[tuple[Tensor, Tensor]],
    new_kv: list[tuple[Tensor, Tensor]],
) -> list[tuple[Tensor, Tensor]]:
    """Concatenate prefix_kv and new_kv along the sequence (T) dimension."""
    assert len(prefix_kv) == len(new_kv), "KV caches must have same number of layers"  # noqa: S101
    return [
        (torch.cat([pk, nk], dim=2), torch.cat([pv, nv], dim=2))
        for (pk, pv), (nk, nv) in zip(prefix_kv, new_kv)
    ]


class PrefixCache:
    """LRU/LFU cache for prefix KV states."""

    def __init__(self, cfg: PrefixCacheConfig) -> None:
        self.cfg = cfg
        self._cache: dict[str, CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    def get(self, token_ids: list[int]) -> tuple[list[tuple[Tensor, Tensor]] | None, int]:
        """Look up longest matching prefix.

        Returns (kv_cache, match_length).
        Updates hit_count and last_accessed. Returns (None, 0) on miss.
        """
        key, match_len = find_longest_prefix_match(token_ids, self._cache)
        if key is not None and match_len > 0:
            entry = self._cache[key]
            entry.hit_count += 1
            entry.last_accessed = time.time()
            self._hits += 1
            return entry.kv_cache, match_len
        self._misses += 1
        return None, 0

    def put(self, token_ids: list[int], kv_cache: list[tuple[Tensor, Tensor]]) -> None:
        """Store KV cache for token_ids.

        Evict if at capacity.
        Only cache if len(token_ids) >= min_prefix_len.
        Truncate to max_prefix_len if needed.
        """
        if len(token_ids) < self.cfg.min_prefix_len:
            return

        # Truncate if needed
        if len(token_ids) > self.cfg.max_prefix_len:
            token_ids = token_ids[: self.cfg.max_prefix_len]
            kv_cache = truncate_kv_cache(kv_cache, self.cfg.max_prefix_len)

        key = compute_prefix_hash(token_ids)

        if key in self._cache:
            # Update existing entry
            entry = self._cache[key]
            entry.kv_cache = kv_cache
            entry.last_accessed = time.time()
            return

        # Evict if at capacity
        if len(self._cache) >= self.cfg.max_entries:
            self._evict()

        now = time.time()
        self._cache[key] = CacheEntry(
            prefix_ids=list(token_ids),
            kv_cache=kv_cache,
            hit_count=0,
            last_accessed=now,
            created_at=now,
        )

    def _evict(self) -> None:
        """Evict one entry per eviction_policy.

        LRU: remove entry with smallest last_accessed
        LFU: remove entry with smallest hit_count (tiebreak: oldest)
        """
        if not self._cache:
            return

        if self.cfg.eviction_policy == "lfu":
            victim_key = min(
                self._cache,
                key=lambda k: (self._cache[k].hit_count, self._cache[k].created_at),
            )
        else:  # default: lru
            victim_key = min(
                self._cache,
                key=lambda k: self._cache[k].last_accessed,
            )

        del self._cache[victim_key]

    def clear(self) -> None:
        """Empty the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, int | float]:
        """Return {"size", "hits", "misses", "hit_rate"}."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }

    def __len__(self) -> int:
        return len(self._cache)


def simulate_prefix_caching(
    requests: list[list[int]],
    cache: PrefixCache,
    n_layers: int,
    n_heads: int,
    head_dim: int,
) -> dict[str, float]:
    """Simulate prefix cache hits/misses for a list of token requests.

    For each request:
      1. Look up prefix in cache
      2. Compute "savings" = match_length tokens don't need recomputing
      3. Create fake KV cache for the full request (random tensors)
      4. Store in cache

    Returns {"total_tokens", "reused_tokens", "reuse_fraction", "hit_rate"}
    """
    total_tokens = 0
    reused_tokens = 0

    for token_ids in requests:
        seq_len = len(token_ids)
        total_tokens += seq_len

        _kv, match_len = cache.get(token_ids)
        reused_tokens += match_len

        # Create fake KV cache for the full request
        fake_kv: list[tuple[Tensor, Tensor]] = [
            (
                torch.randn(1, n_heads, seq_len, head_dim),
                torch.randn(1, n_heads, seq_len, head_dim),
            )
            for _ in range(n_layers)
        ]
        cache.put(token_ids, fake_kv)

    reuse_fraction = reused_tokens / total_tokens if total_tokens > 0 else 0.0
    s = cache.stats()
    hit_rate = s["hit_rate"]

    return {
        "total_tokens": float(total_tokens),
        "reused_tokens": float(reused_tokens),
        "reuse_fraction": reuse_fraction,
        "hit_rate": hit_rate,
    }
