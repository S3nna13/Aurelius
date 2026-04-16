"""Prefix caching v2: reuse KV states for shared prompt prefixes across requests.

Reduces repeated computation by caching and retrieving key-value states for
common token-id prefixes, supporting LRU, LFU, and FIFO eviction policies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CacheConfig:
    """Configuration for the prefix KV cache."""

    max_entries: int = 128
    max_prefix_len: int = 512
    d_model: int = 512
    n_layers: int = 12
    eviction_policy: str = "lru"  # one of "lru" | "lfu" | "fifo"

    def __post_init__(self) -> None:
        valid_policies = {"lru", "lfu", "fifo"}
        if self.eviction_policy not in valid_policies:
            raise ValueError(
                f"eviction_policy must be one of {valid_policies}, "
                f"got {self.eviction_policy!r}"
            )


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

class CacheKey:
    """Hashable wrapper around a tuple of token ids."""

    __slots__ = ("_ids",)

    def __init__(self, token_ids: Tuple[int, ...]) -> None:
        self._ids: Tuple[int, ...] = tuple(token_ids)

    # expose the underlying tuple for iteration / slicing in helpers
    @property
    def ids(self) -> Tuple[int, ...]:
        return self._ids

    def __hash__(self) -> int:
        return hash(self._ids)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CacheKey):
            return self._ids == other._ids
        return NotImplemented

    def __repr__(self) -> str:  # pragma: no cover
        return f"CacheKey({self._ids!r})"

    def __len__(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """One cached prefix entry."""

    key: CacheKey
    kv_states: List[Tensor]          # one tensor per layer
    hit_count: int = 0
    timestamp: float = 0.0           # creation time (or last-access for LRU)


# ---------------------------------------------------------------------------
# Main cache
# ---------------------------------------------------------------------------

class PrefixCache:
    """Cache that maps token-id prefixes to pre-computed KV states."""

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self._store: Dict[CacheKey, CacheEntry] = {}
        self._total_hits: int = 0
        self._total_lookups: int = 0
        # FIFO: we track insertion order via a list of keys
        self._insertion_order: List[CacheKey] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self, token_ids: List[int]
    ) -> Optional[Tuple[int, List[Tensor]]]:
        """Find the longest cached prefix of *token_ids*.

        Returns ``(prefix_len, kv_states)`` on a hit, or ``None`` on a miss.
        Updates ``hit_count`` and ``timestamp`` of the matched entry.
        """
        self._total_lookups += 1

        best_key: Optional[CacheKey] = None
        best_len: int = 0

        for key, entry in self._store.items():
            plen = len(key)
            if plen <= len(token_ids) and list(key.ids) == token_ids[:plen]:
                if plen > best_len:
                    best_len = plen
                    best_key = key

        if best_key is None or best_len == 0:
            return None

        entry = self._store[best_key]
        entry.hit_count += 1
        entry.timestamp = time.monotonic()  # refresh for LRU
        self._total_hits += 1
        return best_len, entry.kv_states

    def store(self, token_ids: List[int], kv_states: List[Tensor]) -> None:
        """Store *kv_states* keyed by *token_ids*.

        Silently ignores sequences longer than ``max_prefix_len``.
        Evicts one entry when the cache is at capacity.
        """
        if len(token_ids) > self.config.max_prefix_len:
            return

        key = CacheKey(tuple(token_ids))

        # Update in place if already present (refresh timestamp)
        if key in self._store:
            self._store[key].kv_states = kv_states
            self._store[key].timestamp = time.monotonic()
            return

        # Evict before inserting if at capacity
        if len(self._store) >= self.config.max_entries:
            self.evict()

        entry = CacheEntry(
            key=key,
            kv_states=kv_states,
            hit_count=0,
            timestamp=time.monotonic(),
        )
        self._store[key] = entry
        self._insertion_order.append(key)

    def evict(self) -> None:
        """Remove one entry according to the configured eviction policy.

        * **lru**  – remove the entry with the smallest (oldest) ``timestamp``
        * **lfu**  – remove the entry with the lowest ``hit_count``
          (ties broken by oldest timestamp)
        * **fifo** – remove the entry that was inserted first
        """
        if not self._store:
            return

        policy = self.config.eviction_policy

        if policy == "lru":
            victim = min(self._store.values(), key=lambda e: e.timestamp)
        elif policy == "lfu":
            victim = min(
                self._store.values(),
                key=lambda e: (e.hit_count, e.timestamp),
            )
        else:  # fifo
            # Walk insertion_order list to find the first key still in store
            while self._insertion_order:
                candidate = self._insertion_order[0]
                if candidate in self._store:
                    victim = self._store[candidate]
                    break
                self._insertion_order.pop(0)
            else:
                return  # nothing to evict

        self._insertion_order = [
            k for k in self._insertion_order if k != victim.key
        ]
        del self._store[victim.key]

    def hit_rate(self) -> float:
        """Return ``total_hits / total_lookups``, or ``0.0`` if no lookups."""
        if self._total_lookups == 0:
            return 0.0
        return self._total_hits / self._total_lookups

    def size(self) -> int:
        """Return the number of cached entries."""
        return len(self._store)

    def clear(self) -> None:
        """Remove all cached entries and reset statistics."""
        self._store.clear()
        self._insertion_order.clear()
        self._total_hits = 0
        self._total_lookups = 0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_prefix_savings(
    original_len: int, cached_prefix_len: int, n_layers: int
) -> float:
    """Fraction of KV computation saved by reusing a cached prefix.

    ``savings = (cached_prefix_len * n_layers) / (original_len * n_layers)``

    Returns a float in ``[0, 1]``. Returns ``0.0`` when *original_len* is 0.
    """
    if original_len <= 0:
        return 0.0
    savings = (cached_prefix_len * n_layers) / (original_len * n_layers)
    return float(max(0.0, min(1.0, savings)))


def find_common_prefix(sequences: List[List[int]]) -> List[int]:
    """Return the longest common prefix shared by **all** *sequences*.

    Returns an empty list when *sequences* is empty or no common prefix exists.
    """
    if not sequences:
        return []

    prefix: List[int] = []
    for tokens in zip(*sequences):
        if len(set(tokens)) == 1:
            prefix.append(tokens[0])
        else:
            break
    return prefix
