"""Adaptive KV cache management: eviction policies, memory budgeting, and prefill chunking."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KVCacheConfig:
    """Configuration for adaptive KV cache management."""

    max_seq_len: int = 2048
    eviction_policy: str = "lru"  # "lru" | "score" | "random"
    memory_budget_mb: float = 512.0
    prefill_chunk_size: int = 512
    n_layers: int = 24
    n_heads: int = 8
    head_dim: int = 64


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def compute_cache_memory_mb(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,
) -> float:
    """Return KV cache memory in MB for the given shape.

    Formula: 2 (K+V) * n_layers * n_heads * seq_len * head_dim * dtype_bytes / 1024^2
    """
    bytes_total = 2 * n_layers * n_heads * seq_len * head_dim * dtype_bytes
    return bytes_total / (1024**2)


def compute_max_seq_from_budget(
    budget_mb: float,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Return the maximum sequence length that fits within *budget_mb* MB.

    Inverts compute_cache_memory_mb:
        seq_len = budget_mb * 1024^2 / (2 * n_layers * n_heads * head_dim * dtype_bytes)
    """
    numerator = budget_mb * (1024**2)
    denominator = 2 * n_layers * n_heads * head_dim * dtype_bytes
    return int(numerator / denominator)


# ---------------------------------------------------------------------------
# LRU Cache
# ---------------------------------------------------------------------------


class LRUCache:
    """Pure-Python LRU cache tracking integer keys."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = capacity
        # OrderedDict: least-recently-used at the front, MRU at the back
        self._store: OrderedDict[int, None] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def access(self, key: int) -> None:
        """Mark *key* as the most recently used."""
        if key not in self._store:
            raise KeyError(f"key {key} not in cache")
        self._store.move_to_end(key)

    def insert(self, key: int) -> int | None:
        """Insert *key*; evict the LRU entry if at capacity.

        Returns the evicted key, or ``None`` if no eviction was needed.
        """
        if key in self._store:
            self._store.move_to_end(key)
            return None

        evicted: int | None = None
        if len(self._store) >= self.capacity:
            evicted = self.evict_lru()

        self._store[key] = None
        self._store.move_to_end(key)
        return evicted

    def evict_lru(self) -> int:
        """Evict and return the least-recently-used key."""
        if not self._store:
            raise RuntimeError("Cache is empty, nothing to evict")
        key, _ = self._store.popitem(last=False)
        return key

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: int) -> bool:
        return key in self._store


# ---------------------------------------------------------------------------
# Score-Based Cache
# ---------------------------------------------------------------------------


class ScoreBasedCache:
    """Cache that evicts the entry with the lowest score."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = capacity
        self._scores: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_score(self, key: int, score: float) -> None:
        """Update the score for *key* (key must already be present)."""
        self._scores[key] = score

    def insert(self, key: int, score: float = 0.0) -> int | None:
        """Insert *key* with *score*; evict the lowest-score entry if full.

        Returns the evicted key, or ``None`` if no eviction was needed.
        """
        if key in self._scores:
            self._scores[key] = score
            return None

        evicted: int | None = None
        if len(self._scores) >= self.capacity:
            evicted = self.evict_lowest()

        self._scores[key] = score
        return evicted

    def evict_lowest(self) -> int:
        """Evict and return the key with the lowest score."""
        if not self._scores:
            raise RuntimeError("Cache is empty, nothing to evict")
        key = min(self._scores, key=lambda k: self._scores[k])
        del self._scores[key]
        return key

    def __len__(self) -> int:
        return len(self._scores)


# ---------------------------------------------------------------------------
# Chunked prefill
# ---------------------------------------------------------------------------


def chunked_prefill(prompt_ids: Tensor, chunk_size: int) -> list[Tensor]:
    """Split *prompt_ids* of shape ``(1, T)`` into chunks of *chunk_size* along dim=1.

    The last chunk may be shorter than *chunk_size*.
    Returns a list of tensors each with shape ``(1, chunk_len)``.
    """
    T = prompt_ids.size(1)
    chunks: list[Tensor] = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunks.append(prompt_ids[:, start:end])
    return chunks


# ---------------------------------------------------------------------------
# KV Cache Manager
# ---------------------------------------------------------------------------


class KVCacheManager:
    """Manages memory allocation for multiple concurrent sequences."""

    def __init__(self, config: KVCacheConfig) -> None:
        self.config = config
        # Maps seq_id -> seq_len
        self._allocations: dict[int, int] = {}

    def allocate(self, seq_id: int, seq_len: int) -> bool:
        """Try to allocate memory for a sequence.

        Returns ``True`` if the allocation fits within the memory budget,
        ``False`` otherwise.  If the sequence was previously allocated it is
        updated to the new length (subject to the same budget check).
        """
        # Compute prospective usage
        existing_len = self._allocations.get(seq_id, 0)
        prospective_mb = (
            self.memory_used_mb()
            - compute_cache_memory_mb(
                self.config.n_layers,
                self.config.n_heads,
                self.config.head_dim,
                existing_len,
            )
            + compute_cache_memory_mb(
                self.config.n_layers,
                self.config.n_heads,
                self.config.head_dim,
                seq_len,
            )
        )

        if prospective_mb > self.config.memory_budget_mb:
            return False

        self._allocations[seq_id] = seq_len
        return True

    def free(self, seq_id: int) -> None:
        """Release the allocation for *seq_id*."""
        self._allocations.pop(seq_id, None)

    def memory_used_mb(self) -> float:
        """Return the total memory used by all current allocations in MB."""
        total = 0.0
        for seq_len in self._allocations.values():
            total += compute_cache_memory_mb(
                self.config.n_layers,
                self.config.n_heads,
                self.config.head_dim,
                seq_len,
            )
        return total

    def stats(self) -> dict:
        """Return a statistics dictionary.

        Keys: ``n_sequences``, ``memory_used_mb``, ``memory_budget_mb``,
        ``utilization``.
        """
        used = self.memory_used_mb()
        budget = self.config.memory_budget_mb
        utilization = used / budget if budget > 0 else 0.0
        return {
            "n_sequences": len(self._allocations),
            "memory_used_mb": used,
            "memory_budget_mb": budget,
            "utilization": utilization,
        }
