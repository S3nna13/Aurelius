"""KV cache compression: token eviction, quantization, streaming compression."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

_rng = random.Random(42)


class EvictionPolicy(str, Enum):
    RECENCY = "recency"
    ATTENTION_SCORE = "attention_score"
    RANDOM = "random"
    H2O = "h2o"


@dataclass
class KVEntry:
    token_id: int
    key: list[float]
    value: list[float]
    attention_score: float = 0.0
    position: int = 0


class KVCacheCompressor:
    def __init__(
        self,
        max_tokens: int = 512,
        policy: EvictionPolicy = EvictionPolicy.ATTENTION_SCORE,
        head_dim: int = 64,
    ) -> None:
        self.max_tokens = max_tokens
        self.policy = policy
        self.head_dim = head_dim
        self._cache: list[KVEntry] = []

    def add(self, entry: KVEntry) -> None:
        self._cache.append(entry)
        if len(self._cache) > self.max_tokens:
            overflow = len(self._cache) - self.max_tokens
            self._evict(overflow)

    def _evict(self, n: int = 1) -> list[KVEntry]:
        n = min(n, len(self._cache))
        if n <= 0:
            return []

        if self.policy == EvictionPolicy.RECENCY:
            evicted = self._cache[:n]
            self._cache = self._cache[n:]

        elif self.policy == EvictionPolicy.ATTENTION_SCORE:
            indexed = sorted(enumerate(self._cache), key=lambda x: x[1].attention_score)
            evict_indices = set(idx for idx, _ in indexed[:n])
            evicted = [self._cache[i] for i in sorted(evict_indices)]
            self._cache = [e for i, e in enumerate(self._cache) if i not in evict_indices]

        elif self.policy == EvictionPolicy.RANDOM:
            indices = list(range(len(self._cache)))
            _rng.shuffle(indices)
            evict_indices = set(indices[:n])
            evicted = [self._cache[i] for i in sorted(evict_indices)]
            self._cache = [e for i, e in enumerate(self._cache) if i not in evict_indices]

        elif self.policy == EvictionPolicy.H2O:
            total = len(self._cache)
            top_half_count = max(1, total // 2)
            # Sort by attention score descending; heavy hitters are top half
            indexed = sorted(enumerate(self._cache), key=lambda x: x[1].attention_score, reverse=True)
            keep_by_score = set(idx for idx, _ in indexed[:top_half_count])
            # Newest 50%: last half by position
            newest_start = total - top_half_count
            keep_by_recency = set(range(newest_start, total))
            keep_indices = keep_by_score | keep_by_recency
            evict_indices_list = [i for i in range(total) if i not in keep_indices]
            # Only evict up to n
            evict_indices = set(evict_indices_list[:n])
            evicted = [self._cache[i] for i in sorted(evict_indices)]
            self._cache = [e for i, e in enumerate(self._cache) if i not in evict_indices]

        else:
            evicted = self._cache[:n]
            self._cache = self._cache[n:]

        return evicted

    def compress_int8(self, values: list[float]) -> tuple[list[int], float, float]:
        if not values:
            return [], 1e-8, 0.0
        min_val = min(values)
        max_val = max(values)
        scale = (max_val - min_val) / 255.0 if (max_val - min_val) > 0 else 1e-8
        zero = min_val
        quants = [int(round((v - zero) / scale)) for v in values]
        # Clamp to [0, 255]
        quants = [max(0, min(255, q)) for q in quants]
        return quants, scale, zero

    def decompress_int8(self, quants: list[int], scale: float, zero: float) -> list[float]:
        return [q * scale + zero for q in quants]

    def __len__(self) -> int:
        return len(self._cache)

    def entries(self) -> list[KVEntry]:
        return list(self._cache)


KV_CACHE_COMPRESSOR = KVCacheCompressor()
