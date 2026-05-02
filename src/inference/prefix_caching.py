"""Prefix-based KV cache sharing for inference.

Stores/retrieves cached KV states keyed by input prefix hash.
Thread-safe LRU eviction via collections.OrderedDict.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrefixCacheConfig:
    max_size: int = 64
    min_prefix_len: int = 4


@dataclass
class CachedPrefix:
    prefix_ids: tuple[int, ...]
    kv_state: Any  # opaque cached KV state, typed as Any for flexibility
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)


def compute_prefix_hash(token_ids: list[int]) -> str:
    data = b"".join(i.to_bytes(4, "little") for i in token_ids)
    return hashlib.sha256(data).hexdigest()


class PrefixCache:
    """Thread-safe LRU prefix KV cache backed by collections.OrderedDict."""

    def __init__(self, config: PrefixCacheConfig | None = None) -> None:
        self._config = config if config is not None else PrefixCacheConfig()
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, CachedPrefix] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    @property
    def config(self) -> PrefixCacheConfig:
        return self._config

    def get(self, token_ids: list[int]) -> tuple[Any | None, int]:
        hash_key = compute_prefix_hash(token_ids)

        with self._lock:
            entry = self._cache.get(hash_key)
            if entry is not None:
                self._cache.move_to_end(hash_key)
                entry.last_accessed = time.time()
                self._hits += 1
                return entry.kv_state, len(entry.prefix_ids)
            self._misses += 1
            return None, 0

    def put(self, token_ids: list[int], kv_state: Any) -> None:
        if len(token_ids) < self._config.min_prefix_len:
            return

        hash_key = compute_prefix_hash(token_ids)
        with self._lock:
            if hash_key in self._cache:
                self._cache.move_to_end(hash_key)
                entry = self._cache[hash_key]
                entry.kv_state = kv_state
                entry.last_accessed = time.time()
                return

            if len(self._cache) >= self._config.max_size:
                self._cache.popitem(last=False)

            self._cache[hash_key] = CachedPrefix(
                prefix_ids=tuple(token_ids),
                kv_state=kv_state,
            )

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, float]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": float(len(self._cache)),
            "max_size": float(self._config.max_size),
            "hits": float(self._hits),
            "misses": float(self._misses),
            "hit_rate": hit_rate,
        }

    def __len__(self) -> int:
        return len(self._cache)
