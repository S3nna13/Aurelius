from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


class SemanticCache:
    """Semantic similarity cache — returns cached result if query is similar enough.

    Uses cosine similarity on embedding vectors to find cache hits.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]] | None = None,
        threshold: float = 0.92,
        max_entries: int = 500,
    ) -> None:
        self.embed_fn = embed_fn or self._default_embed
        self.threshold = threshold
        self.max_entries = max_entries
        self._cache: list[tuple[list[float], Any]] = []
        self._lock = threading.Lock()

    @staticmethod
    def _default_embed(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:64]]

    def get(self, query: str) -> Any | None:
        qv = self.embed_fn(query)
        with self._lock:
            for emb, value in self._cache:
                sim = self._cosine(qv, emb)
                if sim >= self.threshold:
                    return value
        return None

    def set(self, query: str, value: Any) -> None:
        emb = self.embed_fn(query)
        with self._lock:
            self._cache.append((emb, value))
            if len(self._cache) > self.max_entries:
                self._cache.pop(0)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, capacity: int = 1000, default_ttl: float = 3600.0) -> None:
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._store:
                return None
            value, expiry = self._store.pop(key)
            if time.monotonic() > expiry:
                return None
            self._store[key] = (value, expiry)
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        expiry = time.monotonic() + (ttl if ttl is not None else self.default_ttl)
        with self._lock:
            self._store[key] = (value, expiry)
            self._store.move_to_end(key)
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._store)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, e) in self._store.items() if now > e]
        for k in expired:
            del self._store[k]


class CacheService:
    """Combined cache with L1 semantic cache + L2 LRU cache."""

    def __init__(self, semantic: SemanticCache | None = None, lru: LRUCache | None = None) -> None:
        self.semantic = semantic or SemanticCache()
        self.lru = lru or LRUCache()

    def get(self, query: str, key: str | None = None) -> Any | None:
        result = self.semantic.get(query)
        if result is not None:
            return result
        if key is not None:
            return self.lru.get(key)
        return None

    def set(self, query: str, value: Any, key: str | None = None) -> None:
        self.semantic.set(query, value)
        if key is not None:
            self.lru.set(key, value)

    def stats(self) -> dict[str, int]:
        return {"semantic_entries": self.semantic.size(), "lru_entries": self.lru.size()}
