"""Aurelius search – LRU cache for search results with TTL support."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CacheEntry:
    """A single cache entry holding query results and metadata."""
    query: str
    results: List[Any]
    timestamp: float
    hit_count: int = 0


class SearchCache:
    """LRU cache for search results with per-entry TTL.

    Uses :class:`collections.OrderedDict` to track LRU order: the most-recently
    accessed entry is moved to the *end* of the dict; when the cache is full the
    *first* (oldest) entry is evicted.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[List[Any]]:
        """Return cached results for *query*, or ``None`` on miss / expiry."""
        entry = self._store.get(query)
        if entry is None:
            return None

        now = time.monotonic()
        if now - entry.timestamp > self._ttl:
            # Expired – remove silently
            del self._store[query]
            return None

        # Cache hit: move to end (most-recently used) and increment counter
        self._store.move_to_end(query)
        entry.hit_count += 1
        return entry.results

    def put(self, query: str, results: List[Any]) -> None:
        """Store *results* for *query*, evicting the LRU entry when full."""
        if query in self._store:
            # Refresh existing entry
            self._store.move_to_end(query)
            entry = self._store[query]
            entry.results = list(results)
            entry.timestamp = time.monotonic()
            return

        if len(self._store) >= self._max_size:
            # Evict oldest (first) entry
            self._store.popitem(last=False)

        self._store[query] = CacheEntry(
            query=query,
            results=list(results),
            timestamp=time.monotonic(),
        )

    def invalidate(self, query: str) -> bool:
        """Remove *query* from cache.  Returns ``True`` if it existed."""
        if query in self._store:
            del self._store[query]
            return True
        return False

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        total_hits = sum(e.hit_count for e in self._store.values())
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "hit_count": total_hits,
        }


SEARCH_CACHE_REGISTRY: Dict[str, type] = {"default": SearchCache}
