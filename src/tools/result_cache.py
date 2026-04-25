"""Tool result cache with invalidation by key prefix."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    key: str
    value: Any
    ttl: float
    created: float = 0.0

    def expired(self) -> bool:
        return time.monotonic() - self.created > self.ttl


@dataclass
class ResultCache:
    """Cache tool results with TTL and prefix-based invalidation."""

    default_ttl: float = 60.0
    _entries: dict[str, CacheEntry] = field(default_factory=dict, repr=False)

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        if entry is None or entry.expired():
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._entries[key] = CacheEntry(
            key=key, value=value, ttl=ttl or self.default_ttl, created=time.monotonic()
        )

    def invalidate(self, prefix: str = "") -> int:
        if not prefix:
            count = len(self._entries)
            self._entries.clear()
            return count
        keys = [k for k in self._entries if k.startswith(prefix)]
        for k in keys:
            del self._entries[k]
        return len(keys)

    def cleanup(self) -> int:
        expired = [k for k, v in self._entries.items() if v.expired()]
        for k in expired:
            del self._entries[k]
        return len(expired)

    def size(self) -> int:
        return len(self._entries)


RESULT_CACHE = ResultCache()