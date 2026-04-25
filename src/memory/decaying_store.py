"""Time-decaying memory store for recency-weighted retrieval."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class DecayingMemoryItem:
    key: str
    value: str
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


@dataclass
class DecayingMemoryStore:
    """Memory store that decays entries based on age."""

    ttl_seconds: float = 3600.0
    _items: dict[str, DecayingMemoryItem] = field(default_factory=dict, repr=False)

    def put(self, key: str, value: str) -> None:
        self._items[key] = DecayingMemoryItem(key=key, value=value)

    def get(self, key: str) -> str | None:
        item = self._items.get(key)
        if item is None:
            return None
        if time.monotonic() - item.timestamp > self.ttl_seconds:
            del self._items[key]
            return None
        return item.value

    def get_weighted(self, key: str) -> tuple[str, float] | None:
        item = self._items.get(key)
        if item is None:
            return None
        age = time.monotonic() - item.timestamp
        if age > self.ttl_seconds:
            del self._items[key]
            return None
        weight = max(0.0, 1.0 - (age / self.ttl_seconds))
        return item.value, weight

    def size(self) -> int:
        now = time.monotonic()
        alive = sum(1 for i in self._items.values() if now - i.timestamp <= self.ttl_seconds)
        return alive

    def clear(self) -> None:
        self._items.clear()

    def prune(self) -> int:
        now = time.monotonic()
        expired = [k for k, v in self._items.items() if now - v.timestamp > self.ttl_seconds]
        for k in expired:
            del self._items[k]
        return len(expired)


DECAYING_MEMORY = DecayingMemoryStore()