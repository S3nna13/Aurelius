"""Memory-backed key-value store with TTL support."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KVItem:
    value: Any
    ttl: float = 0.0
    created: float = 0.0

    def expired(self) -> bool:
        if self.ttl is None or self.ttl < 0:
            return False
        if self.ttl == 0:
            return time.monotonic() - self.created > 0.001
        return time.monotonic() - self.created > self.ttl


@dataclass
class KVStore:
    """Simple in-memory key-value store with optional TTL."""

    _data: dict[str, KVItem] = field(default_factory=dict, repr=False)

    def put(self, key: str, value: Any, ttl: float = 0.0) -> None:
        self._data[key] = KVItem(value=value, ttl=ttl, created=time.monotonic())

    def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if item is None:
            return None
        if item.expired():
            del self._data[key]
            return None
        return item.value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def size(self) -> int:
        now = time.monotonic()
        return sum(1 for i in self._data.values() if i.ttl <= 0 or now - i.created <= i.ttl)

    def clear(self) -> None:
        self._data.clear()

    def cleanup(self) -> int:
        now = time.monotonic()
        expired = [k for k, v in self._data.items() if v.ttl > 0 and now - v.created > v.ttl]
        for k in expired:
            del self._data[k]
        return len(expired)


KV_STORE = KVStore()
