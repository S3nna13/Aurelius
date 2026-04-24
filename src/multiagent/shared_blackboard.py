"""Shared blackboard for multi-agent coordination."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BlackboardEntry:
    key: str
    value: object
    author: str
    timestamp: float
    version: int = 1


class Blackboard:
    def __init__(self) -> None:
        self._store: dict[str, BlackboardEntry] = {}
        self._subscribers: dict[str, list[Callable[[BlackboardEntry], None]]] = defaultdict(list)

    def write(self, key: str, value: object, author: str) -> BlackboardEntry:
        existing = self._store.get(key)
        version = (existing.version + 1) if existing is not None else 1
        entry = BlackboardEntry(
            key=key,
            value=value,
            author=author,
            timestamp=time.monotonic(),
            version=version,
        )
        self._store[key] = entry
        self._notify(entry)
        return entry

    def read(self, key: str) -> BlackboardEntry | None:
        return self._store.get(key)

    def read_all(self) -> dict[str, BlackboardEntry]:
        return dict(self._store)

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def subscribe(self, key: str, callback: Callable[[BlackboardEntry], None]) -> None:
        self._subscribers[key].append(callback)

    def _notify(self, entry: BlackboardEntry) -> None:
        for cb in self._subscribers.get(entry.key, []):
            cb(entry)

    def keys(self) -> list[str]:
        return sorted(self._store.keys())

    def version_of(self, key: str) -> int:
        entry = self._store.get(key)
        return entry.version if entry is not None else 0


SHARED_BLACKBOARD_REGISTRY: dict[str, type] = {"default": Blackboard}
