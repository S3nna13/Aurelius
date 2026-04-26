"""History store for CLI session persistence."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class HistoryEntry:
    command: str
    timestamp: float = 0.0
    duration_ms: float = 0.0
    exit_code: int = 0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class HistoryStore:
    max_entries: int = 1000
    _entries: list[HistoryEntry] = field(default_factory=list, repr=False)

    def append(self, entry: HistoryEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

    def recent(self, n: int = 10) -> list[HistoryEntry]:
        return self._entries[-n:]

    def search(self, query: str) -> list[HistoryEntry]:
        return [e for e in self._entries if query.lower() in e.command.lower()]

    def all(self) -> list[HistoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def count(self) -> int:
        return len(self._entries)


HISTORY_STORE = HistoryStore()
