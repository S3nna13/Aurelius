"""Episodic memory: event-stamped entries with recency scoring."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class MemoryEntry:
    """A single episodic memory event."""

    role: str
    content: str
    importance: float = 1.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class EpisodicMemory:
    """Episodic memory store with capacity-bounded, recency/importance retrieval."""

    def __init__(self, max_entries: int = 1000) -> None:
        self._max_entries = max_entries
        self._entries: list[MemoryEntry] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def store(self, role: str, content: str, importance: float = 1.0) -> MemoryEntry:
        """Create and store a new MemoryEntry, evicting oldest if over capacity."""
        entry = MemoryEntry(role=role, content=content, importance=importance)
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            # Evict oldest (lowest index / earliest timestamp)
            self._entries = self._entries[-self._max_entries :]
        return entry

    def forget(self, entry_id: str) -> bool:
        """Remove entry by id. Returns True if found and removed."""
        for i, e in enumerate(self._entries):
            if e.id == entry_id:
                self._entries.pop(i)
                return True
        return False

    def clear(self) -> int:
        """Remove all entries. Returns count removed."""
        count = len(self._entries)
        self._entries.clear()
        return count

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return the last *n* entries in timestamp order."""
        return list(self._entries[-n:]) if self._entries else []

    def retrieve_by_importance(self, threshold: float = 0.5) -> list[MemoryEntry]:
        """Return entries with importance >= threshold, sorted descending."""
        filtered = [e for e in self._entries if e.importance >= threshold]
        return sorted(filtered, key=lambda e: e.importance, reverse=True)

    def search(self, query: str) -> list[MemoryEntry]:
        """Case-insensitive substring search over entry content."""
        lower_q = query.lower()
        return [e for e in self._entries if lower_q in e.content.lower()]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)
