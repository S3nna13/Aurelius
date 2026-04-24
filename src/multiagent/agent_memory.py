"""Per-agent memory store with episodic, semantic, procedural, and working layers."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


@dataclass(frozen=True)
class MemoryEntry:
    content: str
    memory_type: MemoryType
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    importance: float = 0.5
    timestamp_s: float = field(default_factory=time.monotonic)
    tags: list[str] = field(default_factory=list)
    ttl_s: float | None = None


class AgentMemory:
    def __init__(self, agent_id: str, max_entries: int = 1000) -> None:
        self.agent_id = agent_id
        self.max_entries = max_entries
        self._entries: list[MemoryEntry] = []

    def store(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        tags: list[str] | None = None,
        ttl_s: float | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=list(tags) if tags else [],
            ttl_s=ttl_s,
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            # Evict lowest-importance entry.
            idx = min(range(len(self._entries)), key=lambda i: self._entries[i].importance)
            self._entries.pop(idx)
        return entry

    def recall(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        words = [w.lower() for w in query.split() if w]
        matches: list[MemoryEntry] = []
        for entry in self._entries:
            if memory_type is not None and entry.memory_type != memory_type:
                continue
            content_lower = entry.content.lower()
            tags_lower = [t.lower() for t in entry.tags]
            if any(w in content_lower or w in tags_lower for w in words):
                matches.append(entry)
        matches.sort(key=lambda e: e.importance, reverse=True)
        return matches[:top_k]

    def forget(self, entry_id: str) -> bool:
        for i, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                self._entries.pop(i)
                return True
        return False

    def expire_old(self, current_time: float | None = None) -> int:
        now = current_time if current_time is not None else time.monotonic()
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if e.ttl_s is None or e.timestamp_s + e.ttl_s >= now
        ]
        return before - len(self._entries)

    def consolidate(self, min_importance: float = 0.3) -> int:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.importance >= min_importance]
        return before - len(self._entries)

    def all_entries(self, memory_type: MemoryType | None = None) -> list[MemoryEntry]:
        if memory_type is None:
            return list(self._entries)
        return [e for e in self._entries if e.memory_type == memory_type]

    def stats(self) -> dict:
        counts: dict[str, int] = {mt.value: 0 for mt in MemoryType}
        total_importance = 0.0
        for e in self._entries:
            counts[e.memory_type.value] += 1
            total_importance += e.importance
        total = len(self._entries)
        avg = total_importance / total if total else 0.0
        return {"count_by_type": counts, "avg_importance": avg, "total": total}


AGENT_MEMORY_REGISTRY: dict[str, type[AgentMemory]] = {"default": AgentMemory}
