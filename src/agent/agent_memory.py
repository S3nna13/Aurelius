"""Agent memory — episodic + semantic recall with temporal decay."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    content: str
    memory_type: str = "episodic"  # episodic, semantic, procedural
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    access_count: int = 0
    tags: list[str] = field(default_factory=list)


class AgentMemory:
    """Dual-store memory with episodic and semantic recall."""

    def __init__(self, decay_rate: float = 0.99, importance_threshold: float = 0.3):
        self.episodic: list[MemoryEntry] = []
        self.semantic: dict[str, MemoryEntry] = {}
        self.procedural: dict[str, str] = {}
        self.decay_rate = decay_rate
        self.importance_threshold = importance_threshold

    def remember(
        self,
        content: str,
        memory_type: str = "episodic",
        tags: list[str] | None = None,
        importance: float = 1.0,
    ) -> None:
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
        )
        if memory_type == "episodic":
            self.episodic.append(entry)
        elif memory_type == "semantic":
            key = content[:50]
            self.semantic[key] = entry

    def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str | None = None,
    ) -> list[MemoryEntry]:
        now = time.time()
        candidates: list[MemoryEntry] = []

        pool = []
        if memory_type in (None, "episodic"):
            pool.extend(self.episodic)
        if memory_type in (None, "semantic"):
            pool.extend(self.semantic.values())

        for entry in pool:
            age = now - entry.timestamp
            recency = self.decay_rate ** (age / 3600)
            relevance = (
                1.0 if any(kw in entry.content.lower() for kw in query.lower().split()) else 0.1
            )
            score = entry.importance * recency * relevance
            entry.access_count += 1
            candidates.append((score, entry))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in candidates[:top_k] if s > self.importance_threshold]

    def learn_procedure(self, name: str, steps: str) -> None:
        self.procedural[name] = steps

    def recall_procedure(self, name: str) -> str | None:
        return self.procedural.get(name)

    def forget(self, max_age_hours: float = 24.0) -> int:
        now = time.time()
        before = len(self.episodic)
        self.episodic = [e for e in self.episodic if now - e.timestamp < max_age_hours * 3600]
        return before - len(self.episodic)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
            "procedural": len(self.procedural),
        }
