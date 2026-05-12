"""Long-term memory with importance decay and priority eviction.

Inspired by memory consolidation in cognitive science (Atkinson & Shiffrin 1968)
and Generative Agents memory stream (Park et al. 2303.17580); Aurelius-native
implementation. License: MIT.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

_MAX_KEY_LEN = 256
_MAX_VALUE_STR_LEN = 65536
_MAX_CAPACITY = 100_000
_MAX_TAGS = 64


@dataclass
class LTMEntry:
    key: str
    value: Any
    importance: float  # 0.0–1.0; higher = more important
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    access_count: int = 0
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError(f"importance must be in [0, 1], got {self.importance}")
        if len(self.tags) > _MAX_TAGS:
            raise ValueError(f"too many tags (max {_MAX_TAGS})")

    def decayed_importance(self, now: float, decay_rate: float = 0.01) -> float:
        """Exponential time-decay: importance * exp(-decay_rate * elapsed_hours)."""
        elapsed_hours = (now - self.created_at) / 3600.0
        return self.importance * math.exp(-decay_rate * elapsed_hours)

    def score(self, now: float, decay_rate: float = 0.01) -> float:
        """Combined score: decayed_importance + recency bonus + access frequency."""
        decayed = self.decayed_importance(now, decay_rate)
        recency_hours = (now - self.last_accessed) / 3600.0
        recency = math.exp(-0.1 * recency_hours)
        freq = math.log1p(self.access_count) * 0.1
        return decayed + recency * 0.3 + freq


class LongTermMemory:
    def __init__(self, capacity: int = 10_000, decay_rate: float = 0.01) -> None:
        if capacity > _MAX_CAPACITY:
            raise ValueError(f"capacity exceeds max {_MAX_CAPACITY}")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.decay_rate = decay_rate
        self._store: dict[str, LTMEntry] = {}

    def store(
        self, key: str, value: Any, importance: float = 0.5, tags: set[str] | None = None
    ) -> LTMEntry:
        """Store a memory entry. Evicts lowest-score entry if at capacity."""
        if len(key) > _MAX_KEY_LEN:
            raise ValueError(f"key exceeds {_MAX_KEY_LEN} chars")
        if isinstance(value, str) and len(value) > _MAX_VALUE_STR_LEN:
            raise ValueError(f"string value exceeds {_MAX_VALUE_STR_LEN} chars")
        tag_set = frozenset(tags) if tags else frozenset()
        entry = LTMEntry(key=key, value=value, importance=importance, tags=tag_set)
        if key in self._store:
            self._store[key] = entry
            return entry
        if len(self._store) >= self.capacity:
            self._evict_one()
        self._store[key] = entry
        return entry

    def retrieve(self, key: str) -> LTMEntry | None:
        """Retrieve by exact key; updates last_accessed and access_count."""
        entry = self._store.get(key)
        if entry is not None:
            entry.last_accessed = time.monotonic()
            entry.access_count += 1
        return entry

    def search_by_tags(self, tags: set[str], require_all: bool = False) -> list[LTMEntry]:
        """Return entries matching any (or all) of the given tags, sorted by score desc."""
        now = time.monotonic()
        results = []
        for entry in self._store.values():
            if require_all:
                if tags.issubset(entry.tags):
                    results.append(entry)
            else:
                if tags & entry.tags:
                    results.append(entry)
        results.sort(key=lambda e: e.score(now, self.decay_rate), reverse=True)
        return results

    def top_k(self, k: int) -> list[LTMEntry]:
        """Return top-k entries by current score."""
        if k < 1:
            raise ValueError("k must be >= 1")
        now = time.monotonic()
        return sorted(
            self._store.values(), key=lambda e: e.score(now, self.decay_rate), reverse=True
        )[:k]

    def _evict_one(self) -> None:
        """Evict the entry with the lowest current score."""
        now = time.monotonic()
        victim = min(self._store, key=lambda k: self._store[k].score(now, self.decay_rate))
        del self._store[victim]

    def forget(self, key: str) -> bool:
        """Explicitly remove an entry. Returns True if it existed."""
        return self._store.pop(key, None) is not None

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store


LONG_TERM_MEMORY = LongTermMemory()
