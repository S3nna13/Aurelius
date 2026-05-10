from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class FusionStrategy(Enum):
    WEIGHTED_AVERAGE = "weighted"
    MOST_RECENT = "recent"
    UNION = "union"


@dataclass
class MemoryFragment:
    content: str
    source: str
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)


class MemorySource:
    def __init__(self, name: str) -> None:
        self.name = name
        self._fragments: list[MemoryFragment] = []

    def add(self, fragment: MemoryFragment) -> None:
        self._fragments.append(fragment)

    def get_all(self) -> list[MemoryFragment]:
        return list(self._fragments)


def fuse_weighted_average(fragments: list[MemoryFragment]) -> MemoryFragment:
    if not fragments:
        return MemoryFragment(content="", source="fusion", confidence=0.0)
    best = max(fragments, key=lambda f: f.confidence)
    return MemoryFragment(content=best.content, source="fusion", confidence=best.confidence)


def fuse_most_recent(fragments: list[MemoryFragment]) -> MemoryFragment:
    if not fragments:
        return MemoryFragment(content="", source="fusion", confidence=0.0)
    best = max(fragments, key=lambda f: f.timestamp)
    return MemoryFragment(content=best.content, source="fusion", confidence=best.confidence)


def fuse_union(fragments: list[MemoryFragment]) -> MemoryFragment:
    seen: set[str] = set()
    lines: list[str] = []
    for f in fragments:
        if f.content not in seen:
            seen.add(f.content)
            lines.append(f.content)
    combined = "\n".join(lines)
    avg_conf = sum(f.confidence for f in fragments) / max(len(fragments), 1)
    return MemoryFragment(content=combined, source="fusion", confidence=avg_conf)


class MemoryFusion:
    def __init__(self, strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE) -> None:
        self.strategy = strategy
        self._sources: dict[str, MemorySource] = {}
        self._fragments: list[MemoryFragment] = []

    def add_fragment(self, fragment: MemoryFragment) -> None:
        self._fragments.append(fragment)
        if fragment.source not in self._sources:
            self._sources[fragment.source] = MemorySource(fragment.source)
        self._sources[fragment.source].add(fragment)

    def fuse(self) -> MemoryFragment | None:
        if not self._fragments:
            return None
        if self.strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return fuse_weighted_average(self._fragments)
        elif self.strategy == FusionStrategy.MOST_RECENT:
            return fuse_most_recent(self._fragments)
        elif self.strategy == FusionStrategy.UNION:
            return fuse_union(self._fragments)
        return self._fragments[-1] if self._fragments else None

    def get_by_source(self, source: str) -> list[MemoryFragment]:
        return [f for f in self._fragments if f.source == source]


MEMORY_FUSION = MemoryFusion()
