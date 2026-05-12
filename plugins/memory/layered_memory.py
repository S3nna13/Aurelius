"""Layered memory: 5-tier hierarchy inspired by GenericAgent."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


class LayeredMemoryError(Exception):
    """Raised for errors in layered memory operations."""


@dataclass
class LayeredMemoryEntry:
    """A single entry in the layered memory store."""

    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    content: str = ""
    layer: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    access_count: int = 0
    importance_score: float = 0.5


@dataclass
class MemoryLayer:
    """A layer in the memory hierarchy."""

    level: int = 0
    name: str = ""
    ttl_seconds: int | None = None
    max_entries: int | None = None
    entries: list[LayeredMemoryEntry] = field(default_factory=list)


_DEFAULT_LAYER_SPECS: list[dict[str, object]] = [
    {"level": 0, "name": "L0 Meta Rules", "ttl_seconds": None, "max_entries": 100},
    {
        "level": 1,
        "name": "L1 Insight Index",
        "ttl_seconds": 7 * 86400,
        "max_entries": 100,
    },
    {
        "level": 2,
        "name": "L2 Global Facts",
        "ttl_seconds": 30 * 86400,
        "max_entries": 500,
    },
    {
        "level": 3,
        "name": "L3 Task Skills",
        "ttl_seconds": 90 * 86400,
        "max_entries": 50,
    },
    {
        "level": 4,
        "name": "L4 Session Archive",
        "ttl_seconds": 1 * 86400,
        "max_entries": 1000,
    },
]


class LayeredMemory:
    """5-layer memory hierarchy with TTL, capacity, and promotion."""

    def __init__(self, layers: list[MemoryLayer] | None = None) -> None:
        if layers is None:
            self._layers: dict[str, MemoryLayer] = {
                spec["name"]: MemoryLayer(**spec)  # type: ignore[arg-type]
                for spec in _DEFAULT_LAYER_SPECS
            }
        else:
            self._layers = {layer.name: layer for layer in layers}

    def _get_layer(self, layer_name: str) -> MemoryLayer:
        if layer_name not in self._layers:
            raise LayeredMemoryError(f"Unknown layer: {layer_name}")
        return self._layers[layer_name]

    def store(
        self,
        entry: LayeredMemoryEntry | str,
        layer_name: str,
        **kwargs: object,
    ) -> LayeredMemoryEntry:
        """Store an entry (or raw content) into the named layer."""
        if isinstance(entry, str):
            entry = LayeredMemoryEntry(content=entry, layer=layer_name, **kwargs)
        else:
            entry.layer = layer_name
        layer = self._get_layer(layer_name)
        layer.entries.append(entry)
        self._enforce_capacity(layer)
        return entry

    def _enforce_capacity(self, layer: MemoryLayer) -> None:
        """Evict expired and/or lowest-score entries if over capacity."""
        if layer.max_entries is None:
            return
        if len(layer.entries) <= layer.max_entries:
            return
        # Over capacity: first evict expired entries
        self._evict_layer_expired(layer)
        # Then evict by lowest (importance * recency_boost) if still over capacity
        if len(layer.entries) > layer.max_entries:
            now = datetime.now(UTC)

            def _score(e: LayeredMemoryEntry) -> float:
                age_seconds = max(0.0, (now - e.timestamp).total_seconds())
                recency_boost = 1.0 / (1.0 + age_seconds / 3600.0)
                return e.importance_score * recency_boost

            layer.entries.sort(key=_score)
            overflow = len(layer.entries) - layer.max_entries
            layer.entries = layer.entries[overflow:]

    def _evict_layer_expired(self, layer: MemoryLayer) -> int:
        """Remove TTL-expired entries from a single layer. Returns count removed."""
        if layer.ttl_seconds is None:
            return 0
        now = datetime.now(UTC)
        original = len(layer.entries)
        layer.entries = [
            e for e in layer.entries if (now - e.timestamp).total_seconds() <= layer.ttl_seconds
        ]
        return original - len(layer.entries)

    def evict_expired(self) -> int:
        """Remove all TTL-expired entries across every layer. Returns total removed."""
        total = 0
        for layer in self._layers.values():
            total += self._evict_layer_expired(layer)
        return total

    def retrieve(self, query: str, layer_name: str | None = None) -> list[LayeredMemoryEntry]:
        """Case-insensitive substring retrieval. Searches all layers if *layer_name* is None."""
        results: list[LayeredMemoryEntry] = []
        layers = [self._get_layer(layer_name)] if layer_name else list(self._layers.values())
        lower_q = query.lower()
        for layer in layers:
            for entry in layer.entries:
                if lower_q in entry.content.lower():
                    entry.access_count += 1
                    results.append(entry)
        return results

    def promote(self, entry_id: str) -> bool:
        """Promote *entry_id* one layer up if it meets access/importance criteria."""
        for layer in self._layers.values():
            for idx, entry in enumerate(layer.entries):
                if entry.entry_id == entry_id:
                    target_level = layer.level - 1
                    if target_level < 0:
                        return False
                    target_layer = None
                    for tl in self._layers.values():
                        if tl.level == target_level:
                            target_layer = tl
                            break
                    if target_layer is None:
                        return False
                    if entry.access_count < 3 and entry.importance_score < 0.7:
                        return False
                    layer.entries.pop(idx)
                    entry.layer = target_layer.name
                    target_layer.entries.append(entry)
                    self._enforce_capacity(target_layer)
                    return True
        raise LayeredMemoryError(f"Entry not found: {entry_id}")

    def dump_layer(self, layer_name: str) -> list[LayeredMemoryEntry]:
        """Return a shallow copy of all entries in the named layer."""
        layer = self._get_layer(layer_name)
        return list(layer.entries)

    def search(self, query: str, top_k: int = 5) -> list[LayeredMemoryEntry]:
        """Ranked search across all layers using keyword, importance, recency, and access."""
        lower_q = query.lower()
        now = datetime.now(UTC)
        scored: list[tuple[float, LayeredMemoryEntry]] = []
        for layer in self._layers.values():
            for entry in layer.entries:
                keyword_hits = entry.content.lower().count(lower_q)
                if keyword_hits == 0:
                    continue
                age_seconds = max(0.0, (now - entry.timestamp).total_seconds())
                recency_score = 1.0 / (1.0 + age_seconds / 3600.0)
                score = (
                    keyword_hits * 2.0
                    + entry.importance_score * 1.5
                    + entry.access_count * 0.3
                    + recency_score * 1.0
                )
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def __len__(self) -> int:
        return sum(len(layer.entries) for layer in self._layers.values())


DEFAULT_LAYERED_MEMORY = LayeredMemory()
LAYERED_MEMORY_REGISTRY: dict[str, LayeredMemory] = {"default": DEFAULT_LAYERED_MEMORY}
