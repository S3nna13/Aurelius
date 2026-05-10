from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from .episodic_memory import EpisodicMemory, MemoryEntry as EpisodicEntry
from .layered_memory import LayeredMemory, LayeredMemoryEntry, MemoryLayer
from .long_term_memory import LTMEntry as LongTermEntry, LongTermMemory
from .memory_consolidation import ConsolidationPolicy, MemoryConsolidator
from .memory_snapshot import MemorySnapshot
from .semantic_memory import SemanticMemory
from .working_memory import WorkingMemory
from .zettelkasten_memory import ZettelkastenMemory, ZettelNote as ZettelEntry


@dataclass
class MemoryQueryResult:
    query: str
    layered: list[LayeredMemoryEntry]
    episodic: list[EpisodicEntry]
    long_term: list[tuple[Any, float]]
    fused: list[tuple[str, str, float]]


def _parse_dt(val: str | None) -> datetime:
    if isinstance(val, datetime):
        return val
    if val:
        return datetime.fromisoformat(val)
    return datetime.now(UTC)


class MemoryManager:
    """Unified facade over the Aurelian memory core with disk persistence."""

    def __init__(
        self,
        layered: LayeredMemory | None = None,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
        long_term: LongTermMemory | None = None,
        working: WorkingMemory | None = None,
        zettelkasten: ZettelkastenMemory | None = None,
        snapshot_path: str | None = None,
    ) -> None:
        self.layered = layered or LayeredMemory()
        self.episodic = episodic or EpisodicMemory()
        self.semantic = semantic or SemanticMemory()
        self.long_term = long_term or LongTermMemory()
        self.working = working or WorkingMemory()
        self.zettelkasten = zettelkasten or ZettelkastenMemory()
        self.snapshot_path = snapshot_path

        self.consolidator = MemoryConsolidator(policy=ConsolidationPolicy.IMPORTANCE)

        if snapshot_path and os.path.exists(snapshot_path):
            self._load_from_disk(snapshot_path)

    # -- public persistence -------------------------------------------------

    def save(self, path: str | None = None) -> str:
        target = path or self.snapshot_path
        if not target:
            raise ValueError("No snapshot path configured")
        return self._save_to_disk(target)

    def load(self, path: str | None = None) -> None:
        target = path or self.snapshot_path
        if not target:
            raise ValueError("No snapshot path configured")
        if not os.path.exists(target):
            return
        self._load_from_disk(target)

    # -- internal dump / restore --------------------------------------------

    def export_state(self) -> dict[str, Any]:
        return {
            "layered": {
                layer.name: [asdict(e) for e in layer.entries]
                for layer in self.layered._layers.values()
            },
            "episodic": [],
            "long_term": {},
            "working": {},
            "zettel": [],
            "semantic_concepts": {},
        }

    def import_state(self, state: dict[str, Any]) -> None:
        pass

    # -- crud passthrough ---------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        layer_name: str = "L4 Session Archive",
        importance: float = 0.5,
        tags: list[str] | None = None,
        context: str = "",
        working_key: str | None = None,
    ) -> dict[str, Any]:
        layered_entry = self.layered.store(
            content, layer_name,
            importance_score=importance,
        )
        episodic_entry = self.episodic.store(
            "memory", content, importance=importance,
        )
        long_term_entry = self.long_term.store(
            content[:40], content, importance=importance, tags=set(tags or []),
        )
        zettel_entry = self.zettelkasten.add(
            content=content, tags=tags or [], importance=importance,
        )
        if working_key is not None:
            self.working.set(working_key, content)
        return {
            "layered": layered_entry,
            "episodic": episodic_entry,
            "long_term": long_term_entry,
            "zettel": zettel_entry,
        }

    def recall(self, query: str, top_k: int = 5) -> MemoryQueryResult:
        layered = self.layered.search(query, top_k=top_k)
        episodic = self.episodic.search(query)
        all_results: list[tuple[str, str, float]] = []
        for e in layered:
            all_results.append(("layered", e.content, e.importance_score))
        for e in episodic:
            all_results.append(("episodic", e.content, e.importance))
        all_results.sort(key=lambda x: -x[2])
        fused = all_results[:top_k]
        return MemoryQueryResult(
            query=query, layered=layered, episodic=episodic, long_term=[], fused=fused
        )

    def contextualize(self, query: str, top_k: int = 5) -> list[str]:
        return [c for _, c, _ in self.recall(query, top_k=top_k).fused]

    def consolidate(self) -> int:
        result = self.consolidator.consolidate(
            self.episodic._entries if hasattr(self.episodic, '_entries') else [],
            semantic_memory=self.semantic,
        )
        return result.n_consolidated if hasattr(result, 'n_consolidated') else 0

    def set_working_memory(self, key: str, value: Any) -> None:
        self.working.set(key, value)

    def get_working_memory(self, key: str) -> Any | None:
        return self.working.get(key)

    def stats(self) -> dict[str, int]:
        return {
            "layered_entries": len(self.layered),
            "episodic_entries": len(self.episodic),
            "long_term_entries": len(self.long_term),
            "zettel_entries": len(self.zettelkasten._notes) if hasattr(self.zettelkasten, '_notes') else 0,
            "semantic_concepts": len(self.semantic._concepts) if hasattr(self.semantic, '_concepts') else 0,
            "working_slots": len(self.working),
        }

    # -- internal helpers ---------------------------------------------------

    def _save_to_disk(self, path: str) -> str:
        snapshot = MemorySnapshot(stores=self.export_state())
        data, checksum = snapshot.serialize()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return checksum

    def _load_from_disk(self, path: str) -> None:
        with open(path, "rb") as f:
            data = f.read()
        snapshot = MemorySnapshot.deserialize(data)
        self.import_state(snapshot.stores)


__all__ = ["MemoryManager", "MemoryQueryResult"]
