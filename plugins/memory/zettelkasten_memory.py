"""Zettelkasten graph memory with auto-linking and importance-based eviction — A-MEM pattern (arXiv 2502.12110)."""  # noqa: E501

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ZettelNote:
    note_id: str
    content: str
    tags: list[str]
    importance: float
    created_at: float
    links: list[str] = field(default_factory=list)


class ZettelkastenMemory:
    """Auto-linking graph memory with importance-based eviction (A-MEM pattern)."""

    def __init__(self, max_notes: int = 1000, similarity_threshold: float = 0.7) -> None:
        self.max_notes = max_notes
        self.similarity_threshold = similarity_threshold
        self._notes: dict[str, ZettelNote] = {}

    def _tag_overlap_ratio(self, tags_a: list[str], tags_b: list[str]) -> float:
        if not tags_a or not tags_b:
            return 0.0
        set_a, set_b = set(tags_a), set(tags_b)
        return len(set_a & set_b) / len(set_a | set_b)

    def add(
        self,
        content: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
    ) -> ZettelNote:
        if len(self._notes) >= self.max_notes:
            self.evict_lowest()

        note = ZettelNote(
            note_id=str(uuid.uuid4()),
            content=content,
            tags=tags or [],
            importance=max(0.0, min(1.0, importance)),
            created_at=time.time(),
        )

        for existing in self._notes.values():
            if self._tag_overlap_ratio(note.tags, existing.tags) >= self.similarity_threshold:
                note.links.append(existing.note_id)
                existing.links.append(note.note_id)

        self._notes[note.note_id] = note
        return note

    def get(self, note_id: str) -> ZettelNote | None:
        return self._notes.get(note_id)

    def search(self, query: str, top_k: int = 5) -> list[ZettelNote]:
        query_lower = query.lower()
        query_tags = query_lower.split()

        def score(note: ZettelNote) -> float:
            tag_hits = sum(1 for t in note.tags if t.lower() in query_tags)
            substring_hit = 1.0 if query_lower in note.content.lower() else 0.0
            return tag_hits + substring_hit + note.importance

        ranked = sorted(self._notes.values(), key=score, reverse=True)
        return ranked[:top_k]

    def link(self, from_id: str, to_id: str) -> None:
        if from_id not in self._notes or to_id not in self._notes:
            return
        from_note = self._notes[from_id]
        to_note = self._notes[to_id]
        if to_id not in from_note.links:
            from_note.links.append(to_id)
        if from_id not in to_note.links:
            to_note.links.append(from_id)

    def get_linked(self, note_id: str) -> list[ZettelNote]:
        note = self._notes.get(note_id)
        if note is None:
            return []
        return [self._notes[lid] for lid in note.links if lid in self._notes]

    def evict_lowest(self) -> ZettelNote | None:
        if not self._notes:
            return None
        lowest_id = min(self._notes, key=lambda nid: self._notes[nid].importance)
        return self._notes.pop(lowest_id)

    def list_all(self) -> list[ZettelNote]:
        return list(self._notes.values())

    def size(self) -> int:
        return len(self._notes)
