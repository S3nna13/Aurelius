"""Scratchpad: mutable working surface for intermediate reasoning steps."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class ScratchEntry:
    tag: str
    content: str
    pinned: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class Scratchpad:
    def __init__(self, max_entries: int = 100) -> None:
        self.max_entries = max_entries
        self._entries: list[ScratchEntry] = []

    def write(self, tag: str, content: str, pin: bool = False) -> ScratchEntry:
        entry = ScratchEntry(tag=tag, content=content, pinned=pin)
        self._entries.append(entry)
        # Evict oldest non-pinned entries when over capacity
        while len(self._entries) > self.max_entries:
            for i, e in enumerate(self._entries):
                if not e.pinned:
                    self._entries.pop(i)
                    break
            else:
                # All entries pinned; just truncate from front
                self._entries.pop(0)
        return entry

    def read(self, tag: str) -> list[ScratchEntry]:
        tag_lower = tag.lower()
        return [e for e in self._entries if e.tag.lower() == tag_lower]

    def read_all(self) -> list[ScratchEntry]:
        return list(self._entries)

    def erase(self, entry_id: str, force: bool = False) -> bool:
        for i, e in enumerate(self._entries):
            if e.id == entry_id:
                if e.pinned and not force:
                    return False
                self._entries.pop(i)
                return True
        return False

    def clear_unpinned(self) -> int:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.pinned]
        return before - len(self._entries)

    def tags(self) -> list[str]:
        seen: dict[str, None] = {}
        for e in self._entries:
            seen.setdefault(e.tag, None)
        return list(seen.keys())

    def render(self) -> str:
        lines: list[str] = []
        for e in self._entries:
            star = "★ " if e.pinned else ""
            lines.append(f"{star}[{e.tag}] {e.content}")
        return "\n".join(lines)


SCRATCHPAD = Scratchpad()
