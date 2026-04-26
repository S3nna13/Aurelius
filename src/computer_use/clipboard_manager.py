"""
clipboard_manager.py
Manages clipboard history for computer use tasks.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class ClipboardEntry:
    content: str
    content_type: str = "text"
    timestamp: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


class ClipboardManager:
    def __init__(self, max_history: int = 50) -> None:
        self._max_history = max_history
        self._history: deque[ClipboardEntry] = deque(maxlen=max_history)
        self._total_copies: int = 0

    def copy(
        self,
        content: str,
        content_type: str = "text",
        source: str = "",
    ) -> ClipboardEntry:
        entry = ClipboardEntry(content=content, content_type=content_type, source=source)
        self._history.appendleft(entry)
        self._total_copies += 1
        return entry

    def paste(self) -> ClipboardEntry | None:
        if not self._history:
            return None
        return self._history[0]

    def paste_nth(self, n: int) -> ClipboardEntry | None:
        if n < 0 or n >= len(self._history):
            return None
        return self._history[n]

    def search(self, query: str) -> list[ClipboardEntry]:
        needle = query.lower()
        results = [e for e in self._history if needle in e.content.lower()]
        # Already newest-first because we appendleft
        return results

    def clear_history(self) -> int:
        count = len(self._history)
        self._history.clear()
        return count

    def __len__(self) -> int:
        return len(self._history)

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for entry in self._history:
            by_type[entry.content_type] = by_type.get(entry.content_type, 0) + 1

        unique_content = len({e.content for e in self._history})

        return {
            "total_copies": self._total_copies,
            "unique_content": unique_content,
            "by_type": by_type,
        }


CLIPBOARD_MANAGER_REGISTRY: dict[str, type] = {"default": ClipboardManager}

REGISTRY = CLIPBOARD_MANAGER_REGISTRY
