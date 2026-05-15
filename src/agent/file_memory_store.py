"""File-backed memory store — MEMORY.md / USER.md pattern.

Inspired by Hermes Agent: persistent agent notes and user profiles stored
as plain-text files, frozen (snapshotted) at session start, and editable
via add / replace / remove operations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MEMORY_CHAR_LIMIT = 2200
_USER_CHAR_LIMIT = 1375


@dataclass
class FileMemoryEntry:
    content: str
    timestamp: float = field(default_factory=time.time)


class FileMemoryStore:
    """Persistent file-backed memory for agent notes and user profiles.

    Two slots:
        *agent* — agent's own notes about environment, conventions, lessons learned.
        *user* — user profile (preferences, communication style, expectations).
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        base = Path(base_dir) if base_dir else Path.home() / ".aurelius" / "memories"
        self._base = base
        self._base.mkdir(parents=True, exist_ok=True)
        self._agent_path = self._base / "MEMORY.md"
        self._user_path = self._base / "USER.md"

    # -- agent memory -------------------------------------------------------

    def read_agent_memory(self) -> str:
        if self._agent_path.exists():
            return self._agent_path.read_text(encoding="utf-8")
        return ""

    def write_agent_memory(self, content: str) -> None:
        if len(content) > _MEMORY_CHAR_LIMIT:
            content = content[:_MEMORY_CHAR_LIMIT]
        existing = self.read_agent_memory()
        if content == existing:
            return
        self._agent_path.write_text(content, encoding="utf-8")

    def add_agent_note(self, note: str) -> str:
        existing = self.read_agent_memory()
        if note in existing:
            return "duplicate"
        combined = (existing + "\n" + note).strip()
        if len(combined) > _MEMORY_CHAR_LIMIT:
            combined = combined[-_MEMORY_CHAR_LIMIT:]
        self._agent_path.write_text(combined, encoding="utf-8")
        return "added"

    def replace_agent_note(self, old: str, new: str) -> str:
        existing = self.read_agent_memory()
        if old not in existing:
            return "not_found"
        replaced = existing.replace(old, new, 1)
        self._agent_path.write_text(replaced, encoding="utf-8")
        return "replaced"

    def remove_agent_note(self, note: str) -> str:
        existing = self.read_agent_memory()
        if note not in existing:
            return "not_found"
        cleaned = existing.replace(note, "").strip()
        self._agent_path.write_text(cleaned, encoding="utf-8")
        return "removed"

    # -- user profile -------------------------------------------------------

    def read_user_profile(self) -> str:
        if self._user_path.exists():
            return self._user_path.read_text(encoding="utf-8")
        return ""

    def write_user_profile(self, content: str) -> None:
        if len(content) > _USER_CHAR_LIMIT:
            content = content[:_USER_CHAR_LIMIT]
        existing = self.read_user_profile()
        if content == existing:
            return
        self._user_path.write_text(content, encoding="utf-8")

    def update_user_profile(self, note: str) -> str:
        existing = self.read_user_profile()
        if note in existing:
            return "duplicate"
        combined = (existing + "\n" + note).strip()
        if len(combined) > _USER_CHAR_LIMIT:
            combined = combined[-_USER_CHAR_LIMIT:]
        self._user_path.write_text(combined, encoding="utf-8")
        return "updated"

    # -- snapshot (frozen at session start) ---------------------------------

    @property
    def session_snapshot(self) -> dict[str, str]:
        return {
            "agent_memory": self.read_agent_memory(),
            "user_profile": self.read_user_profile(),
        }

    # -- stats --------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        a = self.read_agent_memory()
        u = self.read_user_profile()
        return {
            "agent_memory_chars": len(a),
            "agent_memory_lines": len(a.splitlines()) if a else 0,
            "user_profile_chars": len(u),
            "user_profile_lines": len(u.splitlines()) if u else 0,
            "base_dir": str(self._base),
        }


__all__ = [
    "FileMemoryEntry",
    "FileMemoryStore",
]
