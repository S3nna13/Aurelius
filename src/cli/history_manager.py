from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone


class HistoryError(ValueError):
    pass


@dataclass
class HistoryEntry:
    command: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HistoryManager:
    def __init__(self, history_file: str = "~/.aurelius_history.jsonl", max_entries: int = 1000) -> None:
        resolved = os.path.expanduser(history_file)
        abs_path = os.path.abspath(resolved)
        cwd = os.path.abspath(os.getcwd())
        if not abs_path.startswith(cwd) and ".." in history_file.split(os.sep):
            raise HistoryError(f"Path traversal rejected: {history_file}")
        self.history_file = abs_path
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: list[HistoryEntry] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.history_file):
            return
        with open(self.history_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._entries.append(HistoryEntry(
                    command=data["command"], timestamp=data.get("timestamp", "")
                ))

    def _save(self) -> None:
        with open(self.history_file, "w") as f:
            for entry in self._entries:
                f.write(json.dumps({"command": entry.command, "timestamp": entry.timestamp}) + "\n")

    def add(self, command: str) -> None:
        if not isinstance(command, str):
            raise HistoryError("Command must be a string")
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", command):
            raise HistoryError("Command contains control characters")
        with self._lock:
            if self._entries and self._entries[-1].command == command:
                return
            self._entries.append(HistoryEntry(command=command))
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]
            self._save()

    def get_recent(self, n: int) -> list[HistoryEntry]:
        with self._lock:
            return self._entries[-n:]

    def search(self, query: str, limit: int | None = None) -> list[HistoryEntry]:
        query_lower = query.lower()
        results = [e for e in reversed(self._entries) if query_lower in e.command.lower()]
        if limit is not None:
            results = results[:limit]
        return results

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            if os.path.exists(self.history_file):
                os.remove(self.history_file)

    def __len__(self) -> int:
        return len(self._entries)
