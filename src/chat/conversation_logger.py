"""Persistent JSON-lines conversation logger.

Each session is stored as a separate ``.jsonl`` file under *log_dir*.
Turns are appended atomically (open-append-close per line).  When the
number of session files exceeds *max_files* the oldest file by mtime is
removed before a new one is created.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

__all__ = [
    "ConversationLogger",
    "CONVERSATION_LOGGER_REGISTRY",
]


#: Valid characters for a session identifier.
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(
            "session_id may only contain alphanumeric characters, hyphens, "
            f"and underscores; got {session_id!r}"
        )


def _session_path(log_dir: Path, session_id: str) -> Path:
    return log_dir / f"{session_id}.jsonl"


class ConversationLogger:
    """Append-only JSON-lines logger for chat turns."""

    def __init__(self, log_dir: str, max_files: int = 100) -> None:
        self.log_dir = Path(log_dir)
        self.max_files = max_files
        if self.max_files < 1:
            raise ValueError("max_files must be >= 1")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _rotate_if_needed(self, target: Path) -> None:
        """Delete the oldest session file if we are at capacity."""
        if target.exists():
            return
        files = sorted(
            (p for p in self.log_dir.iterdir() if p.suffix == ".jsonl"),
            key=lambda p: p.stat().st_mtime,
        )
        if len(files) >= self.max_files:
            oldest = files[0]
            oldest.unlink()

    def log_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: float | None = None,
    ) -> None:
        """Append a turn to *session_id*'s log file."""
        _validate_session_id(session_id)
        path = _session_path(self.log_dir, session_id)
        self._rotate_if_needed(path)
        record = {
            "role": role,
            "content": content,
            "timestamp": timestamp if timestamp is not None else time.time(),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def get_history(self, session_id: str) -> list[dict]:
        """Return all turns for *session_id* as a list of dicts."""
        _validate_session_id(session_id)
        path = _session_path(self.log_dir, session_id)
        if not path.exists():
            return []
        turns: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    turns.append(json.loads(line))
        return turns

    def list_sessions(self) -> list[str]:
        """Return all session identifiers with existing log files."""
        return sorted(
            p.stem for p in self.log_dir.iterdir() if p.suffix == ".jsonl"
        )

    def delete_session(self, session_id: str) -> None:
        """Remove *session_id*'s log file if it exists."""
        _validate_session_id(session_id)
        path = _session_path(self.log_dir, session_id)
        if path.exists():
            path.unlink()


CONVERSATION_LOGGER_REGISTRY: dict[str, ConversationLogger] = {
    "default": ConversationLogger(
        log_dir=str(Path.home() / ".cache" / "aurelius" / "conversation_logs"),
        max_files=100,
    ),
}
