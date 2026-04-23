"""Streaming conversation transcript panel for the Aurelius terminal UI surface.

Inspired by MoonshotAI/kimi-cli (MIT, terminal session lifecycle),
Aider-AI/aider (MIT, diff/edit formats), clean-room reimplementation
with original Aurelius branding.

Only rich, stdlib, and project-local imports are used.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class TranscriptViewerError(Exception):
    """Raised when the transcript viewer encounters malformed state or input."""


class TranscriptRole(enum.Enum):
    """Role of a :class:`TranscriptEntry` author."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


_ROLE_STYLES: dict[TranscriptRole, str] = {
    TranscriptRole.USER: "bold cyan",
    TranscriptRole.ASSISTANT: "bold green",
    TranscriptRole.SYSTEM: "bold yellow",
    TranscriptRole.TOOL_CALL: "bold magenta",
    TranscriptRole.TOOL_RESULT: "dim magenta",
}

_ROLE_LABELS: dict[TranscriptRole, str] = {
    TranscriptRole.USER: "USER",
    TranscriptRole.ASSISTANT: "ASSISTANT",
    TranscriptRole.SYSTEM: "SYSTEM",
    TranscriptRole.TOOL_CALL: "TOOL_CALL",
    TranscriptRole.TOOL_RESULT: "TOOL_RESULT",
}


@dataclass
class TranscriptEntry:
    """A single entry in a conversation transcript.

    Attributes:
        role: The role of the author.
        content: The text content of the entry.
        timestamp: Unix timestamp (seconds since epoch).
        metadata: Arbitrary key/value pairs for tooling.
        entry_id: Stable unique identifier for this entry.
    """

    role: TranscriptRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    entry_id: str = field(default_factory=lambda: f"entry-{uuid.uuid4()}")

    def __post_init__(self) -> None:
        if not isinstance(self.role, TranscriptRole):
            raise TranscriptViewerError("role must be a TranscriptRole")
        if not isinstance(self.content, str):
            raise TranscriptViewerError("content must be a str")
        if not isinstance(self.timestamp, (int, float)):
            raise TranscriptViewerError("timestamp must be a number")
        if not isinstance(self.metadata, dict):
            raise TranscriptViewerError("metadata must be a dict")
        if not isinstance(self.entry_id, str) or not self.entry_id.strip():
            raise TranscriptViewerError("entry_id must be a non-empty str")


class TranscriptViewer:
    """Renders a streaming conversation transcript via Rich.

    Each :class:`TranscriptViewer` instance is isolated; entries are appended
    via :meth:`add_entry` and rendered on demand via :meth:`render`.
    Rendering is always separate from mutation — no side effects fire on
    :meth:`add_entry`.
    """

    def __init__(self) -> None:
        self._entries: list[TranscriptEntry] = []

    @property
    def entries(self) -> list[TranscriptEntry]:
        """Read-only view of transcript entries."""
        return list(self._entries)

    def add_entry(self, entry: TranscriptEntry) -> None:
        """Append *entry* to the transcript; fires no side effects.

        Args:
            entry: The :class:`TranscriptEntry` to append.

        Raises:
            TranscriptViewerError: If *entry* is not a :class:`TranscriptEntry`.
        """
        if not isinstance(entry, TranscriptEntry):
            raise TranscriptViewerError("entry must be a TranscriptEntry")
        self._entries.append(entry)

    def render(
        self,
        console: Console,
        max_entries: int = 50,
        show_timestamps: bool = False,
    ) -> None:
        """Render the transcript to *console* inside a Rich Panel.

        Args:
            console: Rich :class:`Console` instance to print to.
            max_entries: Maximum number of most-recent entries to display.
            show_timestamps: If ``True``, prefix each entry with an ISO-8601
                timestamp.
        """
        visible = self._entries[-max_entries:] if self._entries else []
        body = Text()
        if not visible:
            body.append("(no entries)", style="dim")
        else:
            for idx, entry in enumerate(visible):
                if idx > 0:
                    body.append("\n")
                role_style = _ROLE_STYLES.get(entry.role, "")
                label = _ROLE_LABELS.get(entry.role, entry.role.value.upper())
                if show_timestamps:
                    ts = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc)
                    body.append(f"[{ts.strftime('%H:%M:%S')}] ", style="dim")
                body.append(f"{label}: ", style=role_style)
                body.append(entry.content)
        console.print(
            Panel(body, title="[bold]Aurelius Transcript[/bold]", border_style="dim")
        )

    def export_text(self) -> str:
        """Return a plain-text dump of all entries in ``role: content`` format."""
        lines: list[str] = []
        for entry in self._entries:
            label = _ROLE_LABELS.get(entry.role, entry.role.value.upper())
            lines.append(f"{label}: {entry.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset all entries."""
        self._entries = []

    def search(self, query: str) -> list[TranscriptEntry]:
        """Case-insensitive substring search over entry content.

        Args:
            query: Search string.

        Returns:
            A list of matching :class:`TranscriptEntry` objects (preserving
            insertion order).  Returns an empty list when no matches are found.
        """
        if not isinstance(query, str):
            raise TranscriptViewerError("query must be a str")
        lowered = query.lower()
        return [e for e in self._entries if lowered in e.content.lower()]


TRANSCRIPT_VIEWER_REGISTRY: dict[str, TranscriptViewer] = {}

__all__ = [
    "TranscriptRole",
    "TranscriptEntry",
    "TranscriptViewer",
    "TRANSCRIPT_VIEWER_REGISTRY",
    "TranscriptViewerError",
]
