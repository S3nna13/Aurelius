"""Task list panel wired to the Aurelius status hierarchy.

Inspired by MoonshotAI/kimi-cli (MIT, terminal session lifecycle),
Aider-AI/aider (MIT, diff/edit formats), clean-room reimplementation
with original Aurelius branding.

Only rich, stdlib, and project-local imports are used.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TaskPanelError(Exception):
    """Raised when the task panel encounters malformed state or input."""


_STATUS_EMOJIS: dict[str, str] = {
    "pending": "○",
    "running": "◎",
    "paused": "⏸",
    "done": "✓",
    "completed": "✓",
    "failed": "✗",
    "cancelled": "⊘",
    "blocked": "⛔",
}

_STATUS_STYLES: dict[str, str] = {
    "pending": "dim",
    "running": "bold cyan",
    "paused": "yellow",
    "done": "bold green",
    "completed": "bold green",
    "failed": "bold red",
    "cancelled": "dim red",
    "blocked": "bold red",
}


@dataclass
class TaskEntry:
    """A single task registered in a :class:`TaskPanel`.

    Attributes:
        task_id: Stable unique identifier for this task.
        title: Human-readable title.
        status: Current status string (e.g. ``"running"``, ``"done"``).
        progress: Optional progress fraction in ``[0.0, 1.0]``.
        priority: Integer priority; lower values mean higher priority.
        tags: Ordered list of tag strings.
        created_at: Unix timestamp when the task was created.
    """

    task_id: str
    title: str
    status: str
    progress: float | None = None
    priority: int = 0
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise TaskPanelError("task_id must be a non-empty str")
        if not isinstance(self.title, str):
            raise TaskPanelError("title must be a str")
        if not isinstance(self.status, str):
            raise TaskPanelError("status must be a str")
        if self.progress is not None and not isinstance(self.progress, (int, float)):
            raise TaskPanelError("progress must be a float or None")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TaskPanelError("priority must be an int")
        if not isinstance(self.tags, list):
            raise TaskPanelError("tags must be a list")
        if not isinstance(self.created_at, (int, float)):
            raise TaskPanelError("created_at must be a number")


def _progress_bar(progress: float, width: int = 10) -> str:
    """Return a simple ASCII progress bar of the given *width*."""
    filled = int(round(progress * width))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class TaskPanel:
    """Manages a collection of :class:`TaskEntry` objects and renders them.

    All mutations go through explicit methods; direct attribute assignment
    from outside the class is not the intended usage pattern.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskEntry] = {}

    @property
    def tasks(self) -> dict[str, TaskEntry]:
        """Read-only view of the task dict."""
        return dict(self._tasks)

    def add_task(self, entry: TaskEntry) -> None:
        """Register *entry* in this panel.

        Args:
            entry: The :class:`TaskEntry` to register.

        Raises:
            TaskPanelError: If *entry* is not a :class:`TaskEntry`.
        """
        if not isinstance(entry, TaskEntry):
            raise TaskPanelError("entry must be a TaskEntry")
        self._tasks[entry.task_id] = entry

    def update_task(self, task_id: str, **kwargs: Any) -> None:
        """Mutate a registered task by field name.

        Accepts any :class:`TaskEntry` field as a keyword argument and
        updates the task in-place by replacing the dataclass (dataclasses
        are not frozen, so direct mutation is used).

        Args:
            task_id: The id of the task to update.
            **kwargs: Field names and their new values.

        Raises:
            KeyError: If *task_id* is not registered.
            TaskPanelError: If a supplied field name does not exist on
                :class:`TaskEntry`.
        """
        if task_id not in self._tasks:
            raise KeyError(task_id)
        entry = self._tasks[task_id]
        valid_fields = {f.name for f in entry.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        for key, value in kwargs.items():
            if key not in valid_fields:
                raise TaskPanelError(f"TaskEntry has no field {key!r}")
            setattr(entry, key, value)

    def remove_task(self, task_id: str) -> None:
        """Remove the task with *task_id*.

        Raises:
            KeyError: If *task_id* is not registered.
        """
        if task_id not in self._tasks:
            raise KeyError(task_id)
        del self._tasks[task_id]

    def filter_by_status(self, status: str) -> list[TaskEntry]:
        """Return all tasks whose ``status`` matches *status* (case-sensitive).

        Args:
            status: Status string to filter by.

        Returns:
            A list of matching :class:`TaskEntry` objects sorted by
            ``priority`` ascending then ``created_at`` ascending.
        """
        matches = [t for t in self._tasks.values() if t.status == status]
        return sorted(matches, key=lambda t: (t.priority, t.created_at))

    def render(
        self,
        console: Console,
        show_completed: bool = True,
    ) -> None:
        """Render the task list as a Rich Table inside a Panel.

        Args:
            console: Rich :class:`Console` to print to.
            show_completed: When ``False``, tasks with status ``"done"`` or
                ``"completed"`` are hidden.
        """
        visible = list(self._tasks.values())
        if not show_completed:
            visible = [t for t in visible if t.status not in ("done", "completed")]
        visible = sorted(visible, key=lambda t: (t.priority, t.created_at))

        table = Table(
            show_header=True,
            header_style="bold",
            border_style="dim",
            expand=False,
        )
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Title")
        table.add_column("Status", no_wrap=True)
        table.add_column("Progress", no_wrap=True)
        table.add_column("Tags")

        if not visible:
            table.add_row("[dim](no tasks)[/dim]", "", "", "", "")
        else:
            for task in visible:
                status_key = task.status.lower()
                glyph = _STATUS_EMOJIS.get(status_key, "?")
                style = _STATUS_STYLES.get(status_key, "")
                status_cell = Text(f"{glyph} {task.status}", style=style)

                if task.progress is not None:
                    bar = _progress_bar(task.progress)
                    pct = int(round(task.progress * 100))
                    progress_cell = Text(f"{bar} {pct}%", style="dim")
                else:
                    progress_cell = Text("—", style="dim")

                tags_text = ", ".join(task.tags) if task.tags else ""
                table.add_row(
                    task.task_id[:12],
                    task.title,
                    status_cell,
                    progress_cell,
                    tags_text,
                )

        console.print(
            Panel(table, title="[bold]Aurelius Tasks[/bold]", border_style="dim")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of all tasks."""
        return {
            task_id: {
                "task_id": entry.task_id,
                "title": entry.title,
                "status": entry.status,
                "progress": entry.progress,
                "priority": entry.priority,
                "tags": list(entry.tags),
                "created_at": entry.created_at,
            }
            for task_id, entry in self._tasks.items()
        }


TASK_PANEL_REGISTRY: dict[str, TaskPanel] = {}

__all__ = [
    "TaskEntry",
    "TaskPanel",
    "TASK_PANEL_REGISTRY",
    "TaskPanelError",
]
