"""Rich-based streaming progress bars and ETA estimation for the Aurelius terminal UI surface.

Inspired by MoonshotAI/kimi-cli debug surfaces (MIT), Anthropic Claude Code progress rendering (MIT),
clean-room reimplementation with original Aurelius design.

Only rich, stdlib, and project-local imports are used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text


class ProgressError(Exception):
    """Raised when ProgressRenderer encounters a missing task or invalid state."""


@dataclass
class ProgressTask:
    """Descriptor for a single trackable work unit.

    Attributes:
        task_id: Unique identifier for this task.
        description: Human-readable label rendered in the progress table.
        total: Total number of units; ``None`` for indeterminate tasks.
        completed: Number of units completed so far.
        visible: Whether this task appears in renders.
        metadata: Arbitrary key/value pairs for caller use.
    """

    task_id: str
    description: str
    total: int | None
    completed: int = 0
    visible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class ETAEstimator:
    """Velocity-based ETA estimator backed by a ring-buffer of samples.

    Attributes:
        _samples: Ring buffer of ``(timestamp, completed)`` pairs; capped at 100.
    """

    _MAX_SAMPLES: int = 100

    def __init__(self) -> None:
        self._samples: list[tuple[float, int]] = []

    def record(self, completed: int) -> None:
        """Append a ``(time.time(), completed)`` sample.

        If the buffer is full the oldest sample is dropped.
        """
        self._samples.append((time.time(), completed))
        if len(self._samples) > self._MAX_SAMPLES:
            self._samples.pop(0)

    def eta_seconds(self, total: int) -> float | None:
        """Estimate seconds remaining from the most-recent velocity.

        Args:
            total: The target completion count.

        Returns:
            Estimated seconds, or ``None`` if fewer than 2 samples exist.
        """
        if len(self._samples) < 2:
            return None
        t0, c0 = self._samples[0]
        t1, c1 = self._samples[-1]
        elapsed = t1 - t0
        delta_c = c1 - c0
        if elapsed <= 0 or delta_c <= 0:
            return None
        velocity = delta_c / elapsed  # items / second
        remaining = total - c1
        if remaining <= 0:
            return 0.0
        return remaining / velocity

    def throughput(self) -> float | None:
        """Return items-per-second from the last two samples.

        Returns:
            Throughput as a float, or ``None`` if fewer than 2 samples exist.
        """
        if len(self._samples) < 2:
            return None
        t0, c0 = self._samples[-2]
        t1, c1 = self._samples[-1]
        elapsed = t1 - t0
        if elapsed <= 0:
            return None
        return (c1 - c0) / elapsed


# ---------------------------------------------------------------------------
# Progress bar rendering helpers
# ---------------------------------------------------------------------------

_BAR_WIDTH = 20
_FILLED = "■"
_EMPTY = "□"


def _progress_bar(completed: int, total: int | None, width: int = _BAR_WIDTH) -> str:
    """Return an ASCII progress bar string."""
    if total is None or total <= 0:
        return _EMPTY * width
    fraction = min(completed / total, 1.0)
    filled = int(fraction * width)
    return _FILLED * filled + _EMPTY * (width - filled)


def _percentage(completed: int, total: int | None) -> str:
    if total is None or total <= 0:
        return " — %"
    pct = min(100.0, completed / total * 100)
    return f"{pct:5.1f}%"


def _eta_str(estimator: ETAEstimator, task: ProgressTask) -> str:
    if task.total is None:
        return "  —  "
    if task.completed >= task.total:
        return " done"
    eta = estimator.eta_seconds(task.total)
    if eta is None:
        return "  —  "
    if eta < 60:
        return f"{eta:4.0f}s"
    minutes = eta / 60
    return f"{minutes:4.1f}m"


class ProgressRenderer:
    """Manages and renders a collection of :class:`ProgressTask` objects.

    All mutations go through explicit methods.  Rendering uses Rich Tables
    with ASCII block characters; no Unicode dependency beyond what Rich itself
    requires.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, ProgressTask] = {}
        self._estimators: dict[str, ETAEstimator] = {}

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_task(self, task: ProgressTask) -> None:
        """Register *task* and create a fresh :class:`ETAEstimator` for it."""
        self._tasks[task.task_id] = task
        self._estimators[task.task_id] = ETAEstimator()

    def advance(self, task_id: str, delta: int = 1) -> None:
        """Increment a task's ``completed`` by *delta* and record a sample.

        Args:
            task_id: The task to advance.
            delta: Number of units to add (default 1).

        Raises:
            ProgressError: If *task_id* is not registered.
        """
        if task_id not in self._tasks:
            raise ProgressError(
                f"task {task_id!r} not found; available: {list(self._tasks)}"
            )
        task = self._tasks[task_id]
        task.completed += delta
        self._estimators[task_id].record(task.completed)

    def complete(self, task_id: str) -> None:
        """Set a task's ``completed`` equal to its ``total``.

        If ``total`` is ``None`` the completed count is left unchanged.

        Raises:
            ProgressError: If *task_id* is not registered.
        """
        if task_id not in self._tasks:
            raise ProgressError(
                f"task {task_id!r} not found; available: {list(self._tasks)}"
            )
        task = self._tasks[task_id]
        if task.total is not None:
            task.completed = task.total
            self._estimators[task_id].record(task.completed)

    def remove_task(self, task_id: str) -> None:
        """Remove a registered task and its estimator.

        Raises:
            ProgressError: If *task_id* is not registered.
        """
        if task_id not in self._tasks:
            raise ProgressError(
                f"task {task_id!r} not found; available: {list(self._tasks)}"
            )
        del self._tasks[task_id]
        del self._estimators[task_id]

    # ------------------------------------------------------------------
    # Render API
    # ------------------------------------------------------------------

    def render(self, console: Console) -> None:
        """Render all visible tasks to *console* as a Rich Table.

        Each row shows: description, progress bar (■/□), percentage, ETA.

        Args:
            console: A :class:`~rich.console.Console` to print to.
        """
        visible = [t for t in self._tasks.values() if t.visible]
        if not visible:
            console.print("[dim](no active tasks)[/dim]")
            return

        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 1),
        )
        table.add_column("Task", style="bold", no_wrap=True)
        table.add_column("Progress", no_wrap=True)
        table.add_column("%", justify="right", no_wrap=True)
        table.add_column("ETA", justify="right", no_wrap=True)

        for task in visible:
            bar = _progress_bar(task.completed, task.total)
            pct = _percentage(task.completed, task.total)
            eta = _eta_str(self._estimators[task.task_id], task)
            table.add_row(task.description, bar, pct, eta)

        console.print(table)

    def render_summary(self, console: Console) -> None:
        """Render a one-line summary of task counts.

        Args:
            console: A :class:`~rich.console.Console` to print to.
        """
        total_tasks = len(self._tasks)
        completed_tasks = sum(
            1
            for t in self._tasks.values()
            if t.total is not None and t.completed >= t.total
        )
        msg = Text()
        msg.append(f"{total_tasks}", style="bold")
        msg.append(" task(s), ")
        msg.append(f"{completed_tasks}", style="bold green")
        msg.append(" completed")
        console.print(msg)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Named pool of :class:`ProgressRenderer` instances.
PROGRESS_RENDERER_REGISTRY: dict[str, ProgressRenderer] = {}

__all__ = [
    "ProgressTask",
    "ETAEstimator",
    "ProgressRenderer",
    "ProgressError",
    "PROGRESS_RENDERER_REGISTRY",
]
