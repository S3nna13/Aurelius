"""BackgroundExecutor — 24/7 priority task queue for persistent agents.

Motivated by Kimi K2.6's proactive background-agent design, which supports
autonomous 24/7 task execution without human oversight. This module provides:

- A priority-ordered queue of ``BackgroundTask`` objects
- Synchronous task execution (no threads/async) suitable for test harnesses
- Heartbeat-based fault detection for stale in-flight tasks
- Registration in ``AGENT_LOOP_REGISTRY["background_executor"]``

The design is deliberately stdlib-only: no PyTorch, no asyncio, no
multiprocessing.  The synchronous ``run_next`` pattern lets unit tests drive
execution step-by-step without any scheduler or event loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    """A single unit of work managed by :class:`BackgroundExecutor`."""

    task_id: str
    priority: int          # lower value == higher priority (0 is highest)
    task_fn: Callable      # callable that takes no args, returns Any
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    heartbeat_at: Optional[float] = None
    max_duration_s: float = 300.0  # per-task wall-clock timeout (informational)


@dataclass
class BackgroundExecutorConfig:
    """Configuration knobs for :class:`BackgroundExecutor`."""

    max_queue_size: int = 1000
    heartbeat_interval_s: float = 30.0
    default_priority: int = 5
    stale_threshold_s: float = 60.0  # mark RUNNING task stale if no heartbeat this long


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class BackgroundExecutor:
    """Priority task queue with heartbeat-based fault detection.

    All execution is *synchronous*: call :meth:`run_next` to pop and execute
    the highest-priority PENDING task.  This keeps the class usable in unit
    tests without any threading or event-loop machinery.

    Parameters
    ----------
    config:
        Optional :class:`BackgroundExecutorConfig`.  Defaults are used when
        omitted.
    """

    def __init__(self, config: Optional[BackgroundExecutorConfig] = None) -> None:
        self._config: BackgroundExecutorConfig = config or BackgroundExecutorConfig()
        # task_id → BackgroundTask (all states)
        self._tasks: dict[str, BackgroundTask] = {}

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self,
        task_id: str,
        task_fn: Callable,
        priority: Optional[int] = None,
    ) -> BackgroundTask:
        """Enqueue *task_fn* under *task_id* with the given *priority*.

        Parameters
        ----------
        task_id:
            Unique identifier.  Raises :exc:`ValueError` if already present.
        task_fn:
            Zero-argument callable; its return value becomes
            ``BackgroundTask.result`` on success.
        priority:
            Lower integers run first.  Defaults to
            ``config.default_priority``.

        Raises
        ------
        ValueError
            If *task_id* already exists, or the queue is full.
        """
        if task_id in self._tasks:
            raise ValueError(f"task_id {task_id!r} already exists")
        pending_count = self.pending_count()
        if pending_count >= self._config.max_queue_size:
            raise ValueError(
                f"queue full: max_queue_size={self._config.max_queue_size}"
            )
        if priority is None:
            priority = self._config.default_priority
        task = BackgroundTask(task_id=task_id, priority=priority, task_fn=task_fn)
        self._tasks[task_id] = task
        return task

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_next(self) -> Optional[BackgroundTask]:
        """Pop the highest-priority PENDING task, execute it, return it.

        The task transitions through RUNNING → COMPLETED (or FAILED).
        Returns ``None`` if there are no PENDING tasks.
        """
        pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
        if not pending:
            return None

        # lowest priority integer == highest priority
        task = min(pending, key=lambda t: (t.priority, t.created_at))

        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        task.heartbeat_at = task.started_at

        try:
            task.result = task.task_fn()
            task.status = TaskStatus.COMPLETED
        except Exception as exc:  # noqa: BLE001
            task.status = TaskStatus.FAILED
            task.error = str(exc)
        finally:
            task.completed_at = time.time()

        return task

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self, task_id: str) -> bool:
        """Cancel a PENDING task.

        Returns
        -------
        bool
            ``True`` if the task was found in PENDING state and cancelled;
            ``False`` if not found or already in a non-PENDING state.
        """
        task = self._tasks.get(task_id)
        if task is None or task.status != TaskStatus.PENDING:
            return False
        task.status = TaskStatus.CANCELLED
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """Return the :class:`TaskStatus` for *task_id*, or ``None``."""
        task = self._tasks.get(task_id)
        return task.status if task is not None else None

    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Return the :class:`BackgroundTask` for *task_id*, or ``None``."""
        return self._tasks.get(task_id)

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def heartbeat(self, task_id: str) -> bool:
        """Record a heartbeat for a RUNNING task.

        Returns
        -------
        bool
            ``True`` if the task exists and is RUNNING (heartbeat recorded);
            ``False`` otherwise.
        """
        task = self._tasks.get(task_id)
        if task is None or task.status != TaskStatus.RUNNING:
            return False
        task.heartbeat_at = time.time()
        return True

    def detect_stale(self) -> list[str]:
        """Return task IDs of RUNNING tasks whose last heartbeat is stale.

        A task is *stale* when its ``heartbeat_at`` is older than
        ``config.stale_threshold_s`` seconds ago.
        """
        threshold = self._config.stale_threshold_s
        now = time.time()
        stale: list[str] = []
        for task in self._tasks.values():
            if task.status != TaskStatus.RUNNING:
                continue
            if task.heartbeat_at is None:
                stale.append(task.task_id)
            elif (now - task.heartbeat_at) > threshold:
                stale.append(task.task_id)
        return stale

    # ------------------------------------------------------------------
    # Counts / snapshot
    # ------------------------------------------------------------------

    def pending_count(self) -> int:
        """Number of tasks in PENDING state."""
        return sum(
            1 for t in self._tasks.values() if t.status == TaskStatus.PENDING
        )

    def running_count(self) -> int:
        """Number of tasks in RUNNING state."""
        return sum(
            1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING
        )

    def queue_snapshot(self) -> list[BackgroundTask]:
        """All PENDING tasks sorted by priority (lowest int first), then insertion order."""
        pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
        return sorted(pending, key=lambda t: (t.priority, t.created_at))


__all__ = [
    "BackgroundExecutor",
    "BackgroundExecutorConfig",
    "BackgroundTask",
    "TaskStatus",
]
