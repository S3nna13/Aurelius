"""SwarmScaler — work-stealing pool scaling to 300+ agents (Kimi K2.6 §4).

Kimi K2.6 scaled agent swarms from ~15 agents (K2.5) to 300+ agents with
4,000+ steps via *work-stealing*: idle workers pull from a global task queue
rather than sitting idle.  This module extends the existing AgentSwarm with a
work-stealing pool.

Design principles
-----------------
* A global priority queue holds all pending WorkItems.
* Workers are simulated sequentially (no real threads) to remain pure Python
  with no external dependencies.
* When work-stealing is enabled, idle workers pull ``work_steal_batch`` items
  in one shot from the front of the remaining queue.
* Critical-path speedup follows the simplified model:
      critical_path = ceil(total_tasks / n_workers_used)
      speedup = serial_steps / critical_path
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Import from sibling module (do NOT modify agent_swarm.py).
from .agent_swarm import AgentSwarm, SubAgentResult  # noqa: F401


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class WorkItem:
    """A single unit of work inside the SwarmScaler queue."""

    task_id: int
    payload: Any                        # arbitrary task data passed to worker_fn
    priority: int = 0                   # lower value = higher priority
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[int] = None
    result: Any = field(default=None, repr=False)


@dataclass
class WorkerStats:
    """Accumulated statistics for one simulated worker."""

    worker_id: int
    tasks_completed: int = 0
    tasks_stolen: int = 0    # tasks this worker pulled via work-stealing
    idle_time_s: float = 0.0


@dataclass
class SwarmScalerConfig:
    """Configuration knobs for SwarmScaler."""

    max_workers: int = 300
    max_total_steps: int = 4000
    work_steal_batch: int = 4    # how many tasks to steal per steal event
    enable_work_stealing: bool = True


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class SwarmScaler:
    """Work-stealing task pool that scales to 300+ simulated workers.

    Usage
    -----
    >>> scaler = SwarmScaler()
    >>> ids = scaler.submit_tasks([1, 2, 3])
    >>> completed = scaler.dispatch(lambda x: x * 2)
    """

    def __init__(self, config: Optional[SwarmScalerConfig] = None) -> None:
        self._config = config or SwarmScalerConfig()
        # Monotonically increasing task counter across the lifetime of this instance.
        self._next_task_id: int = 0
        # Global task queue — stored as a plain list; we sort by priority on dispatch.
        self._pending: list[WorkItem] = []
        # All completed WorkItems across all dispatch calls.
        self._completed: list[WorkItem] = []
        # Per-worker stats from the most recent (or aggregated) dispatch call.
        self._stats: list[WorkerStats] = []
        # Track how many workers were used in the last dispatch (for speedup calc).
        self._last_n_workers: int = 1

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_tasks(
        self,
        payloads: list[Any],
        priorities: Optional[list[int]] = None,
    ) -> list[int]:
        """Submit a batch of tasks and return their assigned task IDs.

        Parameters
        ----------
        payloads:
            Arbitrary task data items.
        priorities:
            Optional per-task priority values (lower = higher priority).
            Defaults to 0 for every task if omitted.

        Returns
        -------
        list[int]
            Sequential task IDs starting from the current counter.

        Raises
        ------
        ValueError
            If submitting these tasks would push the total completed + pending
            count above ``max_total_steps``.
        """
        if priorities is None:
            priorities = [0] * len(payloads)

        if len(payloads) != len(priorities):
            raise ValueError("payloads and priorities must have the same length")

        # Guard: check if we'd exceed max_total_steps when including already
        # completed tasks plus current pending plus new tasks.
        total_after = (
            len(self._completed) + len(self._pending) + len(payloads)
        )
        if total_after > self._config.max_total_steps:
            raise ValueError(
                f"Submitting {len(payloads)} task(s) would exceed "
                f"max_total_steps={self._config.max_total_steps} "
                f"(current total: {len(self._completed) + len(self._pending)})"
            )

        ids: list[int] = []
        for payload, priority in zip(payloads, priorities):
            task_id = self._next_task_id
            self._next_task_id += 1
            self._pending.append(
                WorkItem(task_id=task_id, payload=payload, priority=priority)
            )
            ids.append(task_id)

        return ids

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        worker_fn: Callable[[Any], Any],
        n_workers: Optional[int] = None,
    ) -> list[WorkItem]:
        """Simulate work-stealing dispatch over all pending tasks.

        Pending tasks are sorted by priority (ascending = highest priority
        first).  Workers are assigned tasks round-robin; when work-stealing is
        enabled, any worker that finishes its initial assignment steals up to
        ``work_steal_batch`` tasks from the remaining global queue.

        Parameters
        ----------
        worker_fn:
            Callable invoked with each task's payload.  Its return value is
            stored as ``WorkItem.result``.
        n_workers:
            Number of simulated workers.  Defaults to
            ``min(max_workers, len(pending_tasks))``.

        Returns
        -------
        list[WorkItem]
            All completed ``WorkItem`` objects from this dispatch call.

        Raises
        ------
        ValueError
            If ``n_workers > max_workers``.
        """
        cfg = self._config

        # Determine effective worker count.
        n_pending = len(self._pending)
        effective = min(n_workers or cfg.max_workers, n_pending) if n_pending else 0

        if n_workers is not None and n_workers > cfg.max_workers:
            raise ValueError(
                f"n_workers={n_workers} exceeds max_workers={cfg.max_workers}"
            )

        if effective == 0:
            return []

        self._last_n_workers = effective

        # Sort pending queue by priority ascending (lower = higher priority).
        self._pending.sort(key=lambda w: (w.priority, w.task_id))

        # Initialise per-worker stats for this dispatch.
        stats_map: dict[int, WorkerStats] = {
            wid: WorkerStats(worker_id=wid) for wid in range(effective)
        }

        # ------------------------------------------------------------------
        # Initial round-robin assignment from the sorted global queue.
        # Each worker gets a personal sub-list.
        # ------------------------------------------------------------------
        worker_queues: list[list[WorkItem]] = [[] for _ in range(effective)]
        for idx, item in enumerate(self._pending):
            worker_queues[idx % effective].append(item)

        # Clear the global pending queue — items are now in worker queues.
        self._pending.clear()

        # ------------------------------------------------------------------
        # Simulate execution worker by worker (sequential simulation).
        # Work-stealing: after a worker exhausts its queue it pulls from
        # remaining items not yet started by other workers.  We model this by
        # maintaining a global "steal pool" — items that have not yet been
        # processed by any worker.
        # ------------------------------------------------------------------
        # We drain worker_queues in worker order.  After each worker finishes
        # its own queue, it attempts to steal from workers that still have
        # un-started items (those appear as tail items in their queues since
        # we haven't started executing them yet).  For simplicity we collect
        # all remaining items into a steal pool before we start the loop and
        # steal from that pool.

        # Flatten in priority order for the steal pool.
        steal_pool: list[WorkItem] = []
        # We'll build steal_pool from items beyond what each worker takes in
        # the first pass.  The simplest correct model: each worker takes its
        # assigned items first; any stealing happens from leftover items in
        # OTHER workers' queues that haven't been run yet.
        # To simulate this we run each worker sequentially:
        #   1. Execute all items in its own queue.
        #   2. If work-stealing enabled, steal up to work_steal_batch items
        #      from a shared residual pool (items in queues of workers not
        #      yet "executed").

        # Because we simulate sequentially, we define: the steal pool is
        # the concatenation of all items in workers [current+1 ... last]
        # that haven't been processed yet.  After worker i finishes, it can
        # steal from workers i+1, i+2, ...

        # Rebuild a flat residual list (already priority-sorted).
        residual: list[WorkItem] = []
        for wid in range(effective):
            residual.extend(worker_queues[wid])

        # Now execute worker by worker, removing from residual as we go.
        batch_completed: list[WorkItem] = []
        residual_set_idx: int = 0  # pointer into residual for stealing

        # Track which items each worker "owns" to compute tasks_stolen.
        # Items originally assigned to worker wid are worker_queues[wid].
        original_ids: list[set[int]] = [
            {item.task_id for item in worker_queues[wid]} for wid in range(effective)
        ]

        for wid in range(effective):
            ws = stats_map[wid]

            # Items originally assigned to this worker (that are still in residual).
            my_items = [item for item in residual if item.task_id in original_ids[wid]]

            # Execute own items.
            for item in my_items:
                item.worker_id = wid
                item.started_at = time.monotonic()
                item.result = worker_fn(item.payload)
                item.completed_at = time.monotonic()
                ws.tasks_completed += 1
                batch_completed.append(item)
                # Remove from residual.
                residual = [r for r in residual if r.task_id != item.task_id]

            # Work-stealing: grab items from residual that belong to later workers.
            if cfg.enable_work_stealing:
                stolen = 0
                while residual and stolen < cfg.work_steal_batch:
                    item = residual.pop(0)
                    item.worker_id = wid
                    item.started_at = time.monotonic()
                    item.result = worker_fn(item.payload)
                    item.completed_at = time.monotonic()
                    ws.tasks_completed += 1
                    ws.tasks_stolen += 1
                    batch_completed.append(item)
                    stolen += 1

        # Any items remaining in residual that were not processed (shouldn't
        # happen in normal flow, but handle gracefully).
        for item in residual:
            item.worker_id = 0
            item.started_at = time.monotonic()
            item.result = worker_fn(item.payload)
            item.completed_at = time.monotonic()
            stats_map[0].tasks_completed += 1
            batch_completed.append(item)

        # Persist stats (overwrite — per-dispatch view).
        self._stats = list(stats_map.values())
        self._completed.extend(batch_completed)

        return batch_completed

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def worker_stats(self) -> list[WorkerStats]:
        """Return per-worker stats from the most recent dispatch call."""
        return list(self._stats)

    def total_steps_used(self) -> int:
        """Total tasks completed across all dispatch calls."""
        return len(self._completed)

    def pending_count(self) -> int:
        """Number of tasks currently awaiting dispatch."""
        return len(self._pending)

    def completed_count(self) -> int:
        """Total tasks completed so far (across all dispatch calls)."""
        return len(self._completed)

    def reset(self) -> None:
        """Clear the queue, completed log, and all worker stats."""
        self._pending.clear()
        self._completed.clear()
        self._stats.clear()
        self._next_task_id = 0
        self._last_n_workers = 1

    # ------------------------------------------------------------------
    # Speedup model
    # ------------------------------------------------------------------

    def speedup_vs_serial(self, serial_steps: int) -> float:
        """Compute estimated parallel speedup versus a serial baseline.

        Uses the simplified model::

            critical_path = ceil(total_tasks / n_workers_used)
            speedup = serial_steps / max(critical_path, 1)

        Parameters
        ----------
        serial_steps:
            Number of steps that would be needed if tasks ran one-at-a-time.

        Returns
        -------
        float
            Estimated parallel speedup ratio.
        """
        total_tasks = self.total_steps_used()
        n = max(self._last_n_workers, 1)
        critical_path = math.ceil(total_tasks / n) if total_tasks else 1
        return serial_steps / max(critical_path, 1)
