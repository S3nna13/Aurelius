"""
parallel_step.py — Parallel step execution using threading.
Stdlib-only. Exports PARALLEL_STEP_REGISTRY.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class StepTask:
    name: str
    fn: Callable[[], object]
    timeout_s: float = 30.0


@dataclass(frozen=True)
class ParallelResult:
    name: str
    success: bool
    output: object
    error: str
    duration_ms: float


class ParallelStepExecutor:
    """Runs workflow steps in parallel using threads."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def _run_task(
        self,
        task: StepTask,
        results: list,
        index: int,
    ) -> None:
        """Worker target: execute task.fn(), record timing and outcome."""
        start = time.monotonic()
        try:
            output = task.fn()
            duration_ms = (time.monotonic() - start) * 1000.0
            results[index] = ParallelResult(
                name=task.name,
                success=True,
                output=output,
                error="",
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000.0
            results[index] = ParallelResult(
                name=task.name,
                success=False,
                output=None,
                error=str(exc),
                duration_ms=duration_ms,
            )

    def run(self, tasks: List[StepTask]) -> List[ParallelResult]:
        """
        Run all tasks in parallel (up to max_workers at a time).
        Returns results in the same order as the input tasks list.
        Captures exceptions instead of propagating them.
        """
        if not tasks:
            return []

        results: list = [None] * len(tasks)
        threads: List[threading.Thread] = []

        # Semaphore to bound concurrency to max_workers
        sem = threading.Semaphore(self.max_workers)

        def worker(task: StepTask, idx: int) -> None:
            with sem:
                self._run_task(task, results, idx)

        for i, task in enumerate(tasks):
            t = threading.Thread(target=worker, args=(task, i), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results  # type: ignore[return-value]

    def run_with_timeout(self, tasks: List[StepTask]) -> List[ParallelResult]:
        """
        Same as run() but respects each task's timeout_s.
        If a task's thread is still alive after timeout_s seconds,
        it is marked as a timeout error. (The thread itself is left
        as daemon and will be abandoned — stdlib has no safe thread kill.)
        """
        if not tasks:
            return []

        results: list = [None] * len(tasks)
        threads: List[threading.Thread] = []

        # Start all threads immediately (no semaphore — timeout semantics
        # are per-task; blocking in a semaphore would eat into the timeout).
        for i, task in enumerate(tasks):
            t = threading.Thread(
                target=self._run_task, args=(task, results, i), daemon=True
            )
            threads.append(t)
            t.start()

        # Wait for each thread up to its own timeout.
        for i, (task, t) in enumerate(zip(tasks, threads)):
            t.join(timeout=task.timeout_s)
            if t.is_alive():
                # Thread exceeded its timeout — record a timeout error.
                results[i] = ParallelResult(
                    name=task.name,
                    success=False,
                    output=None,
                    error=f"Task '{task.name}' timed out after {task.timeout_s}s",
                    duration_ms=task.timeout_s * 1000.0,
                )

        return results  # type: ignore[return-value]


# Public registry
PARALLEL_STEP_REGISTRY: dict = {"default": ParallelStepExecutor}
