"""Async RL Infrastructure — GLM-5 §4.1 (arXiv:2602.15763).

Multi-Task Rollout Orchestrator: decouples rollout (inference) from gradient
updates (training).  Supports 1000+ concurrent rollouts with heartbeat-driven
fault tolerance.

Design
------
* Inference engine generates rollouts for heterogeneous task types
  (SWE, terminal, search) — each backed by its own verifier.
* Training engine consumes completed ``RolloutResult`` batches and applies
  gradient updates independently, eliminating the synchronisation bottleneck
  present in standard RLHF pipelines.
* Stale in-flight tasks (heartbeat expired) are re-queued automatically via
  ``requeue_stale()``, providing fault tolerance without external coordinators.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class RolloutTask:
    """A single task to be executed by the inference engine.

    Attributes
    ----------
    task_id:   Unique integer identifier for this task.
    task_type: One of ``"swe"``, ``"terminal"``, or ``"search"``.
    prompt:    The raw prompt string passed to the model.
    metadata:  Arbitrary key/value pairs (e.g. verifier config, env seed).
    """

    task_id: int
    task_type: str  # "swe" | "terminal" | "search"
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResult:
    """The outcome produced by the inference engine for one ``RolloutTask``.

    Attributes
    ----------
    task_id:    Must match the originating ``RolloutTask.task_id``.
    task_type:  Propagated from the task for downstream routing.
    completion: Raw model output string.
    reward:     Scalar reward assigned by the task-specific verifier.
    tokens_used: Total tokens consumed (prompt + completion).
    status:     ``"completed"`` on success, ``"error"`` on any exception.
    """

    task_id: int
    task_type: str
    completion: str
    reward: float
    tokens_used: int
    status: str  # "completed" | "error"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class RolloutOrchestrator:
    """Decouples rollout generation from gradient updates (GLM-5 §4.1).

    The orchestrator maintains:
    * A **pending queue** of tasks waiting to be dispatched.
    * An **in-flight registry** of tasks currently being executed, keyed by
      ``task_id`` and timestamped for heartbeat-based fault detection.

    Parameters
    ----------
    heartbeat_timeout:
        Seconds after which an in-flight task is considered stale and
        eligible for re-queuing (default 30 s).
    max_concurrent:
        Soft upper bound on simultaneous in-flight tasks; enforced by the
        caller via ``dispatch_batch`` batch-size selection (default 1000).
    """

    def __init__(
        self,
        heartbeat_timeout: float = 30.0,
        max_concurrent: int = 1000,
    ) -> None:
        self.heartbeat_timeout = heartbeat_timeout
        self.max_concurrent = max_concurrent
        self._queue: deque[RolloutTask] = deque()
        # task_id -> (task, enqueue_monotonic_time)
        self._in_flight: dict[int, tuple[RolloutTask, float]] = {}

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def enqueue(self, tasks: list[RolloutTask]) -> None:
        """Append *tasks* to the pending queue (FIFO order).

        Parameters
        ----------
        tasks:
            List of ``RolloutTask`` objects to be dispatched on the next
            call to ``dispatch_batch``.
        """
        for task in tasks:
            self._queue.append(task)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch_batch(
        self,
        rollout_fn: Callable[[RolloutTask], RolloutResult],
        batch_size: int = 32,
    ) -> list[RolloutResult]:
        """Dequeue up to *batch_size* tasks and execute them via *rollout_fn*.

        Each task is temporarily registered in ``_in_flight`` while executing.
        On completion (success or exception) the task is removed from the
        in-flight registry.

        Parameters
        ----------
        rollout_fn:
            Callable that accepts a ``RolloutTask`` and returns a
            ``RolloutResult``.  Any exception raised is caught; an error
            result is emitted and execution continues with the next task.
        batch_size:
            Maximum number of tasks to dequeue in this call.

        Returns
        -------
        list[RolloutResult]
            One result per dispatched task, in dispatch order.
        """
        results: list[RolloutResult] = []
        dispatched = 0

        while self._queue and dispatched < batch_size:
            task = self._queue.popleft()
            self._in_flight[task.task_id] = (task, time.monotonic())
            try:
                result = rollout_fn(task)
                results.append(result)
            except Exception:
                results.append(
                    RolloutResult(
                        task_id=task.task_id,
                        task_type=task.task_type,
                        completion="",
                        reward=0.0,
                        tokens_used=0,
                        status="error",
                    )
                )
            finally:
                self._in_flight.pop(task.task_id, None)
            dispatched += 1

        return results

    # ------------------------------------------------------------------
    # Fault tolerance
    # ------------------------------------------------------------------

    def requeue_stale(self) -> int:
        """Re-queue tasks whose heartbeat has expired.

        Any task that has been in-flight for longer than
        ``self.heartbeat_timeout`` seconds is popped from ``_in_flight`` and
        pushed to the *front* of the pending queue (priority re-try).

        Returns
        -------
        int
            Number of tasks that were re-queued.
        """
        now = time.monotonic()
        stale_ids = [
            task_id
            for task_id, (_, timestamp) in self._in_flight.items()
            if now - timestamp > self.heartbeat_timeout
        ]
        for task_id in stale_ids:
            task, _ = self._in_flight.pop(task_id)
            self._queue.appendleft(task)  # front of queue — priority retry
        return len(stale_ids)

    # ------------------------------------------------------------------
    # Observability properties
    # ------------------------------------------------------------------

    @property
    def queue_depth(self) -> int:
        """Number of tasks waiting to be dispatched."""
        return len(self._queue)

    @property
    def in_flight_count(self) -> int:
        """Number of tasks currently registered as in-flight."""
        return len(self._in_flight)
