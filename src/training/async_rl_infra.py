"""Asynchronous RL infrastructure for Aurelius.

Compatibility layer that preserves the legacy queue/dispatch orchestrator
surface while also exposing the newer submit/poll helpers used by the
training stack.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "RolloutTask",
    "RolloutResult",
    "PARLReward",
    "RolloutOrchestrator",
    "DirectDoubleSidedIS",
    "compute_parl_reward",
]


@dataclass
class RolloutTask:
    task_id: int | str
    task_type: str = "swe"
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    status: str = "pending"
    result: str | None = None
    reward: float | None = None
    tokens_used: int | None = None
    device: str = "cpu"
    created_at: float = field(default_factory=time.time)
    heartbeat: float = field(default_factory=time.monotonic)


@dataclass
class RolloutResult:
    task_id: int | str
    task_type: str
    completion: str
    reward: float
    tokens_used: int
    status: str


@dataclass
class PARLReward:
    performance: float = 0.0
    parallelism: float = 0.0
    finish_rate: float = 0.0

    @property
    def total(self) -> float:
        return self.performance + self.parallelism + self.finish_rate


class RolloutOrchestrator:
    """Async rollout orchestrator with both queue and task-record APIs."""

    def __init__(
        self,
        heartbeat_timeout: float = 30.0,
        max_concurrent: int = 1000,
        reward_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.heartbeat_timeout = heartbeat_timeout
        self.max_concurrent = max_concurrent
        self.reward_fn = reward_fn

        self._queue: deque[RolloutTask] = deque()
        self._in_flight: dict[int | str, tuple[RolloutTask, float]] = {}
        self._tasks: dict[str, RolloutTask] = {}
        self._lock = Lock()

    def enqueue(self, tasks: list[RolloutTask]) -> None:
        """Append tasks to the pending queue in FIFO order."""
        with self._lock:
            for task in tasks:
                self._queue.append(task)

    def dispatch_batch(
        self,
        rollout_fn: Callable[[RolloutTask], RolloutResult],
        batch_size: int = 32,
    ) -> list[RolloutResult]:
        """Dispatch up to ``batch_size`` queued tasks and collect results."""
        results: list[RolloutResult] = []
        limit = min(max(batch_size, 0), self.max_concurrent)

        while len(results) < limit:
            with self._lock:
                if not self._queue:
                    break
                task = self._queue.popleft()
                task.status = "running"
                self._in_flight[task.task_id] = (task, time.monotonic())

            try:
                result = rollout_fn(task)
                if not isinstance(result, RolloutResult):
                    result = RolloutResult(
                        task_id=task.task_id,
                        task_type=task.task_type,
                        completion=str(result),
                        reward=0.0,
                        tokens_used=0,
                        status="completed",
                    )
                task.status = result.status
                task.result = result.completion
                task.reward = result.reward
                task.tokens_used = result.tokens_used
                results.append(result)
            except Exception:
                task.status = "error"
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
                with self._lock:
                    self._in_flight.pop(task.task_id, None)

        return results

    def requeue_stale(self) -> int:
        """Requeue stale in-flight tasks back onto the front of the queue."""
        now = time.monotonic()
        stale_ids: list[int | str] = []

        with self._lock:
            for task_id, (task, started_at) in list(self._in_flight.items()):
                if now - started_at > self.heartbeat_timeout:
                    stale_ids.append(task_id)
                    self._in_flight.pop(task_id, None)
                    task.status = "pending"
                    task.heartbeat = now
                    self._queue.appendleft(task)

        return len(stale_ids)

    @property
    def queue_depth(self) -> int:
        """Number of queued tasks waiting to be dispatched."""
        with self._lock:
            return len(self._queue)

    @property
    def in_flight_count(self) -> int:
        """Number of tasks currently marked as in flight."""
        with self._lock:
            return len(self._in_flight)

    def submit(
        self,
        prompt: str,
        task_type: str = "rollout",
        device: str = "cpu",
        **kwargs: Any,
    ) -> str:
        """Create and store a task record for the newer submit/poll API."""
        task = RolloutTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            prompt=prompt,
            device=device,
            **kwargs,
        )
        with self._lock:
            self._tasks[task.task_id] = task
        return task.task_id

    def poll(self, task_id: str) -> RolloutTask | None:
        """Return the stored task record for the newer submit/poll API."""
        with self._lock:
            return self._tasks.get(task_id)

    def complete(self, task_id: str, result: str, reward: float | None = None) -> None:
        """Mark a submitted task as completed and persist the result."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = "completed"
            task.result = result
            if reward is not None:
                task.reward = reward
            elif self.reward_fn is not None:
                task.reward = self.reward_fn(task.prompt, result)

    def heartbeat(self, task_id: str) -> None:
        """Refresh a submitted task heartbeat."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.heartbeat = time.monotonic()

    def reap_stale(self) -> list[str]:
        """Mark stale submitted tasks as stale."""
        now = time.monotonic()
        stale: list[str] = []
        with self._lock:
            for task_id, task in self._tasks.items():
                if task.status == "running" and now - task.heartbeat > self.heartbeat_timeout:
                    task.status = "stale"
                    stale.append(task_id)
        return stale

    def active_count(self) -> int:
        """Return the number of pending or running submitted tasks."""
        with self._lock:
            return sum(1 for task in self._tasks.values() if task.status in ("pending", "running"))


class DirectDoubleSidedIS:
    """Direct double-sided importance sampling for asynchronous RL stability."""

    def __init__(self, clip_eps_low: float = 0.1, clip_eps_high: float = 10.0):
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high

    def mask(self, ratio: torch.Tensor) -> torch.Tensor:
        """Return a binary mask for tokens with acceptable IS ratios."""
        return ((ratio >= self.clip_eps_low) & (ratio <= self.clip_eps_high)).float()


def compute_parl_reward(
    performance_score: float,
    n_subagents: int,
    n_completed: int,
    lambda_parallel: float = 0.1,
    lambda_finish: float = 0.05,
) -> PARLReward:
    """Compute a PARL reward composition."""
    r_parallel = min(float(n_subagents) / 8.0, 1.0) if n_subagents > 1 else 0.0
    r_finish = float(n_completed) / max(n_subagents, 1) if n_subagents > 0 else 0.0
    return PARLReward(
        performance=performance_score,
        parallelism=lambda_parallel * r_parallel,
        finish_rate=lambda_finish * r_finish,
    )
