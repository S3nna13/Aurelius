"""Unit tests for src/training/async_rl_infra.py — GLM-5 §4.1.

Coverage targets (12 tests):
 1.  test_enqueue_adds_to_queue
 2.  test_dispatch_returns_results
 3.  test_result_task_ids_match
 4.  test_completed_status
 5.  test_error_status
 6.  test_queue_drains_after_dispatch
 7.  test_in_flight_clears_after_dispatch
 8.  test_requeue_stale_when_stuck
 9.  test_requeue_fresh_not_requeued
10.  test_batch_size_limit
11.  test_empty_queue
12.  test_heterogeneous_task_types
"""

from __future__ import annotations

import time

from src.training.async_rl_infra import RolloutOrchestrator, RolloutResult, RolloutTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(task_id: int, task_type: str = "swe", prompt: str = "hello") -> RolloutTask:
    return RolloutTask(task_id=task_id, task_type=task_type, prompt=prompt)


def _success_fn(task: RolloutTask) -> RolloutResult:
    """Dummy rollout function that always succeeds."""
    return RolloutResult(
        task_id=task.task_id,
        task_type=task.task_type,
        completion=f"result-{task.task_id}",
        reward=1.0,
        tokens_used=10,
        status="completed",
    )


def _error_fn(task: RolloutTask) -> RolloutResult:
    """Dummy rollout function that always raises."""
    raise RuntimeError("inference failure")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnqueue:
    def test_enqueue_adds_to_queue(self) -> None:
        """Enqueueing 3 tasks raises queue_depth to 3."""
        orch = RolloutOrchestrator()
        tasks = [_make_task(i) for i in range(3)]
        orch.enqueue(tasks)
        assert orch.queue_depth == 3


class TestDispatch:
    def test_dispatch_returns_results(self) -> None:
        """dispatch_batch with 3 tasks returns exactly 3 results."""
        orch = RolloutOrchestrator()
        orch.enqueue([_make_task(i) for i in range(3)])
        results = orch.dispatch_batch(_success_fn, batch_size=10)
        assert len(results) == 3

    def test_result_task_ids_match(self) -> None:
        """Each result.task_id equals the corresponding task.task_id."""
        orch = RolloutOrchestrator()
        tasks = [_make_task(i) for i in range(3)]
        orch.enqueue(tasks)
        results = orch.dispatch_batch(_success_fn, batch_size=10)
        result_ids = {r.task_id for r in results}
        task_ids = {t.task_id for t in tasks}
        assert result_ids == task_ids

    def test_completed_status(self) -> None:
        """A successful rollout_fn produces status=='completed'."""
        orch = RolloutOrchestrator()
        orch.enqueue([_make_task(0)])
        results = orch.dispatch_batch(_success_fn)
        assert results[0].status == "completed"

    def test_error_status(self) -> None:
        """A rollout_fn that raises produces status=='error' without propagating."""
        orch = RolloutOrchestrator()
        orch.enqueue([_make_task(99)])
        results = orch.dispatch_batch(_error_fn)
        assert len(results) == 1
        assert results[0].status == "error"
        assert results[0].task_id == 99

    def test_queue_drains_after_dispatch(self) -> None:
        """After dispatching all tasks the queue is empty."""
        orch = RolloutOrchestrator()
        orch.enqueue([_make_task(i) for i in range(5)])
        orch.dispatch_batch(_success_fn, batch_size=10)
        assert orch.queue_depth == 0

    def test_in_flight_clears_after_dispatch(self) -> None:
        """In-flight count returns to 0 after dispatch completes."""
        orch = RolloutOrchestrator()
        orch.enqueue([_make_task(i) for i in range(4)])
        orch.dispatch_batch(_success_fn, batch_size=10)
        assert orch.in_flight_count == 0

    def test_batch_size_limit(self) -> None:
        """dispatch_batch with batch_size=3 returns 3 results; 7 remain queued."""
        orch = RolloutOrchestrator()
        orch.enqueue([_make_task(i) for i in range(10)])
        results = orch.dispatch_batch(_success_fn, batch_size=3)
        assert len(results) == 3
        assert orch.queue_depth == 7

    def test_empty_queue(self) -> None:
        """Dispatching on an empty queue returns [] without raising."""
        orch = RolloutOrchestrator()
        results = orch.dispatch_batch(_success_fn)
        assert results == []

    def test_heterogeneous_task_types(self) -> None:
        """Mix of swe/terminal/search tasks are all dispatched correctly."""
        orch = RolloutOrchestrator()
        tasks = [
            _make_task(0, task_type="swe"),
            _make_task(1, task_type="terminal"),
            _make_task(2, task_type="search"),
        ]
        orch.enqueue(tasks)
        results = orch.dispatch_batch(_success_fn, batch_size=10)
        result_types = {r.task_type for r in results}
        assert result_types == {"swe", "terminal", "search"}


class TestHeartbeat:
    def test_requeue_stale_when_stuck(self) -> None:
        """Manually inserting a stale in-flight entry causes requeue_stale() == 1."""
        orch = RolloutOrchestrator(heartbeat_timeout=0.05)
        task = _make_task(42)
        # Simulate a task stuck in-flight with an old timestamp
        stale_ts = time.monotonic() - 1.0  # 1 second ago → well past 50 ms timeout
        orch._in_flight[task.task_id] = (task, stale_ts)
        count = orch.requeue_stale()
        assert count == 1
        assert orch.queue_depth == 1
        assert orch.in_flight_count == 0

    def test_requeue_fresh_not_requeued(self) -> None:
        """A freshly registered in-flight task is NOT re-queued."""
        orch = RolloutOrchestrator(heartbeat_timeout=60.0)
        task = _make_task(7)
        orch._in_flight[task.task_id] = (task, time.monotonic())
        count = orch.requeue_stale()
        assert count == 0
        assert orch.in_flight_count == 1
