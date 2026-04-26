"""Integration tests for Async RL Infrastructure — GLM-5 §4.1.

Checks:
 1. ``TRAINING_REGISTRY["async_rl"]`` is registered in src/training/__init__.py
 2. Constructing an orchestrator, enqueueing 5 tasks, and dispatching all
    returns exactly 5 results.
 3. Pre-existing AUXILIARY_LOSS_REGISTRY key is still present (regression guard).
"""

from __future__ import annotations

from src.training import AUXILIARY_LOSS_REGISTRY, TRAINING_REGISTRY
from src.training.async_rl_infra import RolloutOrchestrator, RolloutResult, RolloutTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(task_id: int, task_type: str = "swe") -> RolloutTask:
    return RolloutTask(task_id=task_id, task_type=task_type, prompt=f"prompt-{task_id}")


def _success_fn(task: RolloutTask) -> RolloutResult:
    return RolloutResult(
        task_id=task.task_id,
        task_type=task.task_type,
        completion=f"done-{task.task_id}",
        reward=1.0,
        tokens_used=5,
        status="completed",
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestTrainingRegistry:
    def test_async_rl_in_training_registry(self) -> None:
        """'async_rl' key must be present in TRAINING_REGISTRY."""
        assert "async_rl" in TRAINING_REGISTRY

    def test_async_rl_registry_value_is_rollout_orchestrator(self) -> None:
        """The registered class must be RolloutOrchestrator."""
        assert TRAINING_REGISTRY["async_rl"] is RolloutOrchestrator

    def test_auxiliary_loss_registry_regression(self) -> None:
        """Pre-existing AUXILIARY_LOSS_REGISTRY['tool_call_supervision'] is intact."""
        assert "tool_call_supervision" in AUXILIARY_LOSS_REGISTRY


class TestEndToEnd:
    def test_enqueue_and_dispatch_five_tasks(self) -> None:
        """Construct orchestrator, enqueue 5 tasks, dispatch all → 5 results."""
        OrchestratorClass = TRAINING_REGISTRY["async_rl"]
        orch: RolloutOrchestrator = OrchestratorClass(heartbeat_timeout=30.0, max_concurrent=1000)
        tasks = [_make_task(i) for i in range(5)]
        orch.enqueue(tasks)
        assert orch.queue_depth == 5

        results = orch.dispatch_batch(_success_fn, batch_size=10)
        assert len(results) == 5
        assert all(r.status == "completed" for r in results)
        assert orch.queue_depth == 0
        assert orch.in_flight_count == 0

    def test_result_ids_cover_all_tasks(self) -> None:
        """Result task_ids are exactly {0,1,2,3,4}."""
        orch = TRAINING_REGISTRY["async_rl"]()
        orch.enqueue([_make_task(i) for i in range(5)])
        results = orch.dispatch_batch(_success_fn, batch_size=10)
        assert {r.task_id for r in results} == {0, 1, 2, 3, 4}

    def test_trainer_async_rl_branch_returns_orchestrator(self) -> None:
        """build_async_rl_orchestrator(enabled=True) returns a RolloutOrchestrator."""
        from src.training.trainer import build_async_rl_orchestrator

        orch = build_async_rl_orchestrator(enabled=True)
        assert isinstance(orch, RolloutOrchestrator)

    def test_trainer_async_rl_branch_disabled_returns_none(self) -> None:
        """build_async_rl_orchestrator(enabled=False) returns None (feature flag off)."""
        from src.training.trainer import build_async_rl_orchestrator

        result = build_async_rl_orchestrator(enabled=False)
        assert result is None
