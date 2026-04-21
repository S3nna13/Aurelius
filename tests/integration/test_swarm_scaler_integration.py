"""Integration test for SwarmScaler — 50 mixed-priority tasks, 10 workers.

Validates:
- All 50 tasks complete
- 10 WorkerStats entries returned
- Work-stealing was active (tasks_stolen > 0)
- Registry wired (AGENT_LOOP_REGISTRY["swarm_scaler"] is SwarmScaler)
"""

from __future__ import annotations

import random

import pytest

from src.agent import AGENT_LOOP_REGISTRY
from src.agent.swarm_scaler import SwarmScaler, SwarmScalerConfig, WorkerStats


def _worker_fn(payload: dict) -> dict:
    """Simulate a small computation: square the value field."""
    return {"task_id": payload["task_id"], "value_sq": payload["value"] ** 2}


def test_swarm_scaler_integration():
    # --- setup ---------------------------------------------------------------
    cfg = SwarmScalerConfig(
        max_workers=300,
        max_total_steps=4000,
        work_steal_batch=4,
        enable_work_stealing=True,
    )
    scaler = SwarmScaler(config=cfg)

    # --- submit 50 tasks with mixed priorities (0, 5, 10) --------------------
    random.seed(42)
    payloads = [
        {"task_id": i, "value": i + 1}
        for i in range(50)
    ]
    priorities = [random.choice([0, 5, 10]) for _ in range(50)]

    ids = scaler.submit_tasks(payloads, priorities=priorities)
    assert len(ids) == 50, "Expected 50 task IDs back from submit_tasks"
    assert scaler.pending_count() == 50

    # --- dispatch with 10 workers --------------------------------------------
    completed = scaler.dispatch(_worker_fn, n_workers=10)

    # All 50 tasks must complete.
    assert len(completed) == 50, f"Expected 50 completed, got {len(completed)}"
    assert scaler.completed_count() == 50
    assert scaler.pending_count() == 0

    # --- results correctness -------------------------------------------------
    result_map = {item.result["task_id"]: item.result["value_sq"] for item in completed}
    for i in range(50):
        expected = (i + 1) ** 2
        assert result_map[i] == expected, (
            f"Task {i}: expected value_sq={expected}, got {result_map[i]}"
        )

    # --- worker stats: exactly 10 entries ------------------------------------
    stats = scaler.worker_stats()
    assert len(stats) == 10, f"Expected 10 WorkerStats, got {len(stats)}"
    assert all(isinstance(ws, WorkerStats) for ws in stats)
    worker_ids = {ws.worker_id for ws in stats}
    assert worker_ids == set(range(10)), "Worker IDs must be 0..9"

    # All tasks accounted for across workers.
    total_completed_in_stats = sum(ws.tasks_completed for ws in stats)
    assert total_completed_in_stats == 50

    # --- work-stealing was active --------------------------------------------
    total_stolen = sum(ws.tasks_stolen for ws in stats)
    assert total_stolen > 0, (
        "Expected work-stealing to have occurred (tasks_stolen > 0) "
        "with 50 tasks and 10 workers and work_steal_batch=4"
    )

    # --- speedup model sanity ------------------------------------------------
    speedup = scaler.speedup_vs_serial(serial_steps=50)
    # critical_path = ceil(50 / 10) = 5 → speedup = 50 / 5 = 10.0
    assert speedup == pytest.approx(10.0), f"Expected speedup=10.0, got {speedup}"

    # --- registry wired correctly --------------------------------------------
    assert "swarm_scaler" in AGENT_LOOP_REGISTRY, (
        "AGENT_LOOP_REGISTRY must contain 'swarm_scaler'"
    )
    assert AGENT_LOOP_REGISTRY["swarm_scaler"] is SwarmScaler, (
        "AGENT_LOOP_REGISTRY['swarm_scaler'] must be the SwarmScaler class"
    )

    # --- second dispatch accumulates total_steps_used ------------------------
    scaler.submit_tasks([{"task_id": 50 + i, "value": i} for i in range(5)])
    scaler.dispatch(_worker_fn, n_workers=5)
    assert scaler.total_steps_used() == 55

    # --- reset clears state --------------------------------------------------
    scaler.reset()
    assert scaler.pending_count() == 0
    assert scaler.completed_count() == 0
    assert scaler.total_steps_used() == 0
    assert scaler.worker_stats() == []
