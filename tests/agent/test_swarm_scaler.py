"""Unit tests for src/agent/swarm_scaler.py (10-16 tests, Cycle 128-E)."""

from __future__ import annotations

import pytest

from src.agent.swarm_scaler import (
    SwarmScaler,
    SwarmScalerConfig,
    WorkItem,
    WorkerStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def identity(x):
    return x


def double(x):
    return x * 2


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = SwarmScalerConfig()
    assert cfg.max_workers == 300
    assert cfg.max_total_steps == 4000
    assert cfg.work_steal_batch == 4
    assert cfg.enable_work_stealing is True


# ---------------------------------------------------------------------------
# 2. Submit tasks basic
# ---------------------------------------------------------------------------

def test_submit_tasks_basic():
    scaler = SwarmScaler()
    scaler.submit_tasks([10, 20, 30, 40, 50])
    assert scaler.pending_count() == 5


# ---------------------------------------------------------------------------
# 3. Submit tasks returns sequential IDs starting at 0
# ---------------------------------------------------------------------------

def test_submit_tasks_returns_ids():
    scaler = SwarmScaler()
    ids = scaler.submit_tasks(["a", "b", "c"])
    assert ids == [0, 1, 2]


# ---------------------------------------------------------------------------
# 4. Submitting beyond max_total_steps raises ValueError
# ---------------------------------------------------------------------------

def test_submit_exceeds_max_steps_raises():
    cfg = SwarmScalerConfig(max_total_steps=5)
    scaler = SwarmScaler(config=cfg)
    scaler.submit_tasks([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="max_total_steps"):
        scaler.submit_tasks([6])


# ---------------------------------------------------------------------------
# 5. Dispatch completes all submitted tasks
# ---------------------------------------------------------------------------

def test_dispatch_completes_all():
    scaler = SwarmScaler()
    scaler.submit_tasks(list(range(8)))
    completed = scaler.dispatch(identity)
    assert len(completed) == 8
    assert scaler.completed_count() == 8
    assert scaler.pending_count() == 0


# ---------------------------------------------------------------------------
# 6. Worker function result is stored correctly (doubles)
# ---------------------------------------------------------------------------

def test_dispatch_result_correct():
    scaler = SwarmScaler()
    scaler.submit_tasks([1, 2, 3, 4])
    completed = scaler.dispatch(double)
    results = sorted(item.result for item in completed)
    assert results == [2, 4, 6, 8]


# ---------------------------------------------------------------------------
# 7. Dispatch respects n_workers parameter
# ---------------------------------------------------------------------------

def test_dispatch_respects_n_workers():
    scaler = SwarmScaler()
    scaler.submit_tasks([10, 20, 30, 40])
    completed = scaler.dispatch(identity, n_workers=2)
    assert len(completed) == 4


# ---------------------------------------------------------------------------
# 8. n_workers > max_workers raises ValueError
# ---------------------------------------------------------------------------

def test_dispatch_max_workers_exceeded_raises():
    cfg = SwarmScalerConfig(max_workers=5)
    scaler = SwarmScaler(config=cfg)
    scaler.submit_tasks([1, 2, 3])
    with pytest.raises(ValueError, match="max_workers"):
        scaler.dispatch(identity, n_workers=10)


# ---------------------------------------------------------------------------
# 9. Priority order: lower priority value processed first
# ---------------------------------------------------------------------------

def test_priority_order():
    scaler = SwarmScaler(config=SwarmScalerConfig(enable_work_stealing=False))
    # Submit high-priority (0) and low-priority (10) tasks interleaved.
    scaler.submit_tasks(
        payloads=["hi-0", "lo-0", "hi-1", "lo-1"],
        priorities=[0, 10, 0, 10],
    )
    completed = scaler.dispatch(identity, n_workers=1)
    # With one worker and no stealing, tasks execute in priority order.
    ordered_payloads = [item.payload for item in completed]
    hi_indices = [i for i, p in enumerate(ordered_payloads) if p.startswith("hi")]
    lo_indices = [i for i, p in enumerate(ordered_payloads) if p.startswith("lo")]
    # All high-priority items should appear before all low-priority items.
    assert max(hi_indices) < min(lo_indices)


# ---------------------------------------------------------------------------
# 10. Work-stealing enabled: tasks_stolen > 0
# ---------------------------------------------------------------------------

def test_work_stealing_enabled():
    cfg = SwarmScalerConfig(
        max_workers=300,
        work_steal_batch=4,
        enable_work_stealing=True,
    )
    scaler = SwarmScaler(config=cfg)
    # Submit enough tasks so that early workers steal from later ones.
    scaler.submit_tasks(list(range(20)))
    scaler.dispatch(identity, n_workers=5)
    stats = scaler.worker_stats()
    total_stolen = sum(ws.tasks_stolen for ws in stats)
    assert total_stolen > 0


# ---------------------------------------------------------------------------
# 11. Work-stealing disabled: tasks_stolen == 0
# ---------------------------------------------------------------------------

def test_work_stealing_disabled():
    cfg = SwarmScalerConfig(enable_work_stealing=False)
    scaler = SwarmScaler(config=cfg)
    scaler.submit_tasks(list(range(20)))
    scaler.dispatch(identity, n_workers=5)
    stats = scaler.worker_stats()
    total_stolen = sum(ws.tasks_stolen for ws in stats)
    assert total_stolen == 0


# ---------------------------------------------------------------------------
# 12. worker_stats returns one entry per active worker
# ---------------------------------------------------------------------------

def test_worker_stats_count():
    scaler = SwarmScaler()
    scaler.submit_tasks(list(range(6)))
    scaler.dispatch(identity, n_workers=3)
    stats = scaler.worker_stats()
    assert len(stats) == 3
    worker_ids = {ws.worker_id for ws in stats}
    assert worker_ids == {0, 1, 2}


# ---------------------------------------------------------------------------
# 13. total_steps_used accumulates across dispatches
# ---------------------------------------------------------------------------

def test_total_steps_used():
    scaler = SwarmScaler()
    scaler.submit_tasks([1, 2, 3])
    scaler.dispatch(identity)
    assert scaler.total_steps_used() == 3

    scaler.submit_tasks([4, 5])
    scaler.dispatch(identity)
    assert scaler.total_steps_used() == 5


# ---------------------------------------------------------------------------
# 14. reset clears everything
# ---------------------------------------------------------------------------

def test_reset_clears_all():
    scaler = SwarmScaler()
    scaler.submit_tasks([1, 2, 3])
    scaler.dispatch(identity)
    scaler.reset()
    assert scaler.pending_count() == 0
    assert scaler.completed_count() == 0
    assert scaler.worker_stats() == []
    assert scaler.total_steps_used() == 0


# ---------------------------------------------------------------------------
# 15. speedup_vs_serial calculation
# ---------------------------------------------------------------------------

def test_speedup_calculation():
    import math
    scaler = SwarmScaler()
    # 10 tasks, dispatch with 10 workers → critical_path = ceil(10/10) = 1
    scaler.submit_tasks(list(range(10)))
    scaler.dispatch(identity, n_workers=10)
    speedup = scaler.speedup_vs_serial(serial_steps=100)
    # critical_path = ceil(10 / 10) = 1 → speedup = 100 / 1 = 100.0
    assert speedup == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 16. WorkItem fields are populated after dispatch
# ---------------------------------------------------------------------------

def test_work_item_fields_populated():
    scaler = SwarmScaler()
    scaler.submit_tasks([42])
    completed = scaler.dispatch(identity)
    item = completed[0]
    assert isinstance(item, WorkItem)
    assert item.task_id == 0
    assert item.result == 42
    assert item.worker_id is not None
    assert item.started_at is not None
    assert item.completed_at is not None
    assert item.completed_at >= item.started_at
