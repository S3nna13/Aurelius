"""Unit tests for src.agent.background_executor (15 tests)."""

from __future__ import annotations

import time

import pytest

from src.agent.background_executor import (
    BackgroundExecutor,
    BackgroundExecutorConfig,
    BackgroundTask,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_executor(**kwargs) -> BackgroundExecutor:
    """Return a BackgroundExecutor with a custom config."""
    return BackgroundExecutor(BackgroundExecutorConfig(**kwargs))


def _noop() -> str:
    return "ok"


def _raise() -> None:
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = BackgroundExecutorConfig()
    assert cfg.max_queue_size == 1000
    assert cfg.heartbeat_interval_s == 30.0
    assert cfg.default_priority == 5
    assert cfg.stale_threshold_s == 60.0


# ---------------------------------------------------------------------------
# 2. Submit basic
# ---------------------------------------------------------------------------


def test_submit_basic():
    ex = BackgroundExecutor()
    task = ex.submit("t1", _noop)
    assert isinstance(task, BackgroundTask)
    assert task.task_id == "t1"
    assert task.status == TaskStatus.PENDING
    assert ex.pending_count() == 1


# ---------------------------------------------------------------------------
# 3. Submit duplicate raises
# ---------------------------------------------------------------------------


def test_submit_duplicate_raises():
    ex = BackgroundExecutor()
    ex.submit("t1", _noop)
    with pytest.raises(ValueError, match="already exists"):
        ex.submit("t1", _noop)


# ---------------------------------------------------------------------------
# 4. Submit queue full raises
# ---------------------------------------------------------------------------


def test_submit_queue_full_raises():
    ex = _make_executor(max_queue_size=2)
    ex.submit("a", _noop)
    ex.submit("b", _noop)
    with pytest.raises(ValueError, match="queue full"):
        ex.submit("c", _noop)


# ---------------------------------------------------------------------------
# 5. run_next completes successfully
# ---------------------------------------------------------------------------


def test_run_next_completes():
    ex = BackgroundExecutor()
    ex.submit("t1", lambda: 42)
    task = ex.run_next()
    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    assert task.result == 42
    assert task.error is None
    assert task.started_at is not None
    assert task.completed_at is not None
    assert ex.pending_count() == 0


# ---------------------------------------------------------------------------
# 6. run_next records failure
# ---------------------------------------------------------------------------


def test_run_next_fails():
    ex = BackgroundExecutor()
    ex.submit("bad", _raise)
    task = ex.run_next()
    assert task is not None
    assert task.status == TaskStatus.FAILED
    assert isinstance(task.error, str)
    assert "boom" in task.error
    assert task.result is None


# ---------------------------------------------------------------------------
# 7. run_next on empty queue returns None
# ---------------------------------------------------------------------------


def test_run_next_empty_returns_none():
    ex = BackgroundExecutor()
    assert ex.run_next() is None


# ---------------------------------------------------------------------------
# 8. Priority ordering
# ---------------------------------------------------------------------------


def test_priority_order():
    ex = BackgroundExecutor()
    results: list[int] = []
    ex.submit("low", lambda: results.append(10), priority=10)
    ex.submit("high", lambda: results.append(1), priority=1)

    ex.run_next()  # should pick priority=1
    ex.run_next()  # should pick priority=10

    assert results == [1, 10]


# ---------------------------------------------------------------------------
# 9. Cancel pending task
# ---------------------------------------------------------------------------


def test_cancel_pending():
    ex = BackgroundExecutor()
    ex.submit("t1", _noop)
    assert ex.cancel("t1") is True
    assert ex.get_status("t1") == TaskStatus.CANCELLED
    assert ex.pending_count() == 0


# ---------------------------------------------------------------------------
# 10. Cancel non-existent task
# ---------------------------------------------------------------------------


def test_cancel_nonexistent():
    ex = BackgroundExecutor()
    assert ex.cancel("ghost") is False


# ---------------------------------------------------------------------------
# 11. Heartbeat updates timestamp
# ---------------------------------------------------------------------------


def test_heartbeat_updates():
    ex = BackgroundExecutor()

    # Manually place a task in RUNNING state to simulate in-flight execution
    task = BackgroundTask(task_id="r1", priority=0, task_fn=_noop)
    task.status = TaskStatus.RUNNING
    task.heartbeat_at = time.time() - 100  # old
    ex._tasks["r1"] = task

    old_hb = task.heartbeat_at
    result = ex.heartbeat("r1")
    assert result is True
    assert task.heartbeat_at > old_hb


# ---------------------------------------------------------------------------
# 12. Heartbeat on non-running task returns False
# ---------------------------------------------------------------------------


def test_heartbeat_not_running():
    ex = BackgroundExecutor()
    ex.submit("t1", _noop)
    # t1 is PENDING, not RUNNING
    assert ex.heartbeat("t1") is False
    # also test unknown task
    assert ex.heartbeat("unknown") is False


# ---------------------------------------------------------------------------
# 13. Detect stale running tasks
# ---------------------------------------------------------------------------


def test_detect_stale():
    ex = _make_executor(stale_threshold_s=10.0)

    task = BackgroundTask(task_id="stale", priority=0, task_fn=_noop)
    task.status = TaskStatus.RUNNING
    task.heartbeat_at = time.time() - 20  # older than threshold
    ex._tasks["stale"] = task

    stale_ids = ex.detect_stale()
    assert "stale" in stale_ids


# ---------------------------------------------------------------------------
# 14. Detect stale — fresh heartbeat not flagged
# ---------------------------------------------------------------------------


def test_detect_stale_fresh():
    ex = _make_executor(stale_threshold_s=60.0)

    task = BackgroundTask(task_id="fresh", priority=0, task_fn=_noop)
    task.status = TaskStatus.RUNNING
    task.heartbeat_at = time.time()  # just now
    ex._tasks["fresh"] = task

    stale_ids = ex.detect_stale()
    assert "fresh" not in stale_ids


# ---------------------------------------------------------------------------
# 15. queue_snapshot returns tasks in priority order
# ---------------------------------------------------------------------------


def test_queue_snapshot_priority_order():
    ex = BackgroundExecutor()
    ex.submit("p5", _noop, priority=5)
    ex.submit("p1", _noop, priority=1)
    ex.submit("p3", _noop, priority=3)

    snapshot = ex.queue_snapshot()
    priorities = [t.priority for t in snapshot]
    assert priorities == sorted(priorities)
    assert [t.task_id for t in snapshot] == ["p1", "p3", "p5"]
