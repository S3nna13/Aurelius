"""Tests for src/ui/progress_renderer.py — ProgressTask, ETAEstimator, ProgressRenderer."""

from __future__ import annotations

import time

import pytest
from rich.console import Console

from src.ui.progress_renderer import (
    PROGRESS_RENDERER_REGISTRY,
    ETAEstimator,
    ProgressError,
    ProgressRenderer,
    ProgressTask,
)


# ---------------------------------------------------------------------------
# ProgressTask defaults
# ---------------------------------------------------------------------------


def test_progress_task_defaults() -> None:
    t = ProgressTask(task_id="t1", description="Build", total=100)
    assert t.completed == 0
    assert t.visible is True
    assert t.metadata == {}


def test_progress_task_none_total() -> None:
    t = ProgressTask(task_id="t2", description="Spin", total=None)
    assert t.total is None


# ---------------------------------------------------------------------------
# ETAEstimator
# ---------------------------------------------------------------------------


def test_eta_estimator_eta_none_with_zero_samples() -> None:
    est = ETAEstimator()
    assert est.eta_seconds(100) is None


def test_eta_estimator_eta_none_with_one_sample() -> None:
    est = ETAEstimator()
    est.record(5)
    assert est.eta_seconds(100) is None


def test_eta_estimator_eta_float_with_three_samples() -> None:
    est = ETAEstimator()
    est.record(0)
    time.sleep(0.01)
    est.record(10)
    time.sleep(0.01)
    est.record(20)
    result = est.eta_seconds(100)
    assert isinstance(result, float)
    assert result > 0


def test_eta_estimator_throughput_none_with_one_sample() -> None:
    est = ETAEstimator()
    est.record(10)
    assert est.throughput() is None


def test_eta_estimator_throughput_float_with_two_samples() -> None:
    est = ETAEstimator()
    est.record(0)
    time.sleep(0.02)
    est.record(10)
    result = est.throughput()
    assert isinstance(result, float)
    assert result > 0


def test_eta_estimator_ring_buffer_cap() -> None:
    est = ETAEstimator()
    for i in range(150):
        est.record(i)
    assert len(est._samples) == 100


def test_eta_estimator_eta_zero_when_complete() -> None:
    est = ETAEstimator()
    est.record(0)
    time.sleep(0.01)
    est.record(100)
    result = est.eta_seconds(100)
    # 100 completed == 100 total → remaining=0
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ProgressRenderer — add_task, advance
# ---------------------------------------------------------------------------


def test_add_task_registers_task() -> None:
    renderer = ProgressRenderer()
    task = ProgressTask(task_id="r1", description="Train", total=1000)
    renderer.add_task(task)
    assert "r1" in renderer._tasks


def test_advance_increments_completed() -> None:
    renderer = ProgressRenderer()
    task = ProgressTask(task_id="r2", description="Step", total=50)
    renderer.add_task(task)
    renderer.advance("r2", delta=5)
    assert renderer._tasks["r2"].completed == 5


def test_advance_default_delta_is_one() -> None:
    renderer = ProgressRenderer()
    task = ProgressTask(task_id="r3", description="Step", total=50)
    renderer.add_task(task)
    renderer.advance("r3")
    assert renderer._tasks["r3"].completed == 1


def test_advance_unknown_task_raises() -> None:
    renderer = ProgressRenderer()
    with pytest.raises(ProgressError, match="not found"):
        renderer.advance("ghost")


def test_advance_records_sample() -> None:
    renderer = ProgressRenderer()
    task = ProgressTask(task_id="r4", description="Sample", total=10)
    renderer.add_task(task)
    renderer.advance("r4", delta=3)
    assert len(renderer._estimators["r4"]._samples) == 1


# ---------------------------------------------------------------------------
# ProgressRenderer — complete, remove_task
# ---------------------------------------------------------------------------


def test_complete_sets_completed_to_total() -> None:
    renderer = ProgressRenderer()
    task = ProgressTask(task_id="r5", description="Done", total=200)
    renderer.add_task(task)
    renderer.advance("r5", delta=50)
    renderer.complete("r5")
    assert renderer._tasks["r5"].completed == 200


def test_complete_unknown_task_raises() -> None:
    renderer = ProgressRenderer()
    with pytest.raises(ProgressError, match="not found"):
        renderer.complete("missing")


def test_remove_task_removes_correctly() -> None:
    renderer = ProgressRenderer()
    task = ProgressTask(task_id="r6", description="Remove me", total=10)
    renderer.add_task(task)
    renderer.remove_task("r6")
    assert "r6" not in renderer._tasks
    assert "r6" not in renderer._estimators


def test_remove_task_unknown_raises() -> None:
    renderer = ProgressRenderer()
    with pytest.raises(ProgressError, match="not found"):
        renderer.remove_task("phantom")


# ---------------------------------------------------------------------------
# ProgressRenderer — render
# ---------------------------------------------------------------------------


def test_render_does_not_crash_with_one_task() -> None:
    renderer = ProgressRenderer()
    task = ProgressTask(task_id="r7", description="Render Test", total=100)
    renderer.add_task(task)
    renderer.advance("r7", delta=30)

    console = Console(record=True)
    renderer.render(console)
    output = console.export_text()
    assert "Render Test" in output


def test_render_shows_progress_bar_characters() -> None:
    renderer = ProgressRenderer()
    task = ProgressTask(task_id="r8", description="Bar", total=10)
    renderer.add_task(task)
    renderer.advance("r8", delta=5)

    console = Console(record=True)
    renderer.render(console)
    output = console.export_text()
    assert "■" in output or "□" in output


def test_render_empty_renderer_does_not_crash() -> None:
    renderer = ProgressRenderer()
    console = Console(record=True)
    renderer.render(console)
    output = console.export_text()
    assert "no active tasks" in output


def test_render_indeterminate_task_does_not_crash() -> None:
    renderer = ProgressRenderer()
    task = ProgressTask(task_id="r9", description="Spin", total=None)
    renderer.add_task(task)
    console = Console(record=True)
    renderer.render(console)
    output = console.export_text()
    assert "Spin" in output


# ---------------------------------------------------------------------------
# ProgressRenderer — render_summary
# ---------------------------------------------------------------------------


def test_render_summary_does_not_crash() -> None:
    renderer = ProgressRenderer()
    task = ProgressTask(task_id="r10", description="Summary Test", total=10)
    renderer.add_task(task)
    renderer.complete("r10")
    console = Console(record=True)
    renderer.render_summary(console)
    output = console.export_text()
    assert len(output) > 0


def test_render_summary_counts_completed() -> None:
    renderer = ProgressRenderer()
    for i in range(3):
        t = ProgressTask(task_id=f"rs{i}", description=f"Task {i}", total=10)
        renderer.add_task(t)
    renderer.complete("rs0")
    renderer.complete("rs1")
    console = Console(record=True)
    renderer.render_summary(console)
    output = console.export_text()
    assert "3" in output  # total tasks
    assert "2" in output  # completed


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_progress_renderer_registry_is_dict() -> None:
    assert isinstance(PROGRESS_RENDERER_REGISTRY, dict)


def test_progress_renderer_registry_accepts_entries() -> None:
    PROGRESS_RENDERER_REGISTRY["test-renderer"] = ProgressRenderer()
    assert "test-renderer" in PROGRESS_RENDERER_REGISTRY
    del PROGRESS_RENDERER_REGISTRY["test-renderer"]
