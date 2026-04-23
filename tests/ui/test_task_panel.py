"""Tests for src.ui.task_panel."""

from __future__ import annotations

import time

import pytest
from rich.console import Console

from src.ui.task_panel import (
    TASK_PANEL_REGISTRY,
    TaskEntry,
    TaskPanel,
    TaskPanelError,
)


def _make_entry(
    task_id: str = "task-1",
    title: str = "Do something",
    status: str = "pending",
    priority: int = 0,
    tags: list[str] | None = None,
) -> TaskEntry:
    return TaskEntry(
        task_id=task_id,
        title=title,
        status=status,
        priority=priority,
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# add_task / filter_by_status
# ---------------------------------------------------------------------------


def test_add_task_retrievable_via_filter() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("t1", status="running"))
    results = panel.filter_by_status("running")
    assert len(results) == 1
    assert results[0].task_id == "t1"


def test_add_task_increases_count() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("a"))
    panel.add_task(_make_entry("b", status="done"))
    assert len(panel.tasks) == 2


def test_add_task_invalid_raises() -> None:
    panel = TaskPanel()
    with pytest.raises(TaskPanelError):
        panel.add_task("not a TaskEntry")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# update_task
# ---------------------------------------------------------------------------


def test_update_task_changes_field() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("t2", status="pending"))
    panel.update_task("t2", status="running")
    assert panel.tasks["t2"].status == "running"


def test_update_task_changes_progress() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("t3"))
    panel.update_task("t3", progress=0.5)
    assert panel.tasks["t3"].progress == pytest.approx(0.5)


def test_update_task_unknown_id_raises_key_error() -> None:
    panel = TaskPanel()
    with pytest.raises(KeyError):
        panel.update_task("does-not-exist", status="done")


def test_update_task_invalid_field_raises_task_panel_error() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("t4"))
    with pytest.raises(TaskPanelError):
        panel.update_task("t4", nonexistent_field="oops")


def test_update_task_multiple_fields() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("t5", title="Old"))
    panel.update_task("t5", title="New", status="done", progress=1.0)
    entry = panel.tasks["t5"]
    assert entry.title == "New"
    assert entry.status == "done"
    assert entry.progress == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# remove_task
# ---------------------------------------------------------------------------


def test_remove_task_removes_entry() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("r1"))
    panel.remove_task("r1")
    assert "r1" not in panel.tasks


def test_remove_task_unknown_id_raises_key_error() -> None:
    panel = TaskPanel()
    with pytest.raises(KeyError):
        panel.remove_task("ghost")


# ---------------------------------------------------------------------------
# filter_by_status
# ---------------------------------------------------------------------------


def test_filter_by_status_returns_only_matching() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("a", status="running"))
    panel.add_task(_make_entry("b", status="pending"))
    panel.add_task(_make_entry("c", status="running"))
    results = panel.filter_by_status("running")
    assert len(results) == 2
    ids = {e.task_id for e in results}
    assert ids == {"a", "c"}


def test_filter_by_status_no_match_returns_empty() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("x", status="done"))
    results = panel.filter_by_status("running")
    assert results == []


def test_filter_by_status_sorted_by_priority() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("lo", status="running", priority=10))
    panel.add_task(_make_entry("hi", status="running", priority=1))
    results = panel.filter_by_status("running")
    assert results[0].task_id == "hi"
    assert results[1].task_id == "lo"


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_zero_tasks_does_not_crash() -> None:
    panel = TaskPanel()
    console = Console(record=True)
    panel.render(console)
    assert True


def test_render_three_tasks_does_not_crash() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("1", title="Alpha", status="pending"))
    panel.add_task(_make_entry("2", title="Beta", status="running"))
    panel.add_task(_make_entry("3", title="Gamma", status="done"))
    console = Console(record=True)
    panel.render(console)
    output = console.export_text()
    assert "Alpha" in output
    assert "Beta" in output
    assert "Gamma" in output


def test_render_hide_completed() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("done1", title="Finished", status="done"))
    panel.add_task(_make_entry("run1", title="Active", status="running"))
    console = Console(record=True)
    panel.render(console, show_completed=False)
    output = console.export_text()
    assert "Active" in output
    assert "Finished" not in output


def test_render_with_progress_does_not_crash() -> None:
    panel = TaskPanel()
    entry = TaskEntry(task_id="p1", title="With progress", status="running", progress=0.75)
    panel.add_task(entry)
    console = Console(record=True)
    panel.render(console)
    output = console.export_text()
    assert "p1" in output or "With progress" in output


def test_render_with_tags_shows_tags() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("tg1", title="Tagged", tags=["alpha", "beta"]))
    console = Console(record=True)
    panel.render(console)
    output = console.export_text()
    assert "alpha" in output


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_returns_dict() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("d1"))
    result = panel.to_dict()
    assert isinstance(result, dict)


def test_to_dict_contains_task_ids() -> None:
    panel = TaskPanel()
    panel.add_task(_make_entry("x1"))
    panel.add_task(_make_entry("x2", status="done"))
    d = panel.to_dict()
    assert "x1" in d
    assert "x2" in d


def test_to_dict_empty_panel_returns_empty_dict() -> None:
    panel = TaskPanel()
    assert panel.to_dict() == {}


# ---------------------------------------------------------------------------
# TASK_PANEL_REGISTRY
# ---------------------------------------------------------------------------


def test_task_panel_registry_is_dict() -> None:
    assert isinstance(TASK_PANEL_REGISTRY, dict)


def test_task_panel_registry_supports_assignment() -> None:
    TASK_PANEL_REGISTRY["test-panel"] = TaskPanel()
    assert "test-panel" in TASK_PANEL_REGISTRY
    del TASK_PANEL_REGISTRY["test-panel"]
