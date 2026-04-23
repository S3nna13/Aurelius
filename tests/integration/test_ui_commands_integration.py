"""Integration tests: command palette, status hierarchy, and keyboard nav.

Verifies cross-module contracts from the integration_requirements spec.
"""

from __future__ import annotations

import pytest
from rich.console import Console

import src.ui as ui
from src.ui.command_palette import COMMAND_PALETTE_REGISTRY, CommandPalette
from src.ui.status_hierarchy import (
    STATUS_TREE_REGISTRY,
    StatusLevel,
    StatusNode,
    StatusState,
    StatusTree,
)
from src.ui.keyboard_nav import KeyboardNav, KEYBOARD_NAV_REGISTRY


# ---------------------------------------------------------------------------
# src.ui namespace exposes COMMAND_PALETTE_REGISTRY
# ---------------------------------------------------------------------------


def test_command_palette_registry_accessible_via_src_ui() -> None:
    """COMMAND_PALETTE_REGISTRY must be importable from src.ui."""
    assert hasattr(ui, "COMMAND_PALETTE_REGISTRY")
    assert isinstance(ui.COMMAND_PALETTE_REGISTRY, dict)


def test_command_palette_registry_has_builtins_via_src_ui() -> None:
    reg = ui.COMMAND_PALETTE_REGISTRY
    assert "clear" in reg
    assert "quit" in reg
    assert "help" in reg


# ---------------------------------------------------------------------------
# StatusTree: create, populate, render
# ---------------------------------------------------------------------------


def test_status_tree_three_nodes_render_no_crash() -> None:
    """Create a StatusTree with 3 nodes and render without crashing."""
    tree = StatusTree()
    session_node = StatusNode(
        id="sess-integ",
        level=StatusLevel.SESSION,
        state=StatusState.RUNNING,
        label="Integration Session",
    )
    workstream_node = StatusNode(
        id="ws-integ",
        level=StatusLevel.WORKSTREAM,
        state=StatusState.RUNNING,
        label="Main Workstream",
    )
    task_node = StatusNode(
        id="task-integ",
        level=StatusLevel.TASK,
        state=StatusState.IDLE,
        label="Pending Task",
    )
    tree.add_node(session_node)
    tree.add_node(workstream_node, parent_id="sess-integ")
    tree.add_node(task_node, parent_id="ws-integ")

    console = Console(record=True)
    tree.render(console)
    output = console.export_text()
    assert "Integration Session" in output


def test_status_tree_update_state_integration() -> None:
    tree = StatusTree()
    tree.add_node(StatusNode(
        id="check-node",
        level=StatusLevel.TASK,
        state=StatusState.IDLE,
        label="Check",
    ))
    tree.update_state("check-node", StatusState.SUCCESS, progress=1.0)
    node = tree.get_node("check-node")
    assert node.state == StatusState.SUCCESS
    assert node.progress == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# KeyboardNav dispatch "/" → "command_palette"
# ---------------------------------------------------------------------------


def test_keyboard_nav_slash_dispatches_command_palette() -> None:
    action = KeyboardNav.dispatch("/")
    assert action == "command_palette"


# ---------------------------------------------------------------------------
# src.ui __all__ contains new symbols
# ---------------------------------------------------------------------------


def test_src_ui_all_contains_command_palette() -> None:
    assert "CommandPalette" in ui.__all__
    assert "COMMAND_PALETTE_REGISTRY" in ui.__all__


def test_src_ui_all_contains_status_tree() -> None:
    assert "StatusTree" in ui.__all__
    assert "STATUS_TREE_REGISTRY" in ui.__all__


def test_src_ui_all_contains_keyboard_nav() -> None:
    assert "KeyboardNav" in ui.__all__
    assert "KEYBOARD_NAV_REGISTRY" in ui.__all__


def test_src_ui_all_contains_onboarding_flow() -> None:
    assert "OnboardingFlow" in ui.__all__
    assert "ONBOARDING_REGISTRY" in ui.__all__


# ---------------------------------------------------------------------------
# TranscriptViewer integration
# ---------------------------------------------------------------------------


def test_transcript_viewer_render() -> None:
    """Create a viewer, add 2 entries, render, and export_text."""
    from src.ui.transcript_viewer import TranscriptEntry, TranscriptRole, TranscriptViewer

    viewer = TranscriptViewer()
    viewer.add_entry(TranscriptEntry(role=TranscriptRole.USER, content="Hello Aurelius"))
    viewer.add_entry(TranscriptEntry(role=TranscriptRole.ASSISTANT, content="Hello user"))

    console = Console(record=True)
    viewer.render(console)
    output = console.export_text()
    assert "Hello Aurelius" in output
    assert "Hello user" in output

    text = viewer.export_text()
    assert "USER:" in text
    assert "ASSISTANT:" in text
    assert len(text) > 0


# ---------------------------------------------------------------------------
# DiffViewer integration
# ---------------------------------------------------------------------------


def test_diff_viewer_parse_and_render() -> None:
    """Parse a 5-line diff, render to console."""
    from src.ui.diff_viewer import DiffViewer, parse_unified_diff

    diff_text = (
        "--- a/sample.py\n"
        "+++ b/sample.py\n"
        "@@ -1,3 +1,3 @@\n"
        " line one\n"
        "-line two old\n"
        "+line two new\n"
        " line three\n"
    )
    diff = parse_unified_diff(diff_text)
    assert len(diff.chunks) == 1

    viewer = DiffViewer()
    console = Console(record=True)
    viewer.render_diff(console, diff)
    output = console.export_text()
    assert "sample.py" in output or "line" in output


# ---------------------------------------------------------------------------
# TaskPanel integration
# ---------------------------------------------------------------------------


def test_task_panel_lifecycle() -> None:
    """Add, update, filter, render, to_dict."""
    from src.ui.task_panel import TaskEntry, TaskPanel

    panel = TaskPanel()
    panel.add_task(TaskEntry(task_id="integ-1", title="Build", status="running"))
    panel.add_task(TaskEntry(task_id="integ-2", title="Test", status="pending"))
    panel.add_task(TaskEntry(task_id="integ-3", title="Deploy", status="running"))

    panel.update_task("integ-2", status="running")
    running = panel.filter_by_status("running")
    assert len(running) == 3

    console = Console(record=True)
    panel.render(console)
    output = console.export_text()
    assert "Build" in output

    snapshot = panel.to_dict()
    assert isinstance(snapshot, dict)
    assert "integ-1" in snapshot


# ---------------------------------------------------------------------------
# StreamingRenderer lifecycle
# ---------------------------------------------------------------------------


def test_streaming_renderer_lifecycle() -> None:
    """Push 5 chunks, complete, verify concatenated text, render panel."""
    from src.ui.streaming_renderer import StreamingRenderer, StreamingState, TokenChunk

    renderer = StreamingRenderer()
    chunks = [
        TokenChunk(text="The "),
        TokenChunk(text="quick "),
        TokenChunk(text="brown "),
        TokenChunk(text="fox "),
        TokenChunk(text="jumps."),
    ]
    for chunk in chunks:
        renderer.push_chunk(chunk)

    assert renderer.token_count() == 5
    assert renderer.get_text() == "The quick brown fox jumps."
    assert renderer.word_count() == 5
    assert renderer._state == StreamingState.STREAMING

    renderer.complete()
    assert renderer._state == StreamingState.COMPLETE

    console = Console(record=True)
    renderer.render_panel(console, title="Integration Test")
    output = console.export_text()
    assert len(output) > 0


# ---------------------------------------------------------------------------
# SessionManager lifecycle
# ---------------------------------------------------------------------------


def test_debug_panel_update_and_render() -> None:
    """Update 2 metrics, render, verify to_dict contains updated values."""
    from src.ui.debug_panel import DebugPanel

    panel = DebugPanel()
    panel.update_metric("Model", "loss", 0.4321)
    panel.update_metric("Memory", "gpu_allocated_gb", 12.5)

    console = Console(record=True)
    panel.render(console)
    output = console.export_text()
    assert len(output) > 0

    snapshot = panel.to_dict()
    assert isinstance(snapshot, dict)
    model_metrics = {m["name"]: m["value"] for m in snapshot["Model"]["metrics"]}
    assert model_metrics["loss"] == pytest.approx(0.4321)
    memory_metrics = {m["name"]: m["value"] for m in snapshot["Memory"]["metrics"]}
    assert memory_metrics["gpu_allocated_gb"] == pytest.approx(12.5)


def test_progress_renderer_lifecycle() -> None:
    """Add task, advance 10 steps, render, remove."""
    from src.ui.progress_renderer import ProgressRenderer, ProgressTask

    renderer = ProgressRenderer()
    task = ProgressTask(task_id="integ-pr-1", description="Integration Train", total=100)
    renderer.add_task(task)

    for _ in range(10):
        renderer.advance("integ-pr-1")

    assert renderer._tasks["integ-pr-1"].completed == 10

    console = Console(record=True)
    renderer.render(console)
    output = console.export_text()
    assert "Integration Train" in output

    renderer.remove_task("integ-pr-1")
    assert "integ-pr-1" not in renderer._tasks


def test_session_manager_lifecycle() -> None:
    """Create 2 sessions, switch_to one, verify other paused, save/load round-trip."""
    from src.ui.session_manager import SessionManager, SessionState

    manager = SessionManager()
    s1 = manager.create("Tab Alpha")
    s2 = manager.create("Tab Beta")

    # Both start ACTIVE.
    assert manager.get(s1.session_id).state == SessionState.ACTIVE
    assert manager.get(s2.session_id).state == SessionState.ACTIVE

    # Switching to s2 should pause s1.
    manager.switch_to(s2.session_id)
    assert manager.get(s1.session_id).state == SessionState.PAUSED
    assert manager.get(s2.session_id).state == SessionState.ACTIVE

    # Snapshot and restore.
    snapshot = manager.save_to_dict()
    new_manager = SessionManager()
    new_manager.load_from_dict(snapshot)

    assert new_manager.get(s1.session_id).state == SessionState.PAUSED
    assert new_manager.get(s2.session_id).state == SessionState.ACTIVE
    assert new_manager.get(s1.session_id).name == "Tab Alpha"

    # Render should not crash.
    console = Console(record=True)
    new_manager.render(console)
    output = console.export_text()
    assert "Tab Alpha" in output or "Tab Beta" in output
