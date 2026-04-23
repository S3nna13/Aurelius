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
