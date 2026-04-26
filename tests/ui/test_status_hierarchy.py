"""Tests for src.ui.status_hierarchy."""

from __future__ import annotations

import pytest
from rich.console import Console

from src.ui.status_hierarchy import (
    STATUS_TREE_REGISTRY,
    StatusLevel,
    StatusNode,
    StatusState,
    StatusTree,
)


def _make_node(node_id: str, label: str = "test") -> StatusNode:
    return StatusNode(
        id=node_id,
        level=StatusLevel.TASK,
        state=StatusState.IDLE,
        label=label,
    )


# ---------------------------------------------------------------------------
# add_node / get_node
# ---------------------------------------------------------------------------


def test_add_and_get_root_node() -> None:
    tree = StatusTree()
    node = _make_node("n1", "Root task")
    tree.add_node(node)
    retrieved = tree.get_node("n1")
    assert retrieved.id == "n1"
    assert retrieved.label == "Root task"


def test_add_child_node() -> None:
    tree = StatusTree()
    root = _make_node("root")
    child = _make_node("child")
    tree.add_node(root)
    tree.add_node(child, parent_id="root")
    assert child in tree.get_node("root").children


def test_add_duplicate_id_raises() -> None:
    tree = StatusTree()
    node = _make_node("dup")
    tree.add_node(node)
    with pytest.raises(ValueError):
        tree.add_node(_make_node("dup"))


def test_add_child_to_missing_parent_raises() -> None:
    tree = StatusTree()
    node = _make_node("orphan")
    with pytest.raises(KeyError):
        tree.add_node(node, parent_id="nonexistent")


# ---------------------------------------------------------------------------
# update_state
# ---------------------------------------------------------------------------


def test_update_state_changes_state() -> None:
    tree = StatusTree()
    tree.add_node(_make_node("t1"))
    tree.update_state("t1", StatusState.RUNNING)
    assert tree.get_node("t1").state == StatusState.RUNNING


def test_update_state_with_progress() -> None:
    tree = StatusTree()
    tree.add_node(_make_node("t2"))
    tree.update_state("t2", StatusState.RUNNING, progress=0.5)
    node = tree.get_node("t2")
    assert node.state == StatusState.RUNNING
    assert node.progress == pytest.approx(0.5)


def test_update_state_unknown_id_raises() -> None:
    tree = StatusTree()
    with pytest.raises(KeyError):
        tree.update_state("ghost", StatusState.FAILED)


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------


def test_get_node_unknown_raises_key_error() -> None:
    tree = StatusTree()
    with pytest.raises(KeyError):
        tree.get_node("does_not_exist")


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_returns_dict() -> None:
    tree = StatusTree()
    tree.add_node(_make_node("a"))
    result = tree.to_dict()
    assert isinstance(result, dict)
    assert "roots" in result


def test_to_dict_contains_node_data() -> None:
    tree = StatusTree()
    tree.add_node(
        StatusNode(
            id="sess1",
            level=StatusLevel.SESSION,
            state=StatusState.RUNNING,
            label="My Session",
        )
    )
    d = tree.to_dict()
    root_ids = [r["id"] for r in d["roots"]]
    assert "sess1" in root_ids


def test_to_dict_nested_children() -> None:
    tree = StatusTree()
    root = _make_node("root")
    child = _make_node("child")
    tree.add_node(root)
    tree.add_node(child, parent_id="root")
    d = tree.to_dict()
    assert len(d["roots"][0]["children"]) == 1
    assert d["roots"][0]["children"][0]["id"] == "child"


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_empty_tree_does_not_crash() -> None:
    tree = StatusTree()
    console = Console(record=True)
    tree.render(console)


def test_render_with_nodes_does_not_crash() -> None:
    tree = StatusTree()
    tree.add_node(_make_node("r1", "Root"))
    tree.add_node(_make_node("c1", "Child"), parent_id="r1")
    console = Console(record=True)
    tree.render(console)


def test_render_output_contains_label() -> None:
    tree = StatusTree()
    tree.add_node(_make_node("visible", "Visible Node"))
    console = Console(record=True)
    tree.render(console)
    output = console.export_text()
    assert "Visible Node" in output


# ---------------------------------------------------------------------------
# STATUS_TREE_REGISTRY
# ---------------------------------------------------------------------------


def test_status_tree_registry_is_dict() -> None:
    assert isinstance(STATUS_TREE_REGISTRY, dict)
