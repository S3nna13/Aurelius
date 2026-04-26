"""Tests for src/agent/goal_hierarchy.py."""

from __future__ import annotations

import pytest

from src.agent.goal_hierarchy import (
    GOAL_HIERARCHY_REGISTRY,
    Goal,
    GoalHierarchy,
    GoalStatus,
)


@pytest.fixture
def hierarchy() -> GoalHierarchy:
    return GoalHierarchy()


def test_registry_default_is_class():
    assert GOAL_HIERARCHY_REGISTRY["default"] is GoalHierarchy


def test_goal_status_enum_values():
    assert GoalStatus.PENDING.value == "pending"
    assert GoalStatus.ACTIVE.value == "active"
    assert GoalStatus.COMPLETED.value == "completed"
    assert GoalStatus.BLOCKED.value == "blocked"
    assert GoalStatus.CANCELLED.value == "cancelled"


def test_goal_auto_id_is_eight_chars():
    g = Goal(description="x")
    assert len(g.goal_id) == 8


def test_goal_default_status_pending():
    g = Goal(description="x")
    assert g.status == GoalStatus.PENDING


def test_add_root_goal(hierarchy):
    g = hierarchy.add_goal("root 1")
    assert g.parent_id is None
    assert hierarchy.get_goal(g.goal_id) is g


def test_add_child_goal(hierarchy):
    root = hierarchy.add_goal("root")
    child = hierarchy.add_goal("child", parent_id=root.goal_id)
    assert child.parent_id == root.goal_id
    assert child.goal_id in hierarchy.get_goal(root.goal_id).children


def test_add_goal_unknown_parent_raises(hierarchy):
    with pytest.raises(KeyError):
        hierarchy.add_goal("orphan", parent_id="doesnotexist")


def test_add_goal_with_priority(hierarchy):
    g = hierarchy.add_goal("p1", priority=7)
    assert g.priority == 7


def test_get_goal_returns_none_for_unknown(hierarchy):
    assert hierarchy.get_goal("unknown") is None


def test_set_status_changes_status(hierarchy):
    g = hierarchy.add_goal("g")
    hierarchy.set_status(g.goal_id, GoalStatus.ACTIVE)
    assert hierarchy.get_goal(g.goal_id).status == GoalStatus.ACTIVE


def test_set_status_unknown_raises(hierarchy):
    with pytest.raises(KeyError):
        hierarchy.set_status("nope", GoalStatus.ACTIVE)


def test_children_of_returns_empty_for_leaf(hierarchy):
    g = hierarchy.add_goal("g")
    assert hierarchy.children_of(g.goal_id) == []


def test_children_of_unknown_returns_empty(hierarchy):
    assert hierarchy.children_of("x") == []


def test_children_of_returns_children(hierarchy):
    root = hierarchy.add_goal("r")
    c1 = hierarchy.add_goal("c1", parent_id=root.goal_id)
    c2 = hierarchy.add_goal("c2", parent_id=root.goal_id)
    ids = {g.goal_id for g in hierarchy.children_of(root.goal_id)}
    assert ids == {c1.goal_id, c2.goal_id}


def test_root_goals_empty(hierarchy):
    assert hierarchy.root_goals() == []


def test_root_goals_multiple(hierarchy):
    a = hierarchy.add_goal("a")
    b = hierarchy.add_goal("b")
    hierarchy.add_goal("child", parent_id=a.goal_id)
    roots = {g.goal_id for g in hierarchy.root_goals()}
    assert roots == {a.goal_id, b.goal_id}


def test_leaf_goals_all_when_no_children(hierarchy):
    a = hierarchy.add_goal("a")
    b = hierarchy.add_goal("b")
    leaves = {g.goal_id for g in hierarchy.leaf_goals()}
    assert leaves == {a.goal_id, b.goal_id}


def test_leaf_goals_excludes_parents(hierarchy):
    root = hierarchy.add_goal("r")
    child = hierarchy.add_goal("c", parent_id=root.goal_id)
    leaves = {g.goal_id for g in hierarchy.leaf_goals()}
    assert leaves == {child.goal_id}


def test_active_path_empty_when_none_active(hierarchy):
    hierarchy.add_goal("a")
    assert hierarchy.active_path() == []


def test_active_path_leaf_first(hierarchy):
    root = hierarchy.add_goal("root")
    mid = hierarchy.add_goal("mid", parent_id=root.goal_id)
    leaf = hierarchy.add_goal("leaf", parent_id=mid.goal_id)
    for gid in (root.goal_id, mid.goal_id, leaf.goal_id):
        hierarchy.set_status(gid, GoalStatus.ACTIVE)
    path = hierarchy.active_path()
    assert [g.goal_id for g in path] == [leaf.goal_id, mid.goal_id, root.goal_id]


def test_active_path_only_active(hierarchy):
    root = hierarchy.add_goal("root")
    leaf = hierarchy.add_goal("leaf", parent_id=root.goal_id)
    hierarchy.set_status(leaf.goal_id, GoalStatus.ACTIVE)
    path = hierarchy.active_path()
    assert len(path) == 1
    assert path[0].goal_id == leaf.goal_id


def test_completion_ratio_zero_when_empty(hierarchy):
    assert hierarchy.completion_ratio() == 0.0


def test_completion_ratio_zero_when_none_completed(hierarchy):
    hierarchy.add_goal("a")
    hierarchy.add_goal("b")
    assert hierarchy.completion_ratio() == 0.0


def test_completion_ratio_partial(hierarchy):
    a = hierarchy.add_goal("a")
    hierarchy.add_goal("b")
    hierarchy.set_status(a.goal_id, GoalStatus.COMPLETED)
    assert hierarchy.completion_ratio() == 0.5


def test_completion_ratio_full(hierarchy):
    a = hierarchy.add_goal("a")
    b = hierarchy.add_goal("b")
    hierarchy.set_status(a.goal_id, GoalStatus.COMPLETED)
    hierarchy.set_status(b.goal_id, GoalStatus.COMPLETED)
    assert hierarchy.completion_ratio() == 1.0


def test_to_dict_empty_hierarchy(hierarchy):
    d = hierarchy.to_dict()
    assert d["roots"] == []
    assert d["count"] == 0
    assert d["completion_ratio"] == 0.0


def test_to_dict_structure(hierarchy):
    root = hierarchy.add_goal("root", priority=1)
    hierarchy.add_goal("child", parent_id=root.goal_id, priority=2)
    d = hierarchy.to_dict()
    assert d["count"] == 2
    assert len(d["roots"]) == 1
    root_d = d["roots"][0]
    assert root_d["description"] == "root"
    assert root_d["priority"] == 1
    assert len(root_d["children"]) == 1
    assert root_d["children"][0]["description"] == "child"
    assert root_d["children"][0]["parent_id"] == root.goal_id


def test_to_dict_serializes_status_as_value(hierarchy):
    g = hierarchy.add_goal("g")
    hierarchy.set_status(g.goal_id, GoalStatus.ACTIVE)
    d = hierarchy.to_dict()
    assert d["roots"][0]["status"] == "active"


def test_to_dict_includes_completion_ratio(hierarchy):
    a = hierarchy.add_goal("a")
    hierarchy.add_goal("b")
    hierarchy.set_status(a.goal_id, GoalStatus.COMPLETED)
    d = hierarchy.to_dict()
    assert d["completion_ratio"] == 0.5


def test_metadata_defaults_empty(hierarchy):
    g = hierarchy.add_goal("x")
    assert g.metadata == {}


def test_metadata_mutation_preserved(hierarchy):
    g = hierarchy.add_goal("x")
    g.metadata["tag"] = "v"
    assert hierarchy.get_goal(g.goal_id).metadata["tag"] == "v"


def test_unique_goal_ids(hierarchy):
    ids = {hierarchy.add_goal(f"g{i}").goal_id for i in range(20)}
    assert len(ids) == 20


def test_deep_nesting(hierarchy):
    root = hierarchy.add_goal("r")
    a = hierarchy.add_goal("a", parent_id=root.goal_id)
    b = hierarchy.add_goal("b", parent_id=a.goal_id)
    c = hierarchy.add_goal("c", parent_id=b.goal_id)
    leaves = {g.goal_id for g in hierarchy.leaf_goals()}
    assert leaves == {c.goal_id}
    roots = {g.goal_id for g in hierarchy.root_goals()}
    assert roots == {root.goal_id}
