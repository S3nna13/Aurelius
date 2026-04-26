"""Tests for src/reasoning/mcts_reasoner.py — at least 20 tests."""

from __future__ import annotations

import math

import pytest

from src.reasoning.mcts_reasoner import _MAX_STATE_LEN, MCTS_REASONER, MCTSNode, MCTSReasoner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_reasoner(**kwargs) -> MCTSReasoner:
    return MCTSReasoner(**kwargs)


def build_chain(reasoner: MCTSReasoner, depth: int = 3) -> tuple[MCTSNode, list[MCTSNode]]:
    """Build a linear chain root -> child1 -> child2 of the given depth."""
    root = reasoner.create_root("root")
    nodes = [root]
    current = root
    for i in range(1, depth):
        children = reasoner.expand(current, [f"state_{i}"])
        current = children[0]
        nodes.append(current)
    return root, nodes


# ---------------------------------------------------------------------------
# 1. Root creation
# ---------------------------------------------------------------------------


def test_root_creation_basic():
    r = MCTSReasoner()
    root = r.create_root("hello world")
    assert root.state == "hello world"
    assert root.parent_id is None
    assert root.children == []
    assert root.visits == 0
    assert root.total_value == 0.0


# ---------------------------------------------------------------------------
# 2. State truncation at _MAX_STATE_LEN
# ---------------------------------------------------------------------------


def test_root_state_truncation():
    long_state = "x" * (_MAX_STATE_LEN + 100)
    r = MCTSReasoner()
    root = r.create_root(long_state)
    assert len(root.state) == _MAX_STATE_LEN


def test_child_state_truncation():
    r = MCTSReasoner()
    root = r.create_root("root")
    long_state = "y" * (_MAX_STATE_LEN + 50)
    children = r.expand(root, [long_state])
    assert len(children[0].state) == _MAX_STATE_LEN


# ---------------------------------------------------------------------------
# 3. Expand with priors
# ---------------------------------------------------------------------------


def test_expand_with_priors():
    r = MCTSReasoner()
    root = r.create_root("root")
    states = ["a", "b", "c"]
    priors = [0.5, 0.3, 0.2]
    children = r.expand(root, states, priors=priors)
    assert len(children) == 3
    for child, p in zip(children, priors):
        assert child.prior == p
        assert child.parent_id == root.id


# ---------------------------------------------------------------------------
# 4. Expand without priors defaults to 1.0
# ---------------------------------------------------------------------------


def test_expand_without_priors():
    r = MCTSReasoner()
    root = r.create_root("root")
    children = r.expand(root, ["a", "b"])
    for c in children:
        assert c.prior == 1.0


# ---------------------------------------------------------------------------
# 5. Prior out of range raises ValueError
# ---------------------------------------------------------------------------


def test_expand_prior_zero_raises():
    r = MCTSReasoner()
    root = r.create_root("root")
    with pytest.raises(ValueError, match="prior"):
        r.expand(root, ["a"], priors=[0.0])


def test_expand_prior_above_one_raises():
    r = MCTSReasoner()
    root = r.create_root("root")
    with pytest.raises(ValueError, match="prior"):
        r.expand(root, ["a"], priors=[1.1])


def test_expand_prior_negative_raises():
    r = MCTSReasoner()
    root = r.create_root("root")
    with pytest.raises(ValueError, match="prior"):
        r.expand(root, ["a"], priors=[-0.5])


# ---------------------------------------------------------------------------
# 6. Priors length mismatch raises ValueError
# ---------------------------------------------------------------------------


def test_expand_priors_length_mismatch():
    r = MCTSReasoner()
    root = r.create_root("root")
    with pytest.raises(ValueError, match="priors length"):
        r.expand(root, ["a", "b"], priors=[0.5])


# ---------------------------------------------------------------------------
# 7. UCB1 formula correctness
# ---------------------------------------------------------------------------


def test_ucb1_unvisited_node():
    """Unvisited node: value=0, exploration = c * sqrt(log(N)/1)."""
    node = MCTSNode(state="s", parent_id=None)
    c = 1.414
    parent_visits = 10
    expected = 0.0 + c * math.sqrt(math.log(parent_visits) / 1)
    assert abs(node.ucb1(c=c, parent_visits=parent_visits) - expected) < 1e-9


def test_ucb1_visited_node():
    node = MCTSNode(state="s", parent_id=None, visits=4, total_value=2.0)
    c = 1.414
    parent_visits = 16
    expected = (2.0 / 4) + c * math.sqrt(math.log(16) / 5)
    assert abs(node.ucb1(c=c, parent_visits=parent_visits) - expected) < 1e-9


def test_ucb1_parent_visits_zero_clamped():
    """parent_visits=0 should not raise; log(max(0,1))=0 → exploration=0."""
    node = MCTSNode(state="s", parent_id=None)
    result = node.ucb1(c=1.414, parent_visits=0)
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. backup_path propagates correctly up 3-level tree
# ---------------------------------------------------------------------------


def test_backup_path_three_levels():
    r = MCTSReasoner()
    root, nodes = build_chain(r, depth=3)
    r.backup_path(nodes, value=0.8)
    for n in nodes:
        assert n.visits == 1
        assert abs(n.total_value - 0.8) < 1e-9


def test_backup_path_multiple_calls_accumulate():
    r = MCTSReasoner()
    root, nodes = build_chain(r, depth=3)
    r.backup_path(nodes, value=0.5)
    r.backup_path(nodes, value=0.5)
    for n in nodes:
        assert n.visits == 2
        assert abs(n.total_value - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 9. backup with out-of-range value raises ValueError
# ---------------------------------------------------------------------------


def test_backup_out_of_range_high():
    r = MCTSReasoner()
    node = r.create_root("root")
    with pytest.raises(ValueError, match="out of range"):
        r.backup(node, value=1.5)


def test_backup_out_of_range_low():
    r = MCTSReasoner()
    node = r.create_root("root")
    with pytest.raises(ValueError, match="out of range"):
        r.backup(node, value=-2.0)


def test_backup_path_out_of_range_raises():
    r = MCTSReasoner()
    root, nodes = build_chain(r, depth=2)
    with pytest.raises(ValueError, match="out of range"):
        r.backup_path(nodes, value=99.0)


# ---------------------------------------------------------------------------
# 10. select_child returns max UCB1 child
# ---------------------------------------------------------------------------


def test_select_child_returns_max_ucb1():
    r = MCTSReasoner(c_puct=1.414)
    root = r.create_root("root")
    # Create two children with known visit counts; the one with fewer visits
    # should have higher UCB1 (exploration bonus dominates).
    children = r.expand(root, ["a", "b"])
    # Give root 10 visits, child[0] 5 visits, child[1] 1 visit
    root.visits = 10
    children[0].visits = 5
    children[0].total_value = 4.0
    children[1].visits = 1
    children[1].total_value = 0.5
    selected = r.select_child(root)
    ucbs = [c.ucb1(r.c_puct, root.visits) for c in children]
    assert selected is children[ucbs.index(max(ucbs))]


# ---------------------------------------------------------------------------
# 11. select_child on leaf (no children) raises ValueError
# ---------------------------------------------------------------------------


def test_select_child_leaf_raises():
    r = MCTSReasoner()
    root = r.create_root("root")
    with pytest.raises(ValueError, match="no children"):
        r.select_child(root)


# ---------------------------------------------------------------------------
# 12. best_child returns max-visits child
# ---------------------------------------------------------------------------


def test_best_child_returns_max_visits():
    r = MCTSReasoner()
    root = r.create_root("root")
    children = r.expand(root, ["a", "b", "c"])
    children[0].visits = 3
    children[1].visits = 10
    children[2].visits = 1
    assert r.best_child(root) is children[1]


def test_best_child_no_children_raises():
    r = MCTSReasoner()
    root = r.create_root("root")
    with pytest.raises(ValueError, match="no children"):
        r.best_child(root)


# ---------------------------------------------------------------------------
# 13. best_path follows best children to leaf
# ---------------------------------------------------------------------------


def test_best_path_follows_best_children():
    r = MCTSReasoner()
    root = r.create_root("root")
    c1, c2 = r.expand(root, ["c1", "c2"])
    c1.visits = 5
    c2.visits = 10
    gc1, gc2 = r.expand(c2, ["gc1", "gc2"])
    gc1.visits = 3
    gc2.visits = 7
    path = r.best_path(root)
    assert [n.state for n in path] == ["root", "c2", "gc2"]


# ---------------------------------------------------------------------------
# 14. rollout_path budget enforcement
# ---------------------------------------------------------------------------


def test_rollout_path_respects_budget():
    r = MCTSReasoner(max_depth=100)
    root = r.create_root("root")
    current = root
    # Build a deep chain of 20 nodes
    for i in range(20):
        children = r.expand(current, [f"s{i}"])
        children[0].visits = 1
        current = children[0]
    path = r.rollout_path(root, budget=5)
    # path includes root + up to 5 steps
    assert len(path) <= 6  # root + 5 steps


def test_rollout_path_budget_zero_raises():
    r = MCTSReasoner()
    root = r.create_root("root")
    with pytest.raises(ValueError, match="budget must be > 0"):
        r.rollout_path(root, budget=0)


def test_rollout_path_budget_negative_raises():
    r = MCTSReasoner()
    root = r.create_root("root")
    with pytest.raises(ValueError, match="budget must be > 0"):
        r.rollout_path(root, budget=-1)


# ---------------------------------------------------------------------------
# 15. Determinism: same initial state → same UCB1 scores
# ---------------------------------------------------------------------------


def test_ucb1_determinism():
    node_a = MCTSNode(state="same", parent_id=None, visits=3, total_value=1.5)
    node_b = MCTSNode(state="same", parent_id=None, visits=3, total_value=1.5)
    assert node_a.ucb1(c=1.414, parent_visits=10) == node_b.ucb1(c=1.414, parent_visits=10)


# ---------------------------------------------------------------------------
# 16. Edge: single node (no children), max_depth=1
# ---------------------------------------------------------------------------


def test_rollout_path_single_node_no_children():
    r = MCTSReasoner(max_depth=1)
    root = r.create_root("root")
    path = r.rollout_path(root, budget=10)
    assert path == [root]


# ---------------------------------------------------------------------------
# 17. MCTS_REASONER singleton is usable
# ---------------------------------------------------------------------------


def test_mcts_reasoner_singleton():
    root = MCTS_REASONER.create_root("test")
    assert root.state == "test"


# ---------------------------------------------------------------------------
# 18. node.value property
# ---------------------------------------------------------------------------


def test_node_value_zero_visits():
    node = MCTSNode(state="s", parent_id=None)
    assert node.value == 0.0


def test_node_value_with_visits():
    node = MCTSNode(state="s", parent_id=None, visits=4, total_value=2.0)
    assert node.value == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 19. rollout_path stops at max_depth
# ---------------------------------------------------------------------------


def test_rollout_path_max_depth():
    r = MCTSReasoner(max_depth=3)
    root = r.create_root("root")
    current = root
    for i in range(10):
        children = r.expand(current, [f"s{i}"])
        children[0].visits = 1
        current = children[0]
    path = r.rollout_path(root, budget=100)
    assert len(path) <= r.max_depth


# ---------------------------------------------------------------------------
# 20. expand adds children to node.children list
# ---------------------------------------------------------------------------


def test_expand_mutates_node_children():
    r = MCTSReasoner()
    root = r.create_root("root")
    assert root.children == []
    r.expand(root, ["x", "y", "z"])
    assert len(root.children) == 3
    assert all(c.parent_id == root.id for c in root.children)
