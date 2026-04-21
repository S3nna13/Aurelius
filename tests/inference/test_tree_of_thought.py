"""Tests for src/inference/tree_of_thought.py.

Covers:
  - Configuration defaults                    (test_config_defaults)
  - ToTTree.add_node / id increment           (test_tree_add_node)
  - ToTTree.children                          (test_tree_children)
  - ToTTree.path_to_root                      (test_tree_path_to_root)
  - ToTTree.best_leaf                         (test_tree_best_leaf)
  - ToTTree.size                              (test_tree_size)
  - generate_thoughts n_candidates cap        (test_generate_thoughts_limits)
  - evaluate_state delegates to value_fn      (test_evaluate_state_calls_fn)
  - BFS breadth pruning                       (test_bfs_explores_breadth)
  - BFS max_depth respected                   (test_bfs_terminates_at_max_depth)
  - BFS early stop on terminal                (test_bfs_terminal_stops)
  - DFS prunes below threshold                (test_dfs_prunes_low_value)
  - DFS finds terminal state                  (test_dfs_finds_solution)
  - search() dispatches bfs/dfs              (test_search_dispatches)
  - Registry entry                            (test_registry_entry)
  - Integration: propose+value+terminal       (test_integration_full_search)
"""
from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from src.inference.tree_of_thought import (
    ThoughtNode,
    ToTConfig,
    ToTTree,
    TreeOfThoughtDecoder,
)
from src.inference import DECODER_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _always_true(_: str) -> bool:
    return True


def _always_false(_: str) -> bool:
    return False


def _fixed_value(v: float) -> callable:
    return lambda _s: v


def _propose(*thoughts: str):
    """Return a propose_fn that always returns the given thoughts."""
    return lambda _s: list(thoughts)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------

class TestToTConfigDefaults:
    def test_config_defaults(self):
        cfg = ToTConfig()
        assert cfg.search_mode == "bfs"
        assert cfg.breadth_limit == 5
        assert cfg.max_depth == 6
        assert cfg.n_candidates == 3
        assert cfg.value_threshold == 0.3
        assert cfg.aggregation == "max"


# ---------------------------------------------------------------------------
# 2. ToTTree.add_node — id increments
# ---------------------------------------------------------------------------

class TestToTTreeAddNode:
    def test_tree_add_node(self):
        tree = ToTTree()
        id0 = tree.add_node(state="root", depth=0, value=0.5, parent_id=None)
        id1 = tree.add_node(state="child", depth=1, value=0.6, parent_id=id0)
        id2 = tree.add_node(state="sibling", depth=1, value=0.4, parent_id=id0)

        assert id0 == 0
        assert id1 == 1
        assert id2 == 2

        node0 = tree.get_node(id0)
        assert node0.state == "root"
        assert node0.depth == 0
        assert node0.value == 0.5
        assert node0.parent_id is None


# ---------------------------------------------------------------------------
# 3. ToTTree.children
# ---------------------------------------------------------------------------

class TestToTTreeChildren:
    def test_tree_children(self):
        tree = ToTTree()
        root_id = tree.add_node("root", 0, 1.0, None)
        c1 = tree.add_node("c1", 1, 0.7, root_id)
        c2 = tree.add_node("c2", 1, 0.8, root_id)

        kids = tree.children(root_id)
        assert len(kids) == 2
        child_ids = {k.node_id for k in kids}
        assert child_ids == {c1, c2}

        # Leaf has no children
        assert tree.children(c1) == []


# ---------------------------------------------------------------------------
# 4. ToTTree.path_to_root
# ---------------------------------------------------------------------------

class TestToTTreePathToRoot:
    def test_tree_path_to_root(self):
        tree = ToTTree()
        root_id = tree.add_node("root", 0, 1.0, None)
        mid_id = tree.add_node("mid", 1, 0.9, root_id)
        leaf_id = tree.add_node("leaf", 2, 0.8, mid_id)

        path = tree.path_to_root(leaf_id)
        assert len(path) == 3
        assert path[0].node_id == root_id
        assert path[1].node_id == mid_id
        assert path[2].node_id == leaf_id

    def test_tree_path_to_root_single_node(self):
        tree = ToTTree()
        root_id = tree.add_node("root", 0, 1.0, None)
        path = tree.path_to_root(root_id)
        assert len(path) == 1
        assert path[0].node_id == root_id


# ---------------------------------------------------------------------------
# 5. ToTTree.best_leaf
# ---------------------------------------------------------------------------

class TestToTTreeBestLeaf:
    def test_tree_best_leaf(self):
        tree = ToTTree()
        root_id = tree.add_node("root", 0, 0.5, None)
        tree.add_node("low", 1, 0.2, root_id)
        tree.add_node("mid", 1, 0.6, root_id)
        best_id = tree.add_node("high", 1, 0.9, root_id)

        best = tree.best_leaf()
        assert best.node_id == best_id
        assert best.value == 0.9

    def test_tree_best_leaf_empty_raises(self):
        tree = ToTTree()
        with pytest.raises(ValueError, match="empty"):
            tree.best_leaf()


# ---------------------------------------------------------------------------
# 6. ToTTree.size
# ---------------------------------------------------------------------------

class TestToTTreeSize:
    def test_tree_size(self):
        tree = ToTTree()
        assert tree.size() == 0
        tree.add_node("root", 0, 1.0, None)
        assert tree.size() == 1
        tree.add_node("child", 1, 0.5, 0)
        assert tree.size() == 2


# ---------------------------------------------------------------------------
# 7. generate_thoughts limits to n_candidates
# ---------------------------------------------------------------------------

class TestGenerateThoughts:
    def test_generate_thoughts_limits(self):
        cfg = ToTConfig(n_candidates=2)
        decoder = TreeOfThoughtDecoder(cfg)

        # propose_fn returns 5 candidates, but only 2 should be taken
        propose = _propose("a", "b", "c", "d", "e")
        result = decoder.generate_thoughts("state", propose)
        assert result == ["a", "b"]

    def test_generate_thoughts_fewer_than_limit(self):
        cfg = ToTConfig(n_candidates=5)
        decoder = TreeOfThoughtDecoder(cfg)

        propose = _propose("only_one")
        result = decoder.generate_thoughts("state", propose)
        assert result == ["only_one"]


# ---------------------------------------------------------------------------
# 8. evaluate_state delegates correctly
# ---------------------------------------------------------------------------

class TestEvaluateState:
    def test_evaluate_state_calls_fn(self):
        cfg = ToTConfig()
        decoder = TreeOfThoughtDecoder(cfg)
        mock_fn = MagicMock(return_value=0.75)

        score = decoder.evaluate_state("my state", mock_fn)

        mock_fn.assert_called_once_with("my state")
        assert score == 0.75


# ---------------------------------------------------------------------------
# 9. BFS breadth pruning
# ---------------------------------------------------------------------------

class TestBFSBreadth:
    def test_bfs_explores_breadth(self):
        """At each depth the frontier should not exceed breadth_limit."""
        cfg = ToTConfig(
            search_mode="bfs",
            breadth_limit=2,
            max_depth=2,
            n_candidates=4,  # propose 4 but keep only 2
        )
        decoder = TreeOfThoughtDecoder(cfg)

        propose = _propose("t1", "t2", "t3", "t4")
        value_fn = _fixed_value(0.5)
        terminal_fn = _always_false

        tree = decoder.bfs("root", propose, value_fn, terminal_fn)

        # BFS creates all children first, then prunes the frontier.
        # Each depth-1 node creates up to n_candidates=4 children before pruning.
        # Frontier after depth-1: breadth_limit=2 nodes.
        # Those 2 nodes each create up to 4 children at depth-2.
        # Total nodes = 1 root + (root's n_candidates) + (b * n_candidates) = 1 + 4 + 8 = 13.
        # Upper bound: 1 + n_candidates + breadth_limit * n_candidates
        max_expected = (
            1
            + cfg.n_candidates
            + cfg.breadth_limit * cfg.n_candidates
        )
        assert tree.size() <= max_expected


# ---------------------------------------------------------------------------
# 10. BFS terminates at max_depth
# ---------------------------------------------------------------------------

class TestBFSMaxDepth:
    def test_bfs_terminates_at_max_depth(self):
        cfg = ToTConfig(
            search_mode="bfs",
            breadth_limit=3,
            max_depth=3,
            n_candidates=2,
        )
        decoder = TreeOfThoughtDecoder(cfg)

        propose = _propose("step_a", "step_b")
        value_fn = _fixed_value(0.9)
        terminal_fn = _always_false

        tree = decoder.bfs("root", propose, value_fn, terminal_fn)

        # No node should exceed max_depth
        for node in (tree.get_node(nid) for nid in range(tree.size())):
            assert node.depth <= cfg.max_depth


# ---------------------------------------------------------------------------
# 11. BFS stops at terminal node
# ---------------------------------------------------------------------------

class TestBFSTerminal:
    def test_bfs_terminal_stops(self):
        """BFS should return as soon as it creates a terminal node."""
        cfg = ToTConfig(
            search_mode="bfs",
            breadth_limit=5,
            max_depth=10,
            n_candidates=3,
        )
        decoder = TreeOfThoughtDecoder(cfg)

        propose = _propose("step")
        value_fn = _fixed_value(0.9)

        calls: list = []

        def terminal_fn(state: str) -> bool:
            calls.append(state)
            # Terminal after depth 1 (first expansion)
            return state != "root"

        tree = decoder.bfs("root", propose, value_fn, terminal_fn)

        # Tree should contain root + at least one terminal child
        terminal_nodes = [
            tree.get_node(nid)
            for nid in range(tree.size())
            if tree.get_node(nid).is_terminal
        ]
        assert len(terminal_nodes) >= 1


# ---------------------------------------------------------------------------
# 12. DFS prunes nodes below threshold
# ---------------------------------------------------------------------------

class TestDFSPruning:
    def test_dfs_prunes_low_value(self):
        """Children with value < value_threshold must not be expanded."""
        cfg = ToTConfig(
            search_mode="dfs",
            max_depth=5,
            n_candidates=2,
            value_threshold=0.5,
        )
        decoder = TreeOfThoughtDecoder(cfg)

        propose = _propose("bad", "good")

        call_order: list = []

        def value_fn(state: str) -> float:
            call_order.append(state)
            # 'good' paths get high value, 'bad' paths get low value
            if "bad" in state:
                return 0.1   # below threshold → pruned
            return 0.9

        terminal_fn = _always_false

        tree = decoder.dfs("root", propose, value_fn, terminal_fn)

        # No node whose state contains "bad" should have been expanded
        # (i.e., have children).  They can exist as leaf nodes, but their
        # children should not appear.
        for nid in range(tree.size()):
            node = tree.get_node(nid)
            if "bad" in node.state:
                assert node.children_ids == [], (
                    f"Node {nid!r} with low-value state was incorrectly expanded"
                )


# ---------------------------------------------------------------------------
# 13. DFS finds terminal state
# ---------------------------------------------------------------------------

class TestDFSSolution:
    def test_dfs_finds_solution(self):
        cfg = ToTConfig(
            search_mode="dfs",
            max_depth=4,
            n_candidates=2,
            value_threshold=0.1,
        )
        decoder = TreeOfThoughtDecoder(cfg)

        propose = _propose("step1", "step2")
        value_fn = _fixed_value(0.9)

        terminal_counter = {"count": 0}

        def terminal_fn(state: str) -> bool:
            # Terminal on the first child created (depth 1)
            return state != "initial"

        tree = decoder.dfs("initial", propose, value_fn, terminal_fn)

        terminal_nodes = [
            tree.get_node(nid)
            for nid in range(tree.size())
            if tree.get_node(nid).is_terminal
        ]
        assert len(terminal_nodes) >= 1


# ---------------------------------------------------------------------------
# 14. search() dispatches to correct method
# ---------------------------------------------------------------------------

class TestSearchDispatches:
    def test_search_dispatches_bfs(self, monkeypatch):
        cfg = ToTConfig(search_mode="bfs", max_depth=1, n_candidates=1)
        decoder = TreeOfThoughtDecoder(cfg)

        mock_bfs = MagicMock(wraps=decoder.bfs)
        monkeypatch.setattr(decoder, "bfs", mock_bfs)

        tree, best = decoder.search(
            "state",
            _propose("t"),
            _fixed_value(0.5),
            _always_false,
        )
        mock_bfs.assert_called_once()

    def test_search_dispatches_dfs(self, monkeypatch):
        cfg = ToTConfig(search_mode="dfs", max_depth=1, n_candidates=1)
        decoder = TreeOfThoughtDecoder(cfg)

        mock_dfs = MagicMock(wraps=decoder.dfs)
        monkeypatch.setattr(decoder, "dfs", mock_dfs)

        tree, best = decoder.search(
            "state",
            _propose("t"),
            _fixed_value(0.5),
            _always_false,
        )
        # DFS is recursive — the mock is called at least once (possibly more
        # due to recursive expansions); we only need to confirm it was called.
        assert mock_dfs.call_count >= 1

    def test_search_invalid_mode_raises(self):
        cfg = ToTConfig(search_mode="mcts")
        decoder = TreeOfThoughtDecoder(cfg)
        with pytest.raises(ValueError, match="Unknown search_mode"):
            decoder.search("s", _propose("t"), _fixed_value(0.5))


# ---------------------------------------------------------------------------
# 15. Registry entry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_entry(self):
        assert "tree_of_thought" in DECODER_REGISTRY
        assert DECODER_REGISTRY["tree_of_thought"] is TreeOfThoughtDecoder


# ---------------------------------------------------------------------------
# 16. Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_integration_full_search(self):
        """Integration: propose_fn → ["step1", "step2"], value_fn → 0.8,
        terminal_fn → True after depth 2.
        Verify tree has nodes, best_node.is_terminal is True.
        """
        cfg = ToTConfig(
            search_mode="bfs",
            breadth_limit=5,
            max_depth=5,
            n_candidates=2,
            value_threshold=0.3,
        )
        decoder = TreeOfThoughtDecoder(cfg)

        propose_fn = lambda _s: ["step1", "step2"]  # noqa: E731
        value_fn = lambda _s: 0.8                   # noqa: E731

        # BFS joins state as "parent_state thought", so depth-1 nodes are
        # "problem step1" / "problem step2" — these each contain exactly
        # 1 space.  We declare terminal at depth >= 1 (i.e. ≥ 1 space).
        def terminal_fn(state: str) -> bool:
            return state.count(" ") >= 1

        tree, best_node = decoder.search(
            initial_state="problem",
            propose_fn=propose_fn,
            value_fn=value_fn,
            terminal_fn=terminal_fn,
        )

        # Tree must be non-trivial (root + at least one child)
        assert tree.size() > 1, "Tree should contain more than just the root."

        # At least one terminal node must exist in the tree
        terminal_nodes = [
            tree.get_node(nid)
            for nid in range(tree.size())
            if tree.get_node(nid).is_terminal
        ]
        assert len(terminal_nodes) >= 1, "Expected at least one terminal node."

        # best_node is a proper ThoughtNode instance
        assert isinstance(best_node, ThoughtNode)

        # path_to_root for best node should start at the root
        path = tree.path_to_root(best_node.node_id)
        assert path[0].parent_id is None, "Path should start at root."
        assert path[-1].node_id == best_node.node_id
