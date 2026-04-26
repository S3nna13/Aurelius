"""Tests for src/reasoning/tot_planner.py"""

from __future__ import annotations

from src.reasoning.tot_planner import TOT_PLANNER, ThoughtNode, ToTPlanner

# ---------- ThoughtNode ----------


class TestThoughtNode:
    def test_auto_id_generated(self):
        node = ThoughtNode(parent_id=None, thought="hello")
        assert node.id is not None

    def test_id_is_8_chars(self):
        node = ThoughtNode(parent_id=None, thought="hello")
        assert len(node.id) == 8

    def test_id_is_hex(self):
        node = ThoughtNode(parent_id=None, thought="hello")
        int(node.id, 16)  # Should not raise

    def test_unique_ids(self):
        n1 = ThoughtNode(parent_id=None, thought="a")
        n2 = ThoughtNode(parent_id=None, thought="b")
        assert n1.id != n2.id

    def test_default_parent_id_none(self):
        node = ThoughtNode(parent_id=None, thought="x")
        assert node.parent_id is None

    def test_default_score_zero(self):
        node = ThoughtNode(parent_id=None, thought="x")
        assert node.score == 0.0

    def test_default_depth_zero(self):
        node = ThoughtNode(parent_id=None, thought="x")
        assert node.depth == 0

    def test_default_children_empty(self):
        node = ThoughtNode(parent_id=None, thought="x")
        assert node.children == []

    def test_custom_score(self):
        node = ThoughtNode(parent_id=None, thought="x", score=0.9)
        assert node.score == 0.9

    def test_custom_depth(self):
        node = ThoughtNode(parent_id=None, thought="x", depth=3)
        assert node.depth == 3

    def test_thought_stored(self):
        node = ThoughtNode(parent_id="abc", thought="my thought")
        assert node.thought == "my thought"
        assert node.parent_id == "abc"


# ---------- ToTPlanner.add_root ----------


class TestAddRoot:
    def setup_method(self):
        self.planner = ToTPlanner()

    def test_returns_thought_node(self):
        root = self.planner.add_root("start")
        assert isinstance(root, ThoughtNode)

    def test_depth_zero(self):
        root = self.planner.add_root("start")
        assert root.depth == 0

    def test_parent_id_none(self):
        root = self.planner.add_root("start")
        assert root.parent_id is None

    def test_thought_stored(self):
        root = self.planner.add_root("my root thought")
        assert root.thought == "my root thought"

    def test_score_default_zero(self):
        root = self.planner.add_root("start")
        assert root.score == 0.0

    def test_custom_score(self):
        root = self.planner.add_root("start", score=0.5)
        assert root.score == 0.5


# ---------- ToTPlanner.expand ----------


class TestExpand:
    def setup_method(self):
        self.planner = ToTPlanner()
        self.root = self.planner.add_root("root")

    def test_creates_child_nodes(self):
        children = self.planner.expand(self.root, ["a", "b"], [0.9, 0.5])
        assert len(children) == 2

    def test_parent_id_set(self):
        children = self.planner.expand(self.root, ["a"], [1.0])
        assert children[0].parent_id == self.root.id

    def test_depth_incremented(self):
        children = self.planner.expand(self.root, ["a"], [1.0])
        assert children[0].depth == 1

    def test_nested_depth(self):
        children = self.planner.expand(self.root, ["a"], [1.0])
        grandchildren = self.planner.expand(children[0], ["b"], [0.5])
        assert grandchildren[0].depth == 2

    def test_children_appended_to_parent(self):
        self.planner.expand(self.root, ["a", "b"], [0.8, 0.6])
        assert len(self.root.children) == 2

    def test_scores_assigned(self):
        children = self.planner.expand(self.root, ["a", "b"], [0.8, 0.6])
        assert children[0].score == 0.8
        assert children[1].score == 0.6

    def test_thoughts_assigned(self):
        children = self.planner.expand(self.root, ["thought_a"], [0.5])
        assert children[0].thought == "thought_a"

    def test_returns_list_of_nodes(self):
        children = self.planner.expand(self.root, ["x", "y", "z"], [1.0, 0.8, 0.6])
        assert all(isinstance(c, ThoughtNode) for c in children)


# ---------- ToTPlanner.beam_select ----------


class TestBeamSelect:
    def setup_method(self):
        self.planner = ToTPlanner(beam_width=3)
        root = self.planner.add_root("root")
        self.nodes = self.planner.expand(root, ["a", "b", "c", "d"], [0.9, 0.5, 0.8, 0.3])

    def test_returns_top_k(self):
        selected = self.planner.beam_select(self.nodes, k=2)
        assert len(selected) == 2

    def test_highest_scores_selected(self):
        selected = self.planner.beam_select(self.nodes, k=2)
        scores = [n.score for n in selected]
        assert 0.9 in scores
        assert 0.8 in scores

    def test_k_none_uses_beam_width(self):
        selected = self.planner.beam_select(self.nodes)
        assert len(selected) == 3

    def test_k_larger_than_list(self):
        selected = self.planner.beam_select(self.nodes, k=10)
        assert len(selected) == len(self.nodes)

    def test_sorted_descending(self):
        selected = self.planner.beam_select(self.nodes, k=3)
        scores = [n.score for n in selected]
        assert scores == sorted(scores, reverse=True)


# ---------- ToTPlanner.best_path ----------


class TestBestPath:
    def setup_method(self):
        self.planner = ToTPlanner()

    def test_leaf_returns_self(self):
        root = self.planner.add_root("root")
        path = self.planner.best_path(root)
        assert path == [root]

    def test_path_from_root_to_leaf(self):
        root = self.planner.add_root("root")
        children = self.planner.expand(root, ["a", "b"], [0.8, 0.5])
        path = self.planner.best_path(root)
        assert path[0] is root
        assert path[-1] is children[0]  # highest score

    def test_path_length(self):
        root = self.planner.add_root("root")
        children = self.planner.expand(root, ["a"], [0.9])
        self.planner.expand(children[0], ["b"], [0.7])
        path = self.planner.best_path(root)
        assert len(path) == 3

    def test_picks_highest_score_child(self):
        root = self.planner.add_root("root")
        self.planner.expand(root, ["low", "high"], [0.3, 0.9])
        path = self.planner.best_path(root)
        assert path[1].thought == "high"


# ---------- ToTPlanner.all_leaves ----------


class TestAllLeaves:
    def setup_method(self):
        self.planner = ToTPlanner()

    def test_single_root_is_leaf(self):
        root = self.planner.add_root("root")
        leaves = self.planner.all_leaves(root)
        assert leaves == [root]

    def test_two_children_are_leaves(self):
        root = self.planner.add_root("root")
        self.planner.expand(root, ["a", "b"], [0.8, 0.5])
        leaves = self.planner.all_leaves(root)
        assert len(leaves) == 2

    def test_leaf_count_deep_tree(self):
        root = self.planner.add_root("root")
        children = self.planner.expand(root, ["a", "b"], [0.8, 0.5])
        self.planner.expand(children[0], ["c", "d"], [0.6, 0.4])
        leaves = self.planner.all_leaves(root)
        assert len(leaves) == 3  # children[1], c, d


# ---------- ToTPlanner.tree_size ----------


class TestTreeSize:
    def setup_method(self):
        self.planner = ToTPlanner()

    def test_root_only(self):
        root = self.planner.add_root("root")
        assert self.planner.tree_size(root) == 1

    def test_root_plus_two_children(self):
        root = self.planner.add_root("root")
        self.planner.expand(root, ["a", "b"], [0.8, 0.5])
        assert self.planner.tree_size(root) == 3

    def test_deep_tree(self):
        root = self.planner.add_root("root")
        children = self.planner.expand(root, ["a", "b"], [0.8, 0.5])
        self.planner.expand(children[0], ["c"], [0.6])
        assert self.planner.tree_size(root) == 4


# ---------- TOT_PLANNER singleton ----------


class TestTotPlannerSingleton:
    def test_exists(self):
        assert TOT_PLANNER is not None

    def test_is_tot_planner(self):
        assert isinstance(TOT_PLANNER, ToTPlanner)

    def test_default_beam_width(self):
        assert TOT_PLANNER.beam_width == 3

    def test_default_max_depth(self):
        assert TOT_PLANNER.max_depth == 5
