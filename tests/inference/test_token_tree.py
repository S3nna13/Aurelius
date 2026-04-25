from __future__ import annotations

import pytest
import torch

from src.inference.token_tree import TokenTree, TreeNode


VOCAB = 32


def uniform_logits(v=VOCAB):
    return torch.zeros(v)


def peaked_logits(tok, v=VOCAB, peak=10.0):
    logits = torch.zeros(v)
    logits[tok] = peak
    return logits


def make_tree(branching=2, depth=3):
    return TokenTree(branching_factor=branching, max_depth=depth)


def make_logits(depth, v=VOCAB):
    return [uniform_logits(v) for _ in range(depth)]


class TestTreeNodeDataclass:
    def test_default_children(self):
        node = TreeNode(token_id=5, prob=0.9)
        assert node.children == []

    def test_default_path(self):
        node = TreeNode(token_id=5, prob=0.9)
        assert node.path == []

    def test_depth_default(self):
        node = TreeNode(token_id=5, prob=0.9)
        assert node.depth == 0


class TestTokenTreeInit:
    def test_bad_branching_factor(self):
        with pytest.raises(ValueError):
            TokenTree(branching_factor=0)

    def test_bad_max_depth(self):
        with pytest.raises(ValueError):
            TokenTree(branching_factor=2, max_depth=0)

    def test_valid_init(self):
        tree = TokenTree(branching_factor=3, max_depth=2)
        assert tree.branching_factor == 3
        assert tree.max_depth == 2


class TestBuild:
    def test_root_is_sentinel(self):
        tree = make_tree()
        root = tree.build(make_logits(3))
        assert root.token_id == -1
        assert root.prob == 1.0

    def test_root_has_branching_factor_children(self):
        tree = make_tree(branching=2, depth=3)
        root = tree.build(make_logits(3))
        assert len(root.children) == 2

    def test_leaf_depth_equals_max_depth(self):
        tree = make_tree(branching=2, depth=3)
        root = tree.build(make_logits(3))

        def max_depth(node):
            if not node.children:
                return node.depth
            return max(max_depth(c) for c in node.children)

        assert max_depth(root) == 3

    def test_fewer_logit_steps_than_max_depth(self):
        tree = make_tree(branching=2, depth=5)
        root = tree.build(make_logits(2))
        assert tree.depth(root) == 2

    def test_children_probs_sum_leq1(self):
        tree = make_tree(branching=2, depth=2)
        root = tree.build(make_logits(2))
        total = sum(c.prob for c in root.children)
        assert total <= 1.0 + 1e-6

    def test_peaked_logits_top_token_selected(self):
        tree = TokenTree(branching_factor=1, max_depth=1)
        root = tree.build([peaked_logits(tok=7)])
        assert root.children[0].token_id == 7


class TestGetPaths:
    def test_paths_count(self):
        tree = make_tree(branching=2, depth=3)
        root = tree.build(make_logits(3))
        paths = tree.get_paths(root)
        assert len(paths) == 2 ** 3

    def test_paths_length(self):
        tree = make_tree(branching=2, depth=3)
        root = tree.build(make_logits(3))
        paths = tree.get_paths(root)
        assert all(len(p) == 3 for p in paths)

    def test_paths_are_lists_of_int(self):
        tree = make_tree(branching=2, depth=2)
        root = tree.build(make_logits(2))
        paths = tree.get_paths(root)
        for path in paths:
            assert all(isinstance(t, int) for t in path)


class TestBestPath:
    def test_best_path_length(self):
        tree = make_tree(branching=2, depth=3)
        root = tree.build(make_logits(3))
        path = tree.best_path(root)
        assert len(path) == 3

    def test_best_path_is_list_of_int(self):
        tree = make_tree(branching=2, depth=2)
        root = tree.build(make_logits(2))
        path = tree.best_path(root)
        assert all(isinstance(t, int) for t in path)

    def test_best_path_in_get_paths(self):
        tree = make_tree(branching=2, depth=3)
        root = tree.build(make_logits(3))
        best = tree.best_path(root)
        all_paths = tree.get_paths(root)
        assert best in all_paths


class TestPrune:
    def test_prune_removes_low_prob(self):
        tree = make_tree(branching=2, depth=2)
        logits = [torch.zeros(VOCAB)] * 2
        logits[0][0] = 100.0
        logits[0][1] = -100.0
        root = tree.build(logits)
        pruned = tree.prune(root, threshold=0.1)
        assert len(pruned.children) <= 2

    def test_prune_keeps_high_prob(self):
        tree = TokenTree(branching_factor=2, max_depth=1)
        root = tree.build([peaked_logits(0)])
        before = tree.n_nodes(root)
        pruned = tree.prune(root, threshold=1e-9)
        assert tree.n_nodes(pruned) == before


class TestNNodes:
    def test_single_level(self):
        tree = TokenTree(branching_factor=2, max_depth=1)
        root = tree.build(make_logits(1))
        assert tree.n_nodes(root) == 3  # root + 2 children

    def test_two_levels(self):
        tree = TokenTree(branching_factor=2, max_depth=2)
        root = tree.build(make_logits(2))
        assert tree.n_nodes(root) == 1 + 2 + 4


class TestDepth:
    def test_depth_matches_max_depth(self):
        tree = make_tree(branching=2, depth=4)
        root = tree.build(make_logits(4))
        assert tree.depth(root) == 4

    def test_depth_leaf_node(self):
        node = TreeNode(token_id=1, prob=0.5)
        tree = make_tree()
        assert tree.depth(node) == 0
