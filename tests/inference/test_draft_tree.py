"""Tests for draft_tree.py — Yggdrasil-inspired equal-growth tree structures."""

from __future__ import annotations

import torch

from src.inference.draft_tree import (
    DraftNode,
    best_leaf_path,
    equal_growth_parent_indices,
    parent_indices,
    root_to_leaf_paths,
    tokens_from_draft,
    tree_causal_mask,
)


class TestEqualGrowthParentIndices:
    """Equal-growth tree: roughly equal nodes per level, static budget."""

    def test_exact_budget(self):
        parents = equal_growth_parent_indices(total_budget=10, nodes_per_level=3)
        assert len(parents) == 10

    def test_root_is_none(self):
        parents = equal_growth_parent_indices(total_budget=10, nodes_per_level=3)
        assert parents[0] is None

    def test_tree_depth(self):
        parents = equal_growth_parent_indices(total_budget=20, nodes_per_level=3)
        depths = [0] * len(parents)
        for i, p in enumerate(parents):
            if p is not None:
                depths[i] = depths[p] + 1
        assert max(depths) >= 2

    def test_edge_case_single_node(self):
        parents = equal_growth_parent_indices(total_budget=1, nodes_per_level=1)
        assert len(parents) == 1
        assert parents[0] is None

    def test_small_budget(self):
        parents = equal_growth_parent_indices(total_budget=5, nodes_per_level=2)
        assert len(parents) == 5


class TestParentIndices:
    """Standard k-ary tree (backward compatibility)."""

    def test_k_ary_structure(self):
        parents = parent_indices(branching_factor=3, depth=2)
        assert len(parents) == 13
        assert parents[0] is None

    def test_root_has_no_parent(self):
        parents = parent_indices(branching_factor=4, depth=2)
        assert parents[0] is None

    def test_children_point_to_correct_parent(self):
        parents = parent_indices(branching_factor=2, depth=2)
        for i in range(1, len(parents)):
            p = parents[i]
            assert p is not None
            assert 0 <= p < i

    def test_exponential_growth(self):
        parents = parent_indices(branching_factor=2, depth=3)
        # Root(1) + depth1(2) + depth2(4) + depth3(8) = 15
        assert len(parents) == 15


class TestRootToLeafPaths:
    """Path extraction from draft tree."""

    def test_single_node_path(self):
        nodes = [DraftNode(token_id=42, score=0.9, parent=None)]
        paths = root_to_leaf_paths(nodes)
        assert len(paths) == 1
        assert 42 in paths[0]

    def test_two_leaves(self):
        nodes = [
            DraftNode(token_id=1, score=0.5, parent=None),
            DraftNode(token_id=2, score=0.4, parent=0),
            DraftNode(token_id=3, score=0.3, parent=0),
        ]
        paths = root_to_leaf_paths(nodes)
        assert len(paths) == 2

    def test_deep_tree(self):
        # root(0,token=1) -> [node1(1,token=2), node3(3,token=4)]; node1 has child node2(2,token=3)
        # Paths: [1,2,3] and [1,4]
        nodes = [
            DraftNode(token_id=1, score=0.5, parent=None),  # 0
            DraftNode(token_id=2, score=0.4, parent=0),  # 1
            DraftNode(token_id=3, score=0.3, parent=1),  # 2
            DraftNode(token_id=4, score=0.35, parent=0),  # 3
        ]
        paths = root_to_leaf_paths(nodes)
        path_sets = [set(p) for p in paths]
        assert {1, 2, 3} in path_sets or [1, 2, 3] in paths  # root->2->3
        assert {1, 4} in path_sets or [1, 4] in paths  # root->4

    def test_empty_for_internal_nodes(self):
        nodes = [
            DraftNode(token_id=1, score=0.5, parent=None),
            DraftNode(token_id=2, score=0.4, parent=0),
        ]
        paths = root_to_leaf_paths(nodes)
        assert len(paths) == 1


class TestBestLeafPath:
    """Best path selection by cumulative log-prob score."""

    def test_prefers_high_scores(self):
        # root -> [token_2(score=0.1), token_3(score=0.9)]; token_2 has child token_4(score=0.2)
        # Path scores: [1,2,4]=0.5+0.1+0.2=0.8, [1,3]=0.5+0.9=1.4 → [1,3] wins
        nodes = [
            DraftNode(token_id=1, score=0.5, parent=None),  # 0: root
            DraftNode(token_id=2, score=0.1, parent=0),  # 1
            DraftNode(token_id=3, score=0.9, parent=0),  # 2
            DraftNode(token_id=4, score=0.2, parent=1),  # 3
        ]
        path = best_leaf_path(nodes)
        # Best path is [1, 3] since its score (1.4) > [1, 2, 4] (0.8)
        assert path == [1, 3]

    def test_tie_breaks_by_length(self):
        # root -> [token_2(score=0.5), token_3(score=0.5)]; token_2 has child token_4(score=0.5)
        # Path scores: [1,2]=1.0, [1,3]=1.0, [1,2,4]=1.5 → [1,2,4] wins
        nodes = [
            DraftNode(token_id=1, score=0.5, parent=None),  # 0
            DraftNode(token_id=2, score=0.5, parent=0),  # 1
            DraftNode(token_id=3, score=0.5, parent=0),  # 2
            DraftNode(token_id=4, score=0.5, parent=1),  # 3
        ]
        path = best_leaf_path(nodes)
        assert path == [1, 2, 4]

    def test_single_node(self):
        nodes = [DraftNode(token_id=99, score=0.7, parent=None)]
        path = best_leaf_path(nodes)
        assert path == [99]


class TestTreeCausalMask:
    """Attention mask for tree-structured verification."""

    def test_mask_shape(self):
        parents = equal_growth_parent_indices(total_budget=10, nodes_per_level=3)
        mask = tree_causal_mask(parents)
        assert mask.shape == (10, 10)

    def test_mask_is_boolean(self):
        parents = equal_growth_parent_indices(total_budget=8, nodes_per_level=2)
        mask = tree_causal_mask(parents)
        assert mask.dtype == torch.bool

    def test_root_attends_to_itself(self):
        parents = [None] + [0] * 3
        mask = tree_causal_mask(parents)
        assert mask[0, 0]

    def test_child_attends_to_parent(self):
        parents = [None, 0, 0, 1]
        mask = tree_causal_mask(parents)
        assert mask[2, 1]
        assert mask[3, 1]

    def test_siblings_attend_to_each_other(self):
        parents = [None, 0, 0]
        mask = tree_causal_mask(parents)
        assert mask[1, 2]
        assert mask[2, 1]


class TestTokensFromDraft:
    """Flatten tree into token sequence for verification."""

    def test_flatten(self):
        nodes = [
            DraftNode(token_id=10, score=0.5, parent=None),
            DraftNode(token_id=20, score=0.4, parent=0),
            DraftNode(token_id=30, score=0.3, parent=0),
            DraftNode(token_id=40, score=0.2, parent=1),
        ]
        tokens, pids = tokens_from_draft(nodes)
        assert list(tokens) == [10, 20, 30, 40]
        assert pids == [None, 0, 0, 1]

    def test_order_preserved(self):
        nodes = [
            DraftNode(token_id=i, score=1.0, parent=None if i == 0 else i - 1) for i in range(5)
        ]
        tokens, _ = tokens_from_draft(nodes)
        assert list(tokens) == list(range(5))
