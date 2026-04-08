"""Tests for tree-structured speculative decoding (tree_speculative.py)."""
from __future__ import annotations

import torch
import pytest

from src.inference.tree_speculative import (
    DraftNode,
    DraftTree,
    build_draft_tree,
    verify_tree,
    TreeSpeculativeDecoder,
)

# ---------------------------------------------------------------------------
# Shared mock model
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256


class MockModel:
    """Tiny mock that returns random logits; matches the (loss, logits, kv) API."""

    def __call__(self, input_ids: torch.Tensor):
        B, S = input_ids.shape
        logits = torch.randn(B, S, VOCAB_SIZE)
        return (torch.tensor(0.0), logits, None)


# ---------------------------------------------------------------------------
# DraftTree unit tests
# ---------------------------------------------------------------------------

class TestDraftTreeAddAndPaths:
    """test_draft_tree_add_and_paths: root + 2 children → 2 paths."""

    def test_paths_count(self):
        tree = DraftTree()
        root0 = tree.add_node(token_id=10, parent_idx=-1, log_prob=-0.5)
        root1 = tree.add_node(token_id=20, parent_idx=-1, log_prob=-0.7)
        # Both nodes have parent_idx=-1, so they are both roots (treated as
        # independent root-level nodes); each is also a leaf.
        p = tree.paths()
        assert len(p) == 2

    def test_paths_with_root_and_two_children(self):
        """Explicit tree: one root node, two child nodes → 2 leaf paths."""
        tree = DraftTree()
        root = tree.add_node(token_id=1, parent_idx=-1, log_prob=-0.1)
        child0 = tree.add_node(token_id=2, parent_idx=root, log_prob=-0.2)
        child1 = tree.add_node(token_id=3, parent_idx=root, log_prob=-0.3)
        paths = tree.paths()
        assert len(paths) == 2
        # Each path should start with the root token
        for path in paths:
            assert path[0] == 1
        # Leaf tokens should be 2 and 3 (in some order)
        leaf_tokens = sorted(path[-1] for path in paths)
        assert leaf_tokens == [2, 3]


class TestDraftTreeDepth:
    """test_draft_tree_depth: tree with depth 3 returns depth() == 3."""

    def test_depth_three(self):
        tree = DraftTree()
        n0 = tree.add_node(token_id=1, parent_idx=-1, log_prob=-0.1)
        n1 = tree.add_node(token_id=2, parent_idx=n0, log_prob=-0.2)
        n2 = tree.add_node(token_id=3, parent_idx=n1, log_prob=-0.3)
        assert tree.depth() == 3

    def test_depth_one(self):
        tree = DraftTree()
        tree.add_node(token_id=5, parent_idx=-1, log_prob=-0.1)
        assert tree.depth() == 1

    def test_depth_empty(self):
        tree = DraftTree()
        assert tree.depth() == 0


class TestDraftTreeGetPathTo:
    """test_draft_tree_get_path_to: correct token sequence root-to-node."""

    def test_root_to_leaf(self):
        tree = DraftTree()
        n0 = tree.add_node(token_id=10, parent_idx=-1, log_prob=-0.1)
        n1 = tree.add_node(token_id=20, parent_idx=n0, log_prob=-0.2)
        n2 = tree.add_node(token_id=30, parent_idx=n1, log_prob=-0.3)
        path = tree.get_path_to(n2)
        assert path == [10, 20, 30]

    def test_get_path_to_root(self):
        tree = DraftTree()
        n0 = tree.add_node(token_id=42, parent_idx=-1, log_prob=-0.5)
        assert tree.get_path_to(n0) == [42]


# ---------------------------------------------------------------------------
# build_draft_tree tests
# ---------------------------------------------------------------------------

class TestBuildDraftTreeStructure:
    """test_build_draft_tree_structure: n_branches=2, depth=2 → correct shape."""

    def test_tree_has_nodes(self):
        model = MockModel()
        input_ids = torch.zeros(1, 4, dtype=torch.long)
        tree = build_draft_tree(model, input_ids, n_branches=2, depth=2)
        # Level 1: 2 nodes; Level 2: 2 nodes (one per branch) → 4 total
        assert tree.size() == 4

    def test_tree_has_two_paths(self):
        model = MockModel()
        input_ids = torch.zeros(1, 4, dtype=torch.long)
        tree = build_draft_tree(model, input_ids, n_branches=2, depth=2)
        paths = tree.paths()
        assert len(paths) == 2

    def test_each_path_has_correct_depth(self):
        model = MockModel()
        input_ids = torch.zeros(1, 4, dtype=torch.long)
        tree = build_draft_tree(model, input_ids, n_branches=2, depth=2)
        for path in tree.paths():
            assert len(path) == 2

    def test_depth_matches_requested(self):
        model = MockModel()
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        tree = build_draft_tree(model, input_ids, n_branches=3, depth=3)
        assert tree.depth() == 3

    def test_single_branch_single_depth(self):
        model = MockModel()
        input_ids = torch.zeros(1, 2, dtype=torch.long)
        tree = build_draft_tree(model, input_ids, n_branches=1, depth=1)
        assert tree.size() == 1
        assert len(tree.paths()) == 1


# ---------------------------------------------------------------------------
# verify_tree tests
# ---------------------------------------------------------------------------

class TestVerifyTreeReturnsTokens:
    """test_verify_tree_returns_tokens: result is list[int]."""

    def test_return_type(self):
        model = MockModel()
        input_ids = torch.zeros(1, 4, dtype=torch.long)
        tree = build_draft_tree(model, input_ids, n_branches=2, depth=2)
        result = verify_tree(model, input_ids, tree)
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)


class TestVerifyTreeAtLeastOneToken:
    """test_verify_tree_at_least_one_token: always returns ≥1 token."""

    def test_at_least_one_with_normal_tree(self):
        model = MockModel()
        input_ids = torch.zeros(1, 4, dtype=torch.long)
        tree = build_draft_tree(model, input_ids, n_branches=2, depth=3)
        result = verify_tree(model, input_ids, tree)
        assert len(result) >= 1

    def test_at_least_one_with_empty_tree(self):
        model = MockModel()
        input_ids = torch.zeros(1, 4, dtype=torch.long)
        tree = DraftTree()   # deliberately empty
        result = verify_tree(model, input_ids, tree)
        assert len(result) >= 1

    def test_at_least_one_multiple_runs(self):
        """Stochastic test: run many times to be confident about the guarantee."""
        torch.manual_seed(0)
        model = MockModel()
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        for _ in range(20):
            tree = build_draft_tree(model, input_ids, n_branches=2, depth=2)
            result = verify_tree(model, input_ids, tree)
            assert len(result) >= 1


# ---------------------------------------------------------------------------
# TreeSpeculativeDecoder tests
# ---------------------------------------------------------------------------

class TestTreeSpeculativeGenerates:
    """test_tree_speculative_generates: generate returns tensor of correct shape."""

    def test_output_shape(self):
        draft = MockModel()
        target = MockModel()
        decoder = TreeSpeculativeDecoder(
            target_model=target,
            draft_model=draft,
            n_branches=2,
            depth=2,
            temperature=1.0,
        )
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        out = decoder.generate(input_ids, max_new_tokens=4)
        assert out.shape == (1, 9)

    def test_output_is_tensor(self):
        draft = MockModel()
        target = MockModel()
        decoder = TreeSpeculativeDecoder(target_model=target, draft_model=draft)
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        out = decoder.generate(input_ids, max_new_tokens=2)
        assert isinstance(out, torch.Tensor)


class TestTreeSpeculativeLongerThanInput:
    """test_tree_speculative_longer_than_input: output is longer than input."""

    def test_output_longer(self):
        draft = MockModel()
        target = MockModel()
        decoder = TreeSpeculativeDecoder(
            target_model=target,
            draft_model=draft,
            n_branches=2,
            depth=2,
        )
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        out = decoder.generate(input_ids, max_new_tokens=6)
        assert out.shape[1] > input_ids.shape[1]

    def test_output_prefix_matches_input(self):
        """The first seq_len tokens of output should equal input_ids."""
        draft = MockModel()
        target = MockModel()
        decoder = TreeSpeculativeDecoder(
            target_model=target,
            draft_model=draft,
            n_branches=2,
            depth=2,
        )
        input_ids = torch.arange(5, dtype=torch.long).unsqueeze(0)  # [[0,1,2,3,4]]
        out = decoder.generate(input_ids, max_new_tokens=3)
        assert torch.equal(out[:, : input_ids.shape[1]], input_ids)
