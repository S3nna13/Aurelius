"""Tests for draft-tree helpers."""

import pytest

from src.inference.draft_tree import DraftNode, best_leaf_path, parent_indices, root_to_leaf_paths


def make_nodes():
    return [
        DraftNode(token_id=10, score=0.5, parent=None),
        DraftNode(token_id=11, score=0.8, parent=0),
        DraftNode(token_id=12, score=0.3, parent=0),
        DraftNode(token_id=13, score=0.9, parent=1),
        DraftNode(token_id=14, score=0.1, parent=2),
    ]


def test_parent_indices_builds_tree_structure():
    parents = parent_indices(branching_factor=2, depth=2)
    assert parents == [None, 0, 0, 1, 1, 2, 2]


def test_root_to_leaf_paths_enumerates_paths():
    paths = root_to_leaf_paths(make_nodes())
    assert paths == [[10, 11, 13], [10, 12, 14]]


def test_best_leaf_path_chooses_highest_scoring_path():
    assert best_leaf_path(make_nodes()) == [10, 11, 13]


def test_parent_indices_rejects_bad_args():
    with pytest.raises(ValueError):
        parent_indices(0, 1)


def test_root_to_leaf_paths_handles_single_root():
    assert root_to_leaf_paths([DraftNode(token_id=1, score=0.1, parent=None)]) == [[1]]


def test_best_leaf_path_handles_single_root():
    assert best_leaf_path([DraftNode(token_id=1, score=0.1, parent=None)]) == [1]


def test_parent_indices_depth_zero_returns_root_only():
    assert parent_indices(2, 0) == [None]

