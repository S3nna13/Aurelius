"""Tests for draft path scoring helpers."""

import pytest
import torch

from src.inference.draft_path_scoring import best_path_index, cumulative_path_score, normalized_path_score


def test_cumulative_path_score_sums_nodes():
    assert cumulative_path_score(torch.tensor([1.0, 2.0, 3.0])).item() == pytest.approx(6.0)


def test_normalized_path_score_averages_nodes():
    assert normalized_path_score(torch.tensor([1.0, 2.0, 3.0])).item() == pytest.approx(2.0)


def test_best_path_index_returns_argmax():
    assert best_path_index(torch.tensor([0.1, 0.9, 0.3])) == 1


def test_normalized_path_score_zero_for_empty_path():
    assert normalized_path_score(torch.tensor([])).item() == pytest.approx(0.0)


def test_cumulative_path_score_rejects_bad_rank():
    with pytest.raises(ValueError):
        cumulative_path_score(torch.tensor([[1.0]]))


def test_best_path_index_rejects_empty_scores():
    with pytest.raises(ValueError):
        best_path_index(torch.tensor([]))


def test_best_path_index_rejects_bad_rank():
    with pytest.raises(ValueError):
        best_path_index(torch.tensor([[1.0, 2.0]]))

