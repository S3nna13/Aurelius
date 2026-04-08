"""Tests for draft path acceptance helpers."""

import pytest
import torch

from src.inference.draft_path_acceptance import best_accepting_path, mean_acceptance_length, path_acceptance_lengths


def make_paths():
    return [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 4]), torch.tensor([1, 0, 0])]


def test_path_acceptance_lengths_computes_prefix_matches():
    lengths = path_acceptance_lengths(make_paths(), torch.tensor([1, 2, 5]))
    assert torch.equal(lengths, torch.tensor([2, 2, 1]))


def test_best_accepting_path_returns_argmax_index():
    assert best_accepting_path(make_paths(), torch.tensor([1, 2, 5])) == 0


def test_mean_acceptance_length_averages_lengths():
    mean_len = mean_acceptance_length(make_paths(), torch.tensor([1, 2, 5]))
    assert mean_len.item() == pytest.approx((2 + 2 + 1) / 3)


def test_best_accepting_path_rejects_empty_paths():
    with pytest.raises(ValueError):
        best_accepting_path([], torch.tensor([1]))


def test_mean_acceptance_length_handles_empty_paths():
    assert mean_acceptance_length([], torch.tensor([1])).item() == pytest.approx(0.0)


def test_path_acceptance_lengths_handles_empty_target():
    lengths = path_acceptance_lengths([torch.tensor([1, 2])], torch.tensor([], dtype=torch.long))
    assert torch.equal(lengths, torch.tensor([0]))


def test_path_acceptance_lengths_handles_empty_path():
    lengths = path_acceptance_lengths([torch.tensor([], dtype=torch.long)], torch.tensor([1, 2]))
    assert torch.equal(lengths, torch.tensor([0]))

