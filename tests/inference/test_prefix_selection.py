"""Tests for prefix selection helpers."""

import pytest
import torch

from src.inference.prefix_selection import longest_common_prefix, prefix_match_mask, select_prefix


def test_longest_common_prefix_full_match():
    assert longest_common_prefix(torch.tensor([1, 2]), torch.tensor([1, 2])) == 2


def test_longest_common_prefix_stops_on_mismatch():
    assert longest_common_prefix(torch.tensor([1, 2, 3]), torch.tensor([1, 4, 3])) == 1


def test_select_prefix_returns_matching_slice():
    selected = select_prefix(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 0]))
    assert torch.equal(selected, torch.tensor([1, 2]))


def test_prefix_match_mask_marks_prefix_only():
    mask = prefix_match_mask(torch.tensor([1, 2, 3]), torch.tensor([1, 0, 3]))
    assert torch.equal(mask, torch.tensor([True, False, False]))


def test_longest_common_prefix_handles_empty_tensor():
    assert (
        longest_common_prefix(
            torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        )
        == 0
    )


def test_longest_common_prefix_rejects_bad_rank():
    with pytest.raises(ValueError):
        longest_common_prefix(torch.tensor([[1]]), torch.tensor([1]))


def test_prefix_match_mask_shape_matches_candidate():
    candidate = torch.tensor([1, 2, 3])
    mask = prefix_match_mask(candidate, torch.tensor([1, 2, 4]))
    assert mask.shape == candidate.shape
