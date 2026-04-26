"""Tests for draft alignment helpers."""

import pytest
import torch

from src.inference.draft_alignment import alignment_rate, first_misalignment, token_alignment


def test_token_alignment_marks_matching_prefix():
    aligned = token_alignment(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 4]))
    assert torch.equal(aligned, torch.tensor([True, True, False]))


def test_alignment_rate_averages_matches():
    rate = alignment_rate(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 4]))
    assert rate.item() == pytest.approx(2.0 / 3.0)


def test_first_misalignment_returns_none_when_fully_aligned():
    assert first_misalignment(torch.tensor([1, 2]), torch.tensor([1, 2])) is None


def test_first_misalignment_returns_first_index():
    assert first_misalignment(torch.tensor([1, 3, 2]), torch.tensor([1, 2, 2])) == 1


def test_alignment_rate_zero_for_empty_overlap():
    rate = alignment_rate(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
    assert rate.item() == pytest.approx(0.0)


def test_token_alignment_rejects_bad_rank():
    with pytest.raises(ValueError):
        token_alignment(torch.tensor([[1]]), torch.tensor([1]))


def test_first_misalignment_handles_shorter_target():
    assert first_misalignment(torch.tensor([1, 2, 3]), torch.tensor([1, 2])) is None
