"""Tests for speculative acceptance helpers."""

import pytest
import torch

from src.inference.speculative_acceptance import (
    acceptance_mask,
    acceptance_stats,
    accept_prefix,
)


def test_accept_prefix_returns_full_length_for_match():
    assert accept_prefix(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])) == 3


def test_accept_prefix_stops_on_mismatch():
    assert accept_prefix(torch.tensor([1, 2, 4]), torch.tensor([1, 2, 3])) == 2


def test_acceptance_mask_marks_only_prefix():
    mask = acceptance_mask(torch.tensor([1, 2, 4]), torch.tensor([1, 2, 3]))
    assert torch.equal(mask, torch.tensor([True, True, False]))


def test_acceptance_stats_aggregates_multiple_sequences():
    stats = acceptance_stats(
        [torch.tensor([1, 2]), torch.tensor([3, 4])],
        [torch.tensor([1, 0]), torch.tensor([3, 4])],
    )
    assert stats.accepted == 3
    assert stats.proposed == 4


def test_acceptance_stats_rate_property():
    stats = acceptance_stats([torch.tensor([1])], [torch.tensor([1])])
    assert stats.rate == pytest.approx(1.0)


def test_accept_prefix_rejects_bad_rank():
    with pytest.raises(ValueError):
        accept_prefix(torch.tensor([[1]]), torch.tensor([1]))


def test_acceptance_stats_rejects_length_mismatch():
    with pytest.raises(ValueError):
        acceptance_stats([torch.tensor([1])], [])
