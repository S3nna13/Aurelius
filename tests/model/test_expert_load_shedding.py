"""Tests for expert load shedding helpers."""

import pytest
import torch

from src.model.expert_load_shedding import expert_overflow, kept_fraction, shed_overflow_tokens


def test_expert_overflow_counts_tokens_above_capacity():
    overflow = expert_overflow(torch.tensor([0, 0, 1, 1, 1]), n_experts=2, capacity=2)
    assert torch.equal(overflow, torch.tensor([0, 1]))


def test_shed_overflow_tokens_marks_excess_with_minus_one():
    output = shed_overflow_tokens(torch.tensor([0, 0, 0]), n_experts=1, capacity=2)
    assert torch.equal(output, torch.tensor([0, 0, -1]))


def test_kept_fraction_matches_retained_tokens():
    fraction = kept_fraction(torch.tensor([0, -1, 1, -1]))
    assert fraction.item() == pytest.approx(0.5)


def test_expert_overflow_rejects_bad_rank():
    with pytest.raises(ValueError):
        expert_overflow(torch.tensor([[0, 1]]), n_experts=2, capacity=1)


def test_expert_overflow_rejects_bad_capacity():
    with pytest.raises(ValueError):
        expert_overflow(torch.tensor([0, 1]), n_experts=2, capacity=-1)


def test_expert_overflow_rejects_bad_indices():
    with pytest.raises(ValueError):
        expert_overflow(torch.tensor([0, 2]), n_experts=2, capacity=1)


def test_kept_fraction_handles_empty_assignments():
    assert kept_fraction(torch.tensor([], dtype=torch.long)).item() == pytest.approx(0.0)

