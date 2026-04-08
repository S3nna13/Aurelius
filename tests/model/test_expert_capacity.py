"""Tests for expert capacity helpers."""

import pytest
import torch

from src.model.expert_capacity import capacity_overflow, capacity_utilization, expert_token_counts


def test_expert_token_counts_counts_assignments():
    counts = expert_token_counts(torch.tensor([0, 1, 1, 2]), n_experts=3)
    assert torch.equal(counts, torch.tensor([1, 2, 1]))


def test_capacity_overflow_clamps_below_capacity():
    overflow = capacity_overflow(torch.tensor([1, 5, 2]), capacity=3)
    assert torch.equal(overflow, torch.tensor([0, 2, 0]))


def test_capacity_utilization_computes_mean_fraction():
    util = capacity_utilization(torch.tensor([1, 2, 3]), capacity=4)
    assert util.item() == pytest.approx((1 / 4 + 2 / 4 + 3 / 4) / 3)


def test_expert_token_counts_rejects_bad_rank():
    with pytest.raises(ValueError):
        expert_token_counts(torch.tensor([[0, 1]]), n_experts=2)


def test_expert_token_counts_rejects_bad_indices():
    with pytest.raises(ValueError):
        expert_token_counts(torch.tensor([0, 3]), n_experts=3)


def test_capacity_overflow_rejects_negative_capacity():
    with pytest.raises(ValueError):
        capacity_overflow(torch.tensor([1, 2]), capacity=-1)


def test_capacity_utilization_rejects_zero_capacity():
    with pytest.raises(ValueError):
        capacity_utilization(torch.tensor([1, 2]), capacity=0)

