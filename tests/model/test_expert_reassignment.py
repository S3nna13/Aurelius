"""Tests for expert reassignment helpers."""

import pytest
import torch

from src.model.expert_reassignment import (
    backup_expert_indices,
    reassign_overflowed_tokens,
    reassignment_success_rate,
)


def test_backup_expert_indices_sorts_descending():
    probs = torch.tensor([[0.1, 0.7, 0.2]])
    order = backup_expert_indices(probs)
    assert torch.equal(order, torch.tensor([[1, 2, 0]]))


def test_reassign_overflowed_tokens_respects_capacities():
    assignments = torch.tensor([0, 0, 0])
    rankings = torch.tensor([[0, 1], [0, 1], [0, 1]])
    capacities = torch.tensor([1, 2])
    out = reassign_overflowed_tokens(assignments, rankings, capacities)
    assert torch.equal(out, torch.tensor([0, 1, 1]))


def test_reassign_overflowed_tokens_marks_unassigned_when_full():
    assignments = torch.tensor([0, 0])
    rankings = torch.tensor([[0], [0]])
    capacities = torch.tensor([1])
    out = reassign_overflowed_tokens(assignments, rankings, capacities)
    assert torch.equal(out, torch.tensor([0, -1]))


def test_reassignment_success_rate_matches_assigned_fraction():
    rate = reassignment_success_rate(torch.tensor([0, -1, 2, -1]))
    assert rate.item() == pytest.approx(0.5)


def test_backup_expert_indices_rejects_bad_rank():
    with pytest.raises(ValueError):
        backup_expert_indices(torch.tensor([0.1, 0.9]))


def test_reassign_overflowed_tokens_rejects_bad_shapes():
    with pytest.raises(ValueError):
        reassign_overflowed_tokens(torch.tensor([0, 1]), torch.tensor([[0, 1]]), torch.tensor([1, 1]))


def test_reassignment_success_rate_handles_empty_assignments():
    assert reassignment_success_rate(torch.tensor([], dtype=torch.long)).item() == pytest.approx(0.0)

