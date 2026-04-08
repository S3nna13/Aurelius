"""Tests for expert pruning helpers."""

import pytest
import torch

from src.model.expert_pruning import (
    apply_expert_pruning,
    expert_importance,
    prune_mask,
    retained_fraction,
)


def test_expert_importance_reduces_over_tokens():
    probs = torch.tensor([[[0.8, 0.2], [0.6, 0.4]]])
    importance = expert_importance(probs)
    assert torch.allclose(importance, torch.tensor([0.7, 0.3]))


def test_prune_mask_keeps_top_k():
    probs = torch.tensor([[[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]]])
    mask = prune_mask(probs, keep_k=2)
    assert mask.sum().item() == 2
    assert mask[0].item()


def test_apply_expert_pruning_zeros_pruned_rows():
    weights = torch.ones(3, 2)
    mask = torch.tensor([True, False, True])
    pruned = apply_expert_pruning(weights, mask)
    assert torch.equal(pruned[1], torch.zeros(2))


def test_retained_fraction_matches_mask_density():
    mask = torch.tensor([True, False, True, False])
    assert retained_fraction(mask).item() == pytest.approx(0.5)


def test_prune_mask_rejects_bad_keep_k():
    with pytest.raises(ValueError):
        prune_mask(torch.ones(1, 2, 3), keep_k=0)


def test_apply_expert_pruning_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        apply_expert_pruning(torch.ones(2, 3), torch.tensor([True]))


def test_expert_importance_rejects_bad_rank():
    with pytest.raises(ValueError):
        expert_importance(torch.tensor([1.0]))
