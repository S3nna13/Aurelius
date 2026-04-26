"""Tests for expert recycling helpers."""

import pytest
import torch

from src.model.expert_recycling import (
    inactive_expert_mask,
    recycle_expert_weights,
    recycled_fraction,
)


def test_inactive_expert_mask_marks_low_usage():
    mask = inactive_expert_mask(torch.tensor([0.0, 0.1, 0.0]), threshold=0.0)
    assert torch.equal(mask, torch.tensor([True, False, True]))


def test_recycle_expert_weights_copies_source_into_inactive_experts():
    weights = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    mask = torch.tensor([True, False, True])
    recycled = recycle_expert_weights(weights, mask, source_expert=1)
    assert torch.equal(recycled[0], weights[1])
    assert torch.equal(recycled[2], weights[1])


def test_recycled_fraction_matches_mask_density():
    frac = recycled_fraction(torch.tensor([True, False, True, False]))
    assert frac.item() == pytest.approx(0.5)


def test_recycle_expert_weights_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        recycle_expert_weights(torch.ones(2, 3), torch.tensor([True]), source_expert=0)


def test_recycle_expert_weights_rejects_bad_source_expert():
    with pytest.raises(ValueError):
        recycle_expert_weights(torch.ones(2, 3), torch.tensor([True, False]), source_expert=2)


def test_recycle_expert_weights_rejects_bad_noise_scale():
    with pytest.raises(ValueError):
        recycle_expert_weights(
            torch.ones(2, 3), torch.tensor([True, False]), source_expert=0, noise_scale=-1.0
        )


def test_recycled_fraction_handles_empty_mask():
    assert recycled_fraction(torch.tensor([], dtype=torch.bool)).item() == pytest.approx(0.0)
