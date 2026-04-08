"""Tests for routing collapse regularizers."""

import pytest
import torch

from src.model.routing_collapse import (
    collapse_score,
    expert_load,
    load_balance_loss,
    routing_entropy,
    z_loss,
)


def test_expert_load_averages_over_tokens():
    probs = torch.tensor([[[0.5, 0.5], [0.25, 0.75]]])
    load = expert_load(probs)
    assert torch.allclose(load, torch.tensor([0.375, 0.625]))


def test_load_balance_loss_zero_for_uniform_distribution():
    probs = torch.full((2, 3, 4), 0.25)
    loss = load_balance_loss(probs)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_load_balance_loss_increases_for_collapsed_distribution():
    uniform = load_balance_loss(torch.full((2, 3, 4), 0.25))
    collapsed = load_balance_loss(torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).expand(2, 3, 4))
    assert collapsed.item() > uniform.item()


def test_routing_entropy_higher_for_uniform_distribution():
    uniform = routing_entropy(torch.full((2, 3, 4), 0.25))
    peaked = routing_entropy(torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).expand(2, 3, 4))
    assert uniform.item() > peaked.item()


def test_z_loss_positive_for_nontrivial_logits():
    logits = torch.randn(2, 3, 4)
    assert z_loss(logits).item() > 0


def test_collapse_score_near_zero_for_uniform_probs():
    score = collapse_score(torch.full((2, 3, 4), 0.25))
    assert score.item() == pytest.approx(0.0, abs=1e-6)


def test_expert_load_rejects_bad_rank():
    with pytest.raises(ValueError):
        expert_load(torch.tensor([1.0]))
