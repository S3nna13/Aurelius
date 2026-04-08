"""Tests for ORPO utilities."""

import pytest
import torch

from src.alignment.orpo import orpo_loss, orpo_metrics


def test_orpo_loss_returns_scalar_mean():
    loss = orpo_loss(torch.tensor([-0.4, -0.3]), torch.tensor([-1.2, -1.0]))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_orpo_prefers_higher_chosen_log_probs():
    better = orpo_loss(torch.tensor([-0.2]), torch.tensor([-1.5]))
    worse = orpo_loss(torch.tensor([-1.5]), torch.tensor([-0.2]))
    assert better.item() < worse.item()


def test_orpo_none_reduction_matches_shape():
    losses = orpo_loss(
        torch.tensor([-0.3, -0.4, -0.5]),
        torch.tensor([-1.1, -1.0, -0.9]),
        reduction="none",
    )
    assert losses.shape == (3,)


def test_orpo_alpha_scales_preference_term():
    base = orpo_loss(torch.tensor([-0.3]), torch.tensor([-0.9]), alpha=0.5)
    larger = orpo_loss(torch.tensor([-0.3]), torch.tensor([-0.9]), alpha=1.5)
    assert larger.item() > base.item()


def test_orpo_nll_adds_directly_to_loss():
    metrics = orpo_metrics(
        torch.tensor([-0.2]),
        torch.tensor([-0.9]),
        chosen_nll=torch.tensor([0.7]),
        alpha=1.0,
    )
    assert metrics.loss.item() == pytest.approx((metrics.nll + metrics.preference_term).item())


def test_orpo_backward_produces_gradients():
    chosen = torch.tensor([-0.4, -0.3], requires_grad=True)
    rejected = torch.tensor([-1.0, -0.8], requires_grad=True)
    loss = orpo_loss(chosen, rejected)
    loss.backward()
    assert chosen.grad is not None
    assert rejected.grad is not None


def test_orpo_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        orpo_loss(torch.tensor([-0.4, -0.2]), torch.tensor([-1.0]))

