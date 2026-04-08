"""Tests for IPO alignment utilities."""

import pytest
import torch

from src.alignment.ipo import ipo_loss, ipo_metrics, ipo_rewards


def test_ipo_loss_returns_scalar_mean():
    loss = ipo_loss(torch.tensor([1.2, 0.9]), torch.tensor([0.2, 0.1]), beta=0.5)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ipo_none_reduction_matches_batch_shape():
    losses = ipo_loss(
        torch.tensor([1.0, 1.5, 2.0]),
        torch.tensor([0.1, 0.3, 0.4]),
        beta=0.25,
        reduction="none",
    )
    assert losses.shape == (3,)


def test_ipo_target_margin_zeroes_loss_when_matched():
    loss = ipo_loss(
        torch.tensor([2.0]),
        torch.tensor([0.0]),
        ref_chosen_logps=torch.tensor([1.0]),
        ref_rejected_logps=torch.tensor([0.0]),
        beta=1.0,
        target_margin=1.0,
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_ipo_rewards_reflect_policy_advantage():
    chosen_reward, rejected_reward = ipo_rewards(
        torch.tensor([1.5]),
        torch.tensor([0.2]),
        torch.tensor([1.0]),
        torch.tensor([0.5]),
        beta=2.0,
    )
    assert chosen_reward.item() == pytest.approx(1.0)
    assert rejected_reward.item() == pytest.approx(-0.6)


def test_ipo_metrics_exposes_preference_gap():
    metrics = ipo_metrics(
        torch.tensor([1.5, 0.8]),
        torch.tensor([0.2, 0.4]),
        torch.tensor([1.0, 0.5]),
        torch.tensor([0.1, 0.3]),
        beta=0.5,
    )
    expected_gap = torch.tensor([0.4, 0.2])
    assert torch.allclose(metrics.preference_gap, expected_gap)


def test_ipo_backward_produces_gradients():
    chosen = torch.tensor([1.1, 0.7], requires_grad=True)
    rejected = torch.tensor([0.2, -0.1], requires_grad=True)
    loss = ipo_loss(chosen, rejected, beta=0.25)
    loss.backward()
    assert chosen.grad is not None
    assert rejected.grad is not None


def test_ipo_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        ipo_loss(torch.tensor([1.0, 2.0]), torch.tensor([0.5]), beta=0.5)

