"""Tests for SimPO (Simple Preference Optimization) loss."""
import torch
import pytest
from src.alignment.simpo import SimPOLoss


def test_simpo_chosen_higher_reward_lower_loss():
    """When chosen logp >> rejected logp, loss should be near 0."""
    chosen_logps = torch.tensor([-0.5, -0.4, -0.6])
    rejected_logps = torch.tensor([-5.0, -4.8, -5.2])
    loss_fn = SimPOLoss(beta=2.0, gamma=0.5)
    loss, chosen_rewards, rejected_rewards = loss_fn(chosen_logps, rejected_logps)
    assert loss.item() < 0.1


def test_simpo_rejected_higher_logp_high_loss():
    """When rejected logp >> chosen logp, loss should be large (> 5.0 for this input).

    With chosen=-5.0, rejected=-0.5, beta=2.0, gamma=0.5 the margin is ~-9.5
    and loss = -log sigmoid(-9.5) ≈ 9.5.
    """
    chosen_logps = torch.tensor([-5.0, -4.8])
    rejected_logps = torch.tensor([-0.5, -0.4])
    loss_fn = SimPOLoss(beta=2.0, gamma=0.5)
    loss, chosen_rewards, rejected_rewards = loss_fn(chosen_logps, rejected_logps)
    assert loss.item() > 5.0


def test_simpo_margin_gamma_effect():
    """Increasing gamma increases the loss for the same logp pair."""
    chosen_logps = torch.tensor([-1.0, -1.0])
    rejected_logps = torch.tensor([-2.0, -2.0])

    loss_fn_low = SimPOLoss(beta=2.0, gamma=0.5)
    loss_fn_high = SimPOLoss(beta=2.0, gamma=3.0)

    loss_low, _, _ = loss_fn_low(chosen_logps, rejected_logps)
    loss_high, _, _ = loss_fn_high(chosen_logps, rejected_logps)

    assert loss_high.item() > loss_low.item()


def test_simpo_rewards_scale_with_beta():
    """At beta=4.0, rewards are exactly 2x the rewards at beta=2.0."""
    chosen_logps = torch.tensor([-1.0, -2.0, -0.5])
    rejected_logps = torch.tensor([-3.0, -4.0, -2.0])

    loss_fn_b2 = SimPOLoss(beta=2.0, gamma=0.0)
    loss_fn_b4 = SimPOLoss(beta=4.0, gamma=0.0)

    _, chosen_b2, rejected_b2 = loss_fn_b2(chosen_logps, rejected_logps)
    _, chosen_b4, rejected_b4 = loss_fn_b4(chosen_logps, rejected_logps)

    torch.testing.assert_close(chosen_b4, 2.0 * chosen_b2)
    torch.testing.assert_close(rejected_b4, 2.0 * rejected_b2)


def test_simpo_batch_shapes():
    """Output tensors have correct shapes for batch_size=4."""
    batch_size = 4
    chosen_logps = torch.randn(batch_size) - 2.0
    rejected_logps = torch.randn(batch_size) - 3.0

    loss_fn = SimPOLoss(beta=2.0, gamma=0.5)
    loss, chosen_rewards, rejected_rewards = loss_fn(chosen_logps, rejected_logps)

    # loss is a scalar
    assert loss.shape == torch.Size([])
    # rewards have shape (batch,)
    assert chosen_rewards.shape == torch.Size([batch_size])
    assert rejected_rewards.shape == torch.Size([batch_size])


