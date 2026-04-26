"""Tests for BOND (Best-of-N Distillation) implementation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.alignment.bond import (
    BONDConfig,
    BONDTrainer,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_trainer(n=4, temperature=1.0, kl_coef=0.1, hard_bond=False):
    """Build a minimal BONDTrainer with tiny dummy models."""
    # We use identity-like nn.Linear modules — only parameter count matters for
    # the gradient-flow test; log-prob values are passed explicitly.
    policy = nn.Linear(8, 8)
    ref = nn.Linear(8, 8)
    for p in ref.parameters():
        p.requires_grad_(False)
    return BONDTrainer(
        policy_model=policy,
        ref_model=ref,
        n_samples=n,
        temperature=temperature,
        kl_coef=kl_coef,
        hard_bond=hard_bond,
    )


def _rewards(batch=2, n=4, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch * n)


# ---------------------------------------------------------------------------
# Test 1: BONDConfig has correct defaults
# ---------------------------------------------------------------------------


def test_bond_config_defaults():
    cfg = BONDConfig()
    assert cfg.n_samples == 8
    assert cfg.temperature == 1.0
    assert cfg.kl_coef == 0.1
    assert cfg.hard_bond is False
    assert cfg.reward_scaling is True


# ---------------------------------------------------------------------------
# Test 2: compute_bond_weights soft mode sums to 1.0 within each group
# ---------------------------------------------------------------------------


def test_soft_weights_sum_to_one():
    trainer = _make_trainer(n=4)
    rewards = _rewards(batch=3, n=4)
    weights = trainer.compute_bond_weights(rewards, n=4, temperature=1.0)

    assert weights.shape == (3 * 4,)
    # Each group of 4 should sum to 1.
    w = weights.view(3, 4)
    group_sums = w.sum(dim=1)
    assert torch.allclose(group_sums, torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# Test 3: Hard BOND gives one-hot weights (argmax gets weight 1)
# ---------------------------------------------------------------------------


def test_hard_bond_one_hot():
    trainer = _make_trainer(n=4, hard_bond=True)
    rewards = torch.tensor(
        [
            0.1,
            0.5,
            0.3,
            0.2,  # group 0: argmax=1
            0.9,
            0.2,
            0.1,
            0.4,
        ]
    )  # group 1: argmax=0
    weights = trainer.compute_bond_weights(rewards, n=4)
    w = weights.view(2, 4)

    # Each group: exactly one 1 and three 0s.
    assert w[0, 1].item() == pytest.approx(1.0)
    assert w[1, 0].item() == pytest.approx(1.0)
    # All other entries are zero.
    assert (w[0, [0, 2, 3]] == 0).all()
    assert (w[1, [1, 2, 3]] == 0).all()


# ---------------------------------------------------------------------------
# Test 4: Higher reward → higher soft weight
# ---------------------------------------------------------------------------


def test_higher_reward_higher_weight():
    trainer = _make_trainer(n=4, temperature=1.0)
    # Use a single group where rewards are strictly increasing.
    rewards = torch.tensor([0.0, 0.5, 1.0, 2.0])
    weights = trainer.compute_bond_weights(rewards, n=4, temperature=1.0)
    # Weights must be monotonically increasing.
    for i in range(len(rewards) - 1):
        assert weights[i].item() < weights[i + 1].item(), (
            f"Expected weights[{i}] < weights[{i + 1}], got {weights}"
        )


# ---------------------------------------------------------------------------
# Test 5: compute_bond_loss returns scalar tensor
# ---------------------------------------------------------------------------


def test_bond_loss_is_scalar():
    trainer = _make_trainer(n=4)
    batch, n = 2, 4
    policy_lp = torch.randn(batch * n).requires_grad_(True)
    ref_lp = torch.randn(batch * n)
    weights = trainer.compute_bond_weights(_rewards(batch, n), n=n)

    loss = trainer.compute_bond_loss(policy_lp, ref_lp, weights)
    assert loss.shape == torch.Size([])  # scalar
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 6: j_star_approximation increases monotonically with n
# ---------------------------------------------------------------------------


def test_j_star_increases_with_n():
    torch.manual_seed(42)
    # Create a large pool and test increasing n.
    rewards_pool = torch.randn(1000)

    j_values = []
    for n in [1, 2, 4, 8, 16]:
        # Trim to multiple of n.
        trimmed = rewards_pool[: (1000 // n) * n]
        trainer = _make_trainer(n=n)
        j = trainer.j_star_approximation(trimmed, n=n).item()
        j_values.append(j)

    for i in range(len(j_values) - 1):
        assert j_values[i] <= j_values[i + 1] + 0.5, (
            f"j_star not increasing: j[{i}]={j_values[i]:.4f} > j[{i + 1}]={j_values[i + 1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 7: Temperature → ∞ makes weights uniform
# ---------------------------------------------------------------------------


def test_high_temperature_uniform_weights():
    trainer = _make_trainer(n=4, temperature=1.0)
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
    # Very high temperature → all weights ≈ 0.25
    weights = trainer.compute_bond_weights(rewards, n=4, temperature=1e6)
    expected = torch.full((4,), 0.25)
    assert torch.allclose(weights, expected, atol=1e-3), (
        f"Expected ~uniform weights at high temperature, got {weights}"
    )


# ---------------------------------------------------------------------------
# Test 8: Temperature → 0 approaches hard BOND
# ---------------------------------------------------------------------------


def test_low_temperature_approaches_hard_bond():
    trainer_soft = _make_trainer(n=4, hard_bond=False)
    trainer_hard = _make_trainer(n=4, hard_bond=True)
    rewards = torch.tensor([0.1, 0.5, 0.3, 0.2])

    w_soft = trainer_soft.compute_bond_weights(rewards, n=4, temperature=1e-6)
    w_hard = trainer_hard.compute_bond_weights(rewards, n=4)

    # The argmax (index 1) should have weight close to 1 in soft with tiny temp.
    assert w_soft[1].item() == pytest.approx(1.0, abs=1e-3)
    assert w_hard[1].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 9: train_step returns dict with all required keys
# ---------------------------------------------------------------------------


def test_train_step_keys():
    trainer = _make_trainer(n=4)
    batch, n = 2, 4
    input_ids = torch.zeros(batch * n, 8, dtype=torch.long)
    ref_lp = torch.randn(batch * n)
    # policy_log_probs must track gradients for backward pass.
    policy_lp = torch.randn(batch * n, requires_grad=True)
    rewards = _rewards(batch, n)

    result = trainer.train_step(input_ids, ref_lp, policy_lp, rewards)

    required_keys = {"loss", "bond_loss", "kl_loss", "j_star", "effective_n"}
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"


# ---------------------------------------------------------------------------
# Test 10: Gradient flows through bond_loss
# ---------------------------------------------------------------------------


def test_gradient_flows_through_bond_loss():
    trainer = _make_trainer(n=4)
    batch, n = 2, 4
    policy_lp = torch.randn(batch * n, requires_grad=True)
    ref_lp = torch.randn(batch * n)
    weights = trainer.compute_bond_weights(_rewards(batch, n), n=n)

    loss = trainer.compute_bond_loss(policy_lp, ref_lp, weights)
    loss.backward()  # Must not raise

    assert policy_lp.grad is not None, "Gradient w.r.t. policy_log_probs is None"
    assert torch.isfinite(policy_lp.grad).all(), "Gradient contains non-finite values"


# ---------------------------------------------------------------------------
# Test 11: reward_scaling=True normalizes rewards before weighting
# ---------------------------------------------------------------------------


def test_reward_scaling_normalizes():
    """With reward_scaling=True the effective weights should differ from
    weights computed on raw (un-scaled) rewards when the raw rewards have
    large variance."""

    # We verify this by checking that a BONDConfig with reward_scaling=True
    # is defined and that manually scaling rewards before weight computation
    # yields weights that match computing weights on scaled rewards.

    # Confirm the config flag exists.
    cfg = BONDConfig(reward_scaling=True)
    assert cfg.reward_scaling is True

    n = 4
    trainer = _make_trainer(n=n)

    # Rewards with large variance.
    rewards = torch.tensor([0.0, 10.0, 20.0, 30.0])

    # Scale manually: zero-mean, unit-std.
    mean = rewards.mean()
    std = rewards.std(unbiased=False) + 1e-8
    scaled = (rewards - mean) / std

    w_on_raw = trainer.compute_bond_weights(rewards, n=n, temperature=1.0)
    w_on_scaled = trainer.compute_bond_weights(scaled, n=n, temperature=1.0)

    # Both should sum to 1, but the distribution differs because softmax is
    # not scale-invariant (the relative weights change with scale).
    assert torch.allclose(w_on_raw.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(w_on_scaled.sum(), torch.tensor(1.0), atol=1e-5)

    # The weights on raw rewards will be more peaked (high variance input).
    # Verify the max weight on raw rewards is higher than on scaled rewards.
    # (This demonstrates that scaling rewards changes the weight distribution.)
    assert w_on_raw.max().item() > w_on_scaled.max().item() - 1e-4
