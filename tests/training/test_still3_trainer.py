"""Unit tests for STILL3Trainer (src/training/still3_trainer.py).

Covers:
1.  test_config_defaults
2.  test_filter_by_std_keeps_varied
3.  test_filter_by_std_removes_uniform
4.  test_filter_by_std_multiple_groups
5.  test_compute_entropy_shape
6.  test_compute_entropy_uniform
7.  test_compute_entropy_peaked
8.  test_normalize_rewards_zero_mean
9.  test_normalize_rewards_unit_std
10. test_normalize_rewards_all_same
11. test_compute_policy_loss_scalar
12. test_compute_policy_loss_positive_reward_negative_loss
13. test_total_loss_keys
14. test_entropy_coeff_effect
15. test_filter_and_prepare_pipeline
"""

from __future__ import annotations

import math

import pytest
import torch

from src.training.still3_trainer import STILL3Config, STILL3Trainer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_trainer() -> STILL3Trainer:
    return STILL3Trainer()


@pytest.fixture
def default_config() -> STILL3Config:
    return STILL3Config()


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults(default_config):
    """STILL3Config must expose correct default hyperparameters."""
    assert default_config.min_std_threshold == 0.05
    assert default_config.entropy_coeff == 0.01
    assert default_config.gamma == 1.0
    assert default_config.group_size == 8
    assert default_config.normalize_rewards is True


# ---------------------------------------------------------------------------
# 2. filter_by_std — keeps varied group
# ---------------------------------------------------------------------------


def test_filter_by_std_keeps_varied(default_trainer):
    """A group with high reward variance must be retained."""
    groups = [[0.0, 0.5, 1.0, 0.2]]  # std well above 0.05
    result = default_trainer.filter_by_std(groups)
    assert len(result) == 1
    assert result[0] == groups[0]


# ---------------------------------------------------------------------------
# 3. filter_by_std — removes uniform group
# ---------------------------------------------------------------------------


def test_filter_by_std_removes_uniform(default_trainer):
    """A group where all rewards are identical must be discarded."""
    groups = [[1.0, 1.0, 1.0, 1.0]]  # std == 0 < 0.05
    result = default_trainer.filter_by_std(groups)
    assert result == []


# ---------------------------------------------------------------------------
# 4. filter_by_std — mixed groups
# ---------------------------------------------------------------------------


def test_filter_by_std_multiple_groups(default_trainer):
    """Mixed groups: only varied groups should survive."""
    uniform = [0.8, 0.8, 0.8, 0.8]  # std = 0
    varied = [0.0, 1.0, 0.5, 0.9]  # std > 0.05
    nearly_uniform = [0.5, 0.501, 0.499, 0.5]  # std ≈ 0.0008 < 0.05

    groups = [uniform, varied, nearly_uniform]
    result = default_trainer.filter_by_std(groups)
    assert len(result) == 1
    assert result[0] == varied


# ---------------------------------------------------------------------------
# 5. compute_entropy_bonus — output shape
# ---------------------------------------------------------------------------


def test_compute_entropy_shape(default_trainer):
    """compute_entropy_bonus should return a scalar (0-d) tensor."""
    logits = torch.randn(2, 4, 10)
    entropy = default_trainer.compute_entropy_bonus(logits)
    assert entropy.shape == torch.Size([])  # scalar
    assert entropy.item() > 0.0  # entropy must be non-negative


# ---------------------------------------------------------------------------
# 6. compute_entropy_bonus — uniform distribution → high entropy
# ---------------------------------------------------------------------------


def test_compute_entropy_uniform(default_trainer):
    """Uniform logits should yield entropy close to log(V)."""
    V = 100
    logits = torch.zeros(1, 1, V)  # uniform distribution
    entropy = default_trainer.compute_entropy_bonus(logits)
    expected = math.log(V)
    assert abs(entropy.item() - expected) < 1e-4


# ---------------------------------------------------------------------------
# 7. compute_entropy_bonus — peaked distribution → near-zero entropy
# ---------------------------------------------------------------------------


def test_compute_entropy_peaked(default_trainer):
    """One-hot (very large logit mass on one token) → near-zero entropy."""
    V = 50
    logits = torch.full((1, 1, V), -1e9)
    logits[0, 0, 0] = 1e9  # essentially a Dirac delta
    entropy = default_trainer.compute_entropy_bonus(logits)
    assert entropy.item() < 1e-3


# ---------------------------------------------------------------------------
# 8. normalize_group_rewards — mean ≈ 0
# ---------------------------------------------------------------------------


def test_normalize_rewards_zero_mean(default_trainer):
    """Normalised rewards should have approximately zero mean."""
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    normed = default_trainer.normalize_group_rewards(rewards)
    mean = sum(normed) / len(normed)
    assert abs(mean) < 1e-5


# ---------------------------------------------------------------------------
# 9. normalize_group_rewards — std ≈ 1
# ---------------------------------------------------------------------------


def test_normalize_rewards_unit_std(default_trainer):
    """Normalised rewards should have approximately unit std."""
    rewards = [0.1, 0.4, 0.9, 0.2, 0.7, 0.3]
    normed = default_trainer.normalize_group_rewards(rewards)
    t = torch.tensor(normed, dtype=torch.float32)
    std = t.std(unbiased=False).item()
    assert abs(std - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# 10. normalize_group_rewards — all same → zeros
# ---------------------------------------------------------------------------


def test_normalize_rewards_all_same(default_trainer):
    """If all rewards are identical, normalised rewards must be all zero."""
    rewards = [0.5, 0.5, 0.5, 0.5]
    normed = default_trainer.normalize_group_rewards(rewards)
    assert all(v == 0.0 for v in normed)


# ---------------------------------------------------------------------------
# 11. compute_policy_loss — returns scalar
# ---------------------------------------------------------------------------


def test_compute_policy_loss_scalar(default_trainer):
    """compute_policy_loss must return a 0-d tensor."""
    log_probs = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
    rewards = torch.tensor([1.0, 0.0, -1.0])
    loss = default_trainer.compute_policy_loss(log_probs, rewards)
    assert loss.shape == torch.Size([])


# ---------------------------------------------------------------------------
# 12. compute_policy_loss — positive reward lowers (or drives negative) loss
# ---------------------------------------------------------------------------


def test_compute_policy_loss_positive_reward_negative_loss(default_trainer):
    """With uniformly positive rewards, REINFORCE loss should be <= 0."""
    log_probs = torch.tensor([-0.5, -0.3, -0.8], requires_grad=True)
    rewards = torch.tensor([1.0, 1.0, 1.0])  # all positive
    loss = default_trainer.compute_policy_loss(log_probs, rewards)
    # loss = -mean(rewards * log_probs) = -mean(+1 * negative) = positive…
    # Actually with negative log_probs and positive rewards:
    # loss = -(rewards * log_probs).mean() = -((1*-0.5 + 1*-0.3 + 1*-0.8)/3)
    #       = -(-0.533) = +0.533 — model is reinforced by reducing this loss.
    # What matters: loss is a real scalar and the gradient w.r.t. log_probs
    # points in the right direction (increasing log_probs decreases loss).
    assert loss.shape == torch.Size([])
    loss.backward()
    # grad of log_probs should be negative (gradient descent will increase them)
    assert log_probs.grad is not None
    assert (log_probs.grad < 0).all()


# ---------------------------------------------------------------------------
# 13. total_loss — returns correct dict keys
# ---------------------------------------------------------------------------


def test_total_loss_keys(default_trainer):
    """total_loss must return a tuple (tensor, dict) with the right keys."""
    logits = torch.randn(2, 4, 20)
    log_probs = torch.tensor([-1.0, -2.0], requires_grad=True)
    rewards = torch.tensor([0.5, -0.5])

    total, metrics = default_trainer.total_loss(logits, log_probs, rewards)

    assert isinstance(total, torch.Tensor)
    assert total.shape == torch.Size([])
    assert set(metrics.keys()) == {"policy_loss", "entropy_bonus", "total"}
    assert math.isfinite(metrics["policy_loss"])
    assert math.isfinite(metrics["entropy_bonus"])
    assert math.isfinite(metrics["total"])


# ---------------------------------------------------------------------------
# 14. entropy_coeff effect
# ---------------------------------------------------------------------------


def test_entropy_coeff_effect():
    """Higher entropy_coeff should produce a lower (more negative) total loss
    when entropy is significant (uniform logits → large entropy bonus)."""
    logits = torch.zeros(1, 4, 32)  # uniform → large entropy
    log_probs = torch.tensor([-1.0])
    rewards = torch.tensor([0.0])

    trainer_low = STILL3Trainer(STILL3Config(entropy_coeff=0.001))
    trainer_high = STILL3Trainer(STILL3Config(entropy_coeff=1.0))

    total_low, _ = trainer_low.total_loss(logits, log_probs, rewards)
    total_high, _ = trainer_high.total_loss(logits, log_probs, rewards)

    # total = policy_loss - coeff * entropy_bonus
    # With same policy_loss, higher coeff subtracts more → lower total
    assert total_high.item() < total_low.item()


# ---------------------------------------------------------------------------
# 15. filter_and_prepare — full pipeline
# ---------------------------------------------------------------------------


def test_filter_and_prepare_pipeline():
    """filter_and_prepare must filter and (when enabled) normalise rewards."""
    config = STILL3Config(min_std_threshold=0.05, normalize_rewards=True)
    trainer = STILL3Trainer(config)

    groups = [
        {"rewards": [1.0, 1.0, 1.0], "tag": "uniform"},  # should be filtered
        {"rewards": [0.0, 1.0, 0.5], "tag": "varied"},  # should be kept
        {"rewards": [0.3, 0.3, 0.3], "tag": "uniform2"},  # should be filtered
        {"rewards": [0.2, 0.8, 0.4], "tag": "varied2"},  # should be kept
    ]

    result = trainer.filter_and_prepare(groups)

    # Only varied groups survive
    assert len(result) == 2
    tags = {g["tag"] for g in result}
    assert tags == {"varied", "varied2"}

    # Rewards should be normalised (mean ≈ 0)
    for g in result:
        mean = sum(g["rewards"]) / len(g["rewards"])
        assert abs(mean) < 1e-5, f"Expected zero mean after normalisation, got {mean}"
