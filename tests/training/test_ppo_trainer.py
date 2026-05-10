"""Tests for PPO trainer (src/training/ppo_trainer.py).

Import path: from aurelius.training.ppo_trainer import ...
≥14 tests covering PPOConfig, RolloutBuffer, PPOLoss, and PPOTrainer.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from aurelius.training.ppo_trainer import (
    PPOConfig,
    PPOLoss,
    PPOTrainer,
    RolloutBuffer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B = 4  # batch size
T = 6  # sequence / timestep length


# ---------------------------------------------------------------------------
# Helpers: tiny trainable models
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Minimal linear model that accepts a (B,) tensor and returns (B,)."""

    def __init__(self, in_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> PPOConfig:
    return PPOConfig()


@pytest.fixture
def loss_fn(cfg) -> PPOLoss:
    return PPOLoss(cfg)


@pytest.fixture
def policy() -> TinyModel:
    return TinyModel()


@pytest.fixture
def value_model() -> TinyModel:
    return TinyModel()


@pytest.fixture
def ref_model() -> TinyModel:
    return TinyModel()


@pytest.fixture
def optimizer(policy, value_model) -> torch.optim.Optimizer:
    params = list(policy.parameters()) + list(value_model.parameters())
    return torch.optim.Adam(params, lr=1e-3)


@pytest.fixture
def trainer(policy, value_model, ref_model, optimizer, cfg, loss_fn) -> PPOTrainer:
    return PPOTrainer(
        policy_model=policy,
        value_model=value_model,
        ref_model=ref_model,
        optimizer=optimizer,
        config=cfg,
        loss_fn=loss_fn,
    )


# ---------------------------------------------------------------------------
# Test 1: PPOConfig defaults
# ---------------------------------------------------------------------------


def test_ppoconfig_defaults():
    cfg = PPOConfig()
    assert cfg.clip_ratio == 0.2
    assert cfg.kl_coef == 0.1
    assert cfg.vf_coef == 0.5
    assert cfg.entropy_coef == 0.01
    assert cfg.gamma == 1.0
    assert cfg.gae_lambda == 0.95
    assert cfg.n_epochs == 4
    assert cfg.minibatch_size == 8


# ---------------------------------------------------------------------------
# Test 2: RolloutBuffer.add increases size
# ---------------------------------------------------------------------------


def test_rollout_buffer_add_increases_size():
    buf = RolloutBuffer()
    assert buf.size() == 0
    for i in range(1, 4):
        buf.add(
            log_probs=torch.randn(B),
            rewards=torch.randn(B),
            values=torch.randn(B),
            ref_log_probs=torch.randn(B),
            masks=torch.ones(B),
        )
        assert buf.size() == i


# ---------------------------------------------------------------------------
# Test 3: RolloutBuffer.clear sets size to 0
# ---------------------------------------------------------------------------


def test_rollout_buffer_clear():
    buf = RolloutBuffer()
    for _ in range(3):
        buf.add(
            log_probs=torch.randn(B),
            rewards=torch.randn(B),
            values=torch.randn(B),
            ref_log_probs=torch.randn(B),
            masks=torch.ones(B),
        )
    assert buf.size() == 3
    buf.clear()
    assert buf.size() == 0


# ---------------------------------------------------------------------------
# Test 4: RolloutBuffer.compute_advantages returns correct shapes
# ---------------------------------------------------------------------------


def test_rollout_buffer_compute_advantages_shapes():
    buf = RolloutBuffer()
    for _ in range(T):
        buf.add(
            log_probs=torch.randn(B),
            rewards=torch.randn(B),
            values=torch.randn(B),
            ref_log_probs=torch.randn(B),
            masks=torch.ones(B),
        )
    advantages, returns = buf.compute_advantages(gamma=1.0, gae_lambda=0.95)
    assert advantages.shape == (T, B), f"advantages shape {advantages.shape} != ({T}, {B})"
    assert returns.shape == (T, B), f"returns shape {returns.shape} != ({T}, {B})"


# ---------------------------------------------------------------------------
# Test 5: PPOLoss.policy_loss is scalar and finite
# ---------------------------------------------------------------------------


def test_policy_loss_scalar_finite(loss_fn):
    log_probs = torch.randn(B, requires_grad=True)
    old_log_probs = torch.randn(B).detach()
    advantages = torch.randn(B)
    loss = loss_fn.policy_loss(log_probs, old_log_probs, advantages)
    assert loss.shape == (), f"policy_loss should be scalar, got {loss.shape}"
    assert torch.isfinite(loss), "policy_loss should be finite"


# ---------------------------------------------------------------------------
# Test 6: Large positive advantage + in-policy → negative (improving) loss
# ---------------------------------------------------------------------------


def test_policy_loss_in_policy_large_positive_advantage_negative(loss_fn):
    """ratio=1 (in-policy) with large positive advantages → loss = -mean(A) < 0."""
    log_probs = torch.zeros(B)  # same as old → ratio = 1
    old_log_probs = torch.zeros(B)
    advantages = torch.ones(B) * 10.0
    loss = loss_fn.policy_loss(log_probs, old_log_probs, advantages)
    assert loss.item() < 0.0, (
        f"Large positive advantage + ratio=1 should yield negative loss, got {loss.item()}"
    )


# ---------------------------------------------------------------------------
# Test 7: PPOLoss.value_loss is MSE (scalar, non-negative)
# ---------------------------------------------------------------------------


def test_value_loss_mse_scalar(loss_fn):
    values = torch.randn(B, requires_grad=True)
    returns = torch.randn(B)
    loss = loss_fn.value_loss(values, returns)
    assert loss.shape == (), f"value_loss should be scalar, got {loss.shape}"
    assert loss.item() >= 0.0, f"value_loss (MSE) should be non-negative, got {loss.item()}"
    # Verify it is MSE
    expected = ((values - returns) ** 2).mean()
    assert abs(loss.item() - expected.item()) < 1e-5


# ---------------------------------------------------------------------------
# Test 8: PPOLoss.kl_loss is scalar
# ---------------------------------------------------------------------------


def test_kl_loss_scalar(loss_fn):
    log_probs = torch.randn(B)
    ref_log_probs = torch.randn(B)
    loss = loss_fn.kl_loss(log_probs, ref_log_probs)
    assert loss.shape == (), f"kl_loss should be scalar, got {loss.shape}"


# ---------------------------------------------------------------------------
# Test 9: PPOLoss.total_loss returns correct keys
# ---------------------------------------------------------------------------


def test_total_loss_correct_keys(loss_fn):
    log_probs = torch.randn(B, requires_grad=True)
    old_log_probs = torch.randn(B).detach()
    advantages = torch.randn(B)
    values = torch.randn(B, requires_grad=True)
    returns = torch.randn(B)
    ref_log_probs = torch.randn(B).detach()

    total, metrics = loss_fn.total_loss(
        log_probs, old_log_probs, advantages, values, returns, ref_log_probs
    )
    assert "policy_loss" in metrics, "Missing key 'policy_loss'"
    assert "value_loss" in metrics, "Missing key 'value_loss'"
    assert "kl_loss" in metrics, "Missing key 'kl_loss'"


# ---------------------------------------------------------------------------
# Test 10: Gradient flows through policy_loss
# ---------------------------------------------------------------------------


def test_policy_loss_gradient_flows(loss_fn):
    log_probs = torch.randn(B, requires_grad=True)
    old_log_probs = torch.randn(B).detach()
    advantages = torch.randn(B)

    loss = loss_fn.policy_loss(log_probs, old_log_probs, advantages)
    loss.backward()

    assert log_probs.grad is not None, "Gradient did not flow to log_probs"
    assert torch.isfinite(log_probs.grad).all(), "Gradient contains non-finite values"


# ---------------------------------------------------------------------------
# Test 11: PPOTrainer.freeze_ref freezes all ref params
# ---------------------------------------------------------------------------


def test_freeze_ref(trainer):
    # First ensure they are trainable
    for p in trainer.ref_model.parameters():
        p.requires_grad_(True)

    trainer.freeze_ref()

    for name, p in trainer.ref_model.named_parameters():
        assert not p.requires_grad, f"Parameter {name} should be frozen after freeze_ref()"


# ---------------------------------------------------------------------------
# Test 12: PPOTrainer.ppo_step returns correct keys
# ---------------------------------------------------------------------------


def test_ppo_step_returns_correct_keys(trainer):
    x = torch.randn(B, 1)
    log_probs = trainer.policy_model(x).detach().clone().requires_grad_(True)
    # Recompute with grad via a fresh forward
    log_probs_grad = trainer.policy_model(x)
    old_log_probs = log_probs.detach()
    advantages = torch.randn(B)
    values = trainer.value_model(x)
    returns = torch.randn(B)
    ref_log_probs = torch.randn(B).detach()

    metrics = trainer.ppo_step(
        log_probs=log_probs_grad,
        old_log_probs=old_log_probs,
        advantages=advantages,
        values=values,
        returns=returns,
        ref_log_probs=ref_log_probs,
    )
    assert "total_loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "kl_loss" in metrics


# ---------------------------------------------------------------------------
# Test 13: PPOTrainer.ppo_step loss is finite
# ---------------------------------------------------------------------------


def test_ppo_step_loss_finite(trainer):
    x = torch.randn(B, 1)
    log_probs_grad = trainer.policy_model(x)
    old_log_probs = log_probs_grad.detach()
    advantages = torch.randn(B)
    values = trainer.value_model(x)
    returns = torch.randn(B)
    ref_log_probs = torch.randn(B).detach()

    metrics = trainer.ppo_step(
        log_probs=log_probs_grad,
        old_log_probs=old_log_probs,
        advantages=advantages,
        values=values,
        returns=returns,
        ref_log_probs=ref_log_probs,
    )
    for key, val in metrics.items():
        assert isinstance(val, float), f"metric '{key}' should be float, got {type(val)}"
        assert math.isfinite(val), f"metric '{key}' is not finite: {val}"


# ---------------------------------------------------------------------------
# Test 14: PPOTrainer.compute_returns shapes correct
# ---------------------------------------------------------------------------


def test_compute_returns_shapes(trainer):
    rewards = torch.randn(T)
    values = torch.randn(T + 1)  # T+1 to include bootstrap value
    advantages, returns = trainer.compute_returns(rewards, values)
    assert advantages.shape == (T,), f"advantages shape {advantages.shape} != ({T},)"
    assert returns.shape == (T,), f"returns shape {returns.shape} != ({T},)"


# ---------------------------------------------------------------------------
# Test 15: compute_returns: returns = advantages + values[:T]
# ---------------------------------------------------------------------------


def test_compute_returns_consistency(trainer):
    rewards = torch.randn(T)
    values = torch.randn(T + 1)
    advantages, returns = trainer.compute_returns(rewards, values)
    diff = (returns - (advantages + values[:T])).abs().max().item()
    assert diff < 1e-5, f"returns != advantages + values[:T], max diff = {diff}"


# ---------------------------------------------------------------------------
# Test 16: RolloutBuffer.compute_advantages returns finite tensors
# ---------------------------------------------------------------------------


def test_rollout_buffer_advantages_finite():
    buf = RolloutBuffer()
    for _ in range(5):
        buf.add(
            log_probs=torch.randn(B),
            rewards=torch.randn(B),
            values=torch.randn(B),
            ref_log_probs=torch.randn(B),
            masks=torch.ones(B),
        )
    advantages, returns = buf.compute_advantages(gamma=0.99, gae_lambda=0.95)
    assert torch.isfinite(advantages).all(), "advantages contain non-finite values"
    assert torch.isfinite(returns).all(), "returns contain non-finite values"
