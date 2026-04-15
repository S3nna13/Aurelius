"""Tests for PPO trainer module (src/training/ppo_trainer.py).

Mock models:
- MockPolicy: nn.Embedding + nn.Linear → (None, logits (B, T, V), None)
- MockValue:  nn.Embedding + nn.Linear(d_model, 1) → (None, values (B, T, 1), None)

Constants: vocab=32, d_model=16, seq=6, batch=2
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.ppo_trainer import (
    PPOConfig,
    compute_gae,
    compute_policy_loss,
    compute_value_loss,
    compute_entropy,
    PPOLoss,
    PPOTrainer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB = 32
D_MODEL = 16
SEQ = 6
BATCH = 2


# ---------------------------------------------------------------------------
# Mock models
# ---------------------------------------------------------------------------

class MockPolicy(nn.Module):
    """Tiny policy: Embedding → Linear → logits (B, T, V)."""

    def __init__(self, vocab: int = VOCAB, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab)

    def forward(self, input_ids: torch.Tensor):
        h = self.embed(input_ids)              # (B, T, d_model)
        logits = self.lm_head(h)               # (B, T, V)
        return None, logits, None


class MockValue(nn.Module):
    """Tiny value model: Embedding → Linear → values (B, T, 1)."""

    def __init__(self, vocab: int = VOCAB, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor):
        h = self.embed(input_ids)              # (B, T, d_model)
        values = self.value_head(h)            # (B, T, 1)
        return None, values, None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return PPOConfig()


@pytest.fixture
def policy():
    return MockPolicy()


@pytest.fixture
def value_model():
    return MockValue()


@pytest.fixture
def ref_model():
    return MockPolicy()


@pytest.fixture
def optimizer(policy, value_model):
    params = list(policy.parameters()) + list(value_model.parameters())
    return torch.optim.Adam(params, lr=1e-3)


@pytest.fixture
def trainer(policy, value_model, ref_model, optimizer, config):
    return PPOTrainer(
        policy_model=policy,
        value_model=value_model,
        ref_model=ref_model,
        optimizer=optimizer,
        config=config,
    )


@pytest.fixture
def input_ids():
    return torch.randint(0, VOCAB, (BATCH, SEQ))


@pytest.fixture
def rewards():
    return torch.rand(BATCH)


@pytest.fixture
def old_log_probs(trainer, input_ids):
    with torch.no_grad():
        return trainer.get_log_probs(trainer.policy_model, input_ids).detach()


# ---------------------------------------------------------------------------
# Test 1: PPOConfig defaults
# ---------------------------------------------------------------------------

def test_ppo_config_defaults():
    cfg = PPOConfig()
    assert cfg.clip_eps == 0.2
    assert cfg.value_clip_eps == 0.2
    assert cfg.gamma == 0.99
    assert cfg.lam == 0.95
    assert cfg.value_coef == 0.5
    assert cfg.entropy_coef == 0.01
    assert cfg.kl_coef == 0.1
    assert cfg.max_grad_norm == 1.0
    assert cfg.ppo_epochs == 4
    assert cfg.mini_batch_size == 8


# ---------------------------------------------------------------------------
# Test 2: compute_gae output shapes are (T,)
# ---------------------------------------------------------------------------

def test_compute_gae_output_shapes():
    T = 10
    rewards = torch.randn(T)
    values = torch.randn(T)
    dones = torch.zeros(T)
    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert adv.shape == (T,), f"advantages shape {adv.shape} != ({T},)"
    assert ret.shape == (T,), f"returns shape {ret.shape} != ({T},)"


# ---------------------------------------------------------------------------
# Test 3: compute_gae advantages correct direction (positive reward → positive advantage)
# ---------------------------------------------------------------------------

def test_compute_gae_positive_reward_positive_advantage():
    T = 5
    rewards = torch.ones(T) * 10.0     # strong positive reward
    values = torch.zeros(T)             # value baseline = 0
    dones = torch.zeros(T)
    adv, _ = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert adv.mean().item() > 0.0, "Positive rewards should yield positive mean advantage"


# ---------------------------------------------------------------------------
# Test 4: compute_gae returns = advantages + values
# ---------------------------------------------------------------------------

def test_compute_gae_returns_equals_advantages_plus_values():
    T = 8
    rewards = torch.randn(T)
    values = torch.randn(T)
    dones = torch.zeros(T)
    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    diff = (ret - (adv + values)).abs().max().item()
    assert diff < 1e-5, f"returns != advantages + values, max diff = {diff}"


# ---------------------------------------------------------------------------
# Test 5: compute_policy_loss is scalar, finite
# ---------------------------------------------------------------------------

def test_compute_policy_loss_scalar_finite():
    T = 10
    log_probs = torch.randn(T)
    old_log_probs = torch.randn(T)
    advantages = torch.randn(T)
    loss = compute_policy_loss(log_probs, old_log_probs, advantages, clip_eps=0.2)
    assert loss.shape == (), f"policy loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), "policy loss should be finite"


# ---------------------------------------------------------------------------
# Test 6: compute_policy_loss with ratio=1 equals -mean(advantages)
# ---------------------------------------------------------------------------

def test_compute_policy_loss_ratio_one():
    T = 10
    advantages = torch.randn(T)
    # ratio = exp(lp - old_lp) = 1 when lp == old_lp
    log_probs = torch.randn(T)
    old_log_probs = log_probs.clone()
    loss = compute_policy_loss(log_probs, old_log_probs, advantages, clip_eps=0.2)
    expected = -advantages.mean()
    assert abs(loss.item() - expected.item()) < 1e-5, (
        f"With ratio=1, policy loss should be -mean(advantages). "
        f"Got {loss.item():.6f}, expected {expected.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 7: compute_value_loss is scalar, nonneg
# ---------------------------------------------------------------------------

def test_compute_value_loss_scalar_nonneg():
    T = 10
    values = torch.randn(T)
    old_values = torch.randn(T)
    returns = torch.randn(T)
    loss = compute_value_loss(values, old_values, returns, clip_eps=0.2)
    assert loss.shape == (), f"value loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0.0, f"value loss should be non-negative, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 8: compute_entropy returns scalar
# ---------------------------------------------------------------------------

def test_compute_entropy_scalar():
    log_probs = torch.randn(10)
    ent = compute_entropy(log_probs)
    assert ent.shape == (), f"entropy should be scalar, got shape {ent.shape}"


# ---------------------------------------------------------------------------
# Test 9: PPOLoss forward returns dict with required keys
# ---------------------------------------------------------------------------

def test_ppo_loss_forward_keys(config):
    loss_fn = PPOLoss(config)
    T = 8
    log_probs = torch.randn(T)
    old_log_probs = torch.randn(T).detach()
    advantages = torch.randn(T)
    values = torch.randn(T)
    old_values = torch.randn(T).detach()
    returns = torch.randn(T)

    result = loss_fn(log_probs, old_log_probs, advantages, values, old_values, returns)
    assert "total_loss" in result
    assert "policy_loss" in result
    assert "value_loss" in result
    assert "entropy" in result


# ---------------------------------------------------------------------------
# Test 10: PPOLoss total_loss is scalar
# ---------------------------------------------------------------------------

def test_ppo_loss_total_loss_scalar(config):
    loss_fn = PPOLoss(config)
    T = 8
    log_probs = torch.randn(T)
    old_log_probs = torch.randn(T).detach()
    advantages = torch.randn(T)
    values = torch.randn(T)
    old_values = torch.randn(T).detach()
    returns = torch.randn(T)

    result = loss_fn(log_probs, old_log_probs, advantages, values, old_values, returns)
    assert result["total_loss"].shape == (), (
        f"total_loss should be scalar, got shape {result['total_loss'].shape}"
    )


# ---------------------------------------------------------------------------
# Test 11: PPOTrainer ppo_step returns dict with loss key
# ---------------------------------------------------------------------------

def test_ppo_step_returns_dict_with_loss(trainer, input_ids, rewards, old_log_probs):
    metrics = trainer.ppo_step(input_ids, rewards, old_log_probs)
    assert isinstance(metrics, dict)
    assert "total_loss" in metrics or "policy_loss" in metrics, (
        f"Expected loss key in metrics, got: {list(metrics.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 12: PPOTrainer get_log_probs returns (B,) tensor
# ---------------------------------------------------------------------------

def test_get_log_probs_shape(trainer, input_ids):
    with torch.no_grad():
        lp = trainer.get_log_probs(trainer.policy_model, input_ids)
    assert lp.shape == (BATCH,), f"get_log_probs shape {lp.shape} != ({BATCH},)"


# ---------------------------------------------------------------------------
# Test 13: get_log_probs values are nonpositive
# ---------------------------------------------------------------------------

def test_get_log_probs_nonpositive(trainer, input_ids):
    with torch.no_grad():
        lp = trainer.get_log_probs(trainer.policy_model, input_ids)
    assert (lp <= 0.0).all(), (
        f"All log-probs should be ≤ 0, got max = {lp.max().item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 14: compute_kl_divergence returns scalar
# ---------------------------------------------------------------------------

def test_compute_kl_divergence_scalar(trainer):
    log_probs = torch.randn(BATCH)
    ref_log_probs = torch.randn(BATCH)
    kl = trainer.compute_kl_divergence(log_probs, ref_log_probs)
    assert kl.shape == (), f"KL divergence should be scalar, got shape {kl.shape}"


# ---------------------------------------------------------------------------
# Test 15: ppo_step loss is finite
# ---------------------------------------------------------------------------

def test_ppo_step_loss_finite(trainer, input_ids, rewards, old_log_probs):
    metrics = trainer.ppo_step(input_ids, rewards, old_log_probs)
    for key, val in metrics.items():
        assert isinstance(val, float), f"metric '{key}' is not a float"
        assert math.isfinite(val), f"metric '{key}' is not finite: {val}"


# ---------------------------------------------------------------------------
# Test 16: PPOLoss policy loss lower when advantages strongly positive and ratio > 1
# ---------------------------------------------------------------------------

def test_ppo_loss_policy_loss_lower_with_positive_advantages_and_high_ratio(config):
    """When advantages are strongly positive and ratio > 1 (outside clip),
    the clipped surrogate is smaller (more negative PPO objective → smaller
    positive loss).  The test verifies that larger positive advantages yield
    a more negative policy loss (i.e. the loss should decrease monotonically
    with the advantage magnitude when ratio > 1 is clipped away).
    """
    loss_fn = PPOLoss(config)
    T = 20

    # Identical log_probs, so ratio = exp(0) = 1 by default
    # Push ratio > 1 by making new log_probs > old
    log_probs = torch.zeros(T)           # new policy
    old_log_probs = torch.full((T,), -1.0)  # old policy had lower log-prob → ratio = e > 1

    # Case A: weakly positive advantages
    adv_weak = torch.ones(T) * 0.1
    # Case B: strongly positive advantages
    adv_strong = torch.ones(T) * 10.0

    values = torch.zeros(T)
    old_values = torch.zeros(T)
    returns = torch.zeros(T)

    result_weak = loss_fn(log_probs, old_log_probs, adv_weak, values, old_values, returns)
    result_strong = loss_fn(log_probs, old_log_probs, adv_strong, values, old_values, returns)

    # Strongly positive advantages should give a lower (more negative) policy loss
    # since -min(ratio*A, clip*A) is more negative when A is larger
    assert result_strong["policy_loss"].item() < result_weak["policy_loss"].item(), (
        f"Expected policy_loss with strong advantages ({result_strong['policy_loss'].item():.4f}) "
        f"< weak advantages ({result_weak['policy_loss'].item():.4f})"
    )
