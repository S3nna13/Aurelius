"""Tests for GRPO Advanced: DeepSeekMath-style GRPO with per-token advantages."""

import math

import pytest
import torch

from src.alignment.grpo_advanced import (
    GRPOAdvancedConfig,
    GRPOAdvancedTrainer,
    compute_group_advantages,
    grpo_clipped_loss,
    per_token_advantage,
    sample_group,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture
def policy_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def ref_model(small_cfg):
    torch.manual_seed(1)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def prompt_ids():
    return torch.tensor([[10, 20, 30]], dtype=torch.long)


@pytest.fixture
def default_config():
    return GRPOAdvancedConfig(n_group=2, max_new_tokens=4)


# ---------------------------------------------------------------------------
# GRPOAdvancedConfig tests
# ---------------------------------------------------------------------------


def test_config_defaults():
    """GRPOAdvancedConfig has correct default values."""
    cfg = GRPOAdvancedConfig()
    assert cfg.n_group == 8
    assert cfg.epsilon == pytest.approx(0.2)
    assert cfg.beta == pytest.approx(0.04)
    assert cfg.max_new_tokens == 32
    assert cfg.temperature == pytest.approx(1.0)
    assert cfg.normalize_advantages is True


# ---------------------------------------------------------------------------
# compute_group_advantages tests
# ---------------------------------------------------------------------------


def test_compute_group_advantages_mean_centering():
    """Advantages without normalization should sum to ~0 (mean-centered)."""
    rewards = torch.tensor([0.2, 0.8, 0.4, 0.6, 1.0, 0.1, 0.9, 0.3])
    adv = compute_group_advantages(rewards, normalize=False)
    assert abs(adv.sum().item()) < 1e-5, f"Expected sum ≈ 0, got {adv.sum().item()}"


def test_compute_group_advantages_normalized_std():
    """Normalized advantages should have std ≈ 1."""
    rewards = torch.tensor([0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4])
    adv = compute_group_advantages(rewards, normalize=True)
    assert abs(adv.std().item() - 1.0) < 0.1, f"Expected std ≈ 1, got {adv.std().item()}"


def test_compute_group_advantages_single_sample():
    """Single sample returns advantage of 0."""
    rewards = torch.tensor([5.0])
    adv = compute_group_advantages(rewards, normalize=True)
    assert adv.shape == (1,)
    assert adv[0].item() == pytest.approx(0.0, abs=1e-7)


def test_compute_group_advantages_mean_zero_normalized():
    """Normalized advantages have mean ≈ 0."""
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    adv = compute_group_advantages(rewards, normalize=True)
    assert abs(adv.mean().item()) < 1e-5


# ---------------------------------------------------------------------------
# per_token_advantage tests
# ---------------------------------------------------------------------------


def test_per_token_advantage_shape():
    """per_token_advantage returns (n_group, seq_len) tensor."""
    advantages = torch.tensor([1.0, -0.5, 0.8, -1.3])
    T = 12
    out = per_token_advantage(advantages, T)
    assert out.shape == (4, T)


def test_per_token_advantage_broadcast():
    """Each token in a sequence gets the same advantage as its sequence."""
    advantages = torch.tensor([2.0, -1.0])
    T = 5
    out = per_token_advantage(advantages, T)
    assert out.shape == (2, T)
    # First group: all tokens should be 2.0
    assert torch.allclose(out[0], torch.full((T,), 2.0))
    # Second group: all tokens should be -1.0
    assert torch.allclose(out[1], torch.full((T,), -1.0))


# ---------------------------------------------------------------------------
# grpo_clipped_loss tests
# ---------------------------------------------------------------------------


def test_grpo_clipped_loss_scalar():
    """grpo_clipped_loss returns a finite scalar."""
    n_group, T = 4, 8
    log_probs_policy = torch.randn(n_group, T, requires_grad=True)
    log_probs_ref = torch.randn(n_group, T)
    advantages = torch.randn(n_group, T)
    loss = grpo_clipped_loss(log_probs_policy, log_probs_ref, advantages, epsilon=0.2, beta=0.04)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert log_probs_policy.grad is not None


def test_grpo_clipped_loss_clipping_applied():
    """When ratio is far from 1, clipping kicks in — loss should differ from unclipped."""
    n_group, T = 2, 4
    # Make policy log probs much higher than ref → ratio >> 1
    log_probs_policy = torch.full((n_group, T), -0.5, requires_grad=True)
    log_probs_ref = torch.full((n_group, T), -5.0)
    advantages = torch.ones(n_group, T)  # positive advantages
    epsilon = 0.2
    beta = 0.0  # no KL so we isolate clipping

    loss = grpo_clipped_loss(
        log_probs_policy, log_probs_ref, advantages, epsilon=epsilon, beta=beta
    )

    # With large ratio and positive advantage, clipping should cap the ratio at 1+epsilon
    ratio_val = math.exp(-0.5 - (-5.0))  # ~90
    clipped_ratio = 1.0 + epsilon
    # Unclipped loss would be -(ratio * 1) = large negative, clipped is -(1.2 * 1)
    # clipped loss is smaller in magnitude than unclipped
    unclipped_loss = -(ratio_val * 1.0)
    clipped_loss_expected = -(clipped_ratio * 1.0)
    assert loss.item() == pytest.approx(clipped_loss_expected, rel=1e-4)
    assert loss.item() > unclipped_loss  # clipping reduced the magnitude


# ---------------------------------------------------------------------------
# sample_group tests
# ---------------------------------------------------------------------------


def test_sample_group_output_shapes(policy_model, prompt_ids):
    """sample_group returns tensors of shape (n_group, max_new_tokens)."""
    n_group = 2
    max_new_tokens = 4
    group_ids, group_log_probs = sample_group(
        policy_model, prompt_ids, n_group=n_group, max_new_tokens=max_new_tokens, temperature=1.0
    )
    assert group_ids.shape == (n_group, max_new_tokens)
    assert group_log_probs.shape == (n_group, max_new_tokens)


def test_sample_group_log_probs_finite(policy_model, prompt_ids):
    """Log probs from sample_group should be finite (all valid log probs)."""
    group_ids, group_log_probs = sample_group(
        policy_model, prompt_ids, n_group=2, max_new_tokens=4, temperature=1.0
    )
    assert torch.all(torch.isfinite(group_log_probs))
    # Log probs should be <= 0
    assert torch.all(group_log_probs <= 0.0)


# ---------------------------------------------------------------------------
# GRPOAdvancedTrainer.train_step tests
# ---------------------------------------------------------------------------


def test_trainer_train_step_keys(policy_model, ref_model, prompt_ids):
    """train_step returns dict with required keys."""
    cfg = GRPOAdvancedConfig(n_group=2, max_new_tokens=4)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

    def reward_fn(completion_ids: torch.Tensor) -> float:
        return float(completion_ids.sum().item() % 10) / 10.0

    trainer = GRPOAdvancedTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_fn=reward_fn,
        config=cfg,
        optimizer=optimizer,
    )
    metrics = trainer.train_step(prompt_ids)
    assert "loss" in metrics
    assert "mean_reward" in metrics
    assert "reward_std" in metrics
    assert "mean_advantage" in metrics


def test_trainer_train_step_loss_finite(policy_model, ref_model, prompt_ids):
    """train_step loss should be finite."""
    cfg = GRPOAdvancedConfig(n_group=2, max_new_tokens=4)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

    def reward_fn(completion_ids: torch.Tensor) -> float:
        return 1.0

    trainer = GRPOAdvancedTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_fn=reward_fn,
        config=cfg,
        optimizer=optimizer,
    )
    metrics = trainer.train_step(prompt_ids)
    assert math.isfinite(metrics["loss"])


def test_trainer_train_step_mean_reward(policy_model, ref_model, prompt_ids):
    """mean_reward in output matches the average of reward_fn calls."""
    cfg = GRPOAdvancedConfig(n_group=2, max_new_tokens=4)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

    # Deterministic reward: always returns 0.42
    call_rewards = []

    def reward_fn(completion_ids: torch.Tensor) -> float:
        val = 0.42
        call_rewards.append(val)
        return val

    trainer = GRPOAdvancedTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_fn=reward_fn,
        config=cfg,
        optimizer=optimizer,
    )
    metrics = trainer.train_step(prompt_ids)
    assert len(call_rewards) == cfg.n_group
    expected_mean = sum(call_rewards) / len(call_rewards)
    assert metrics["mean_reward"] == pytest.approx(expected_mean, rel=1e-5)
