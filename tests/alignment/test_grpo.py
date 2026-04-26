"""Tests for GRPO implementation — 16+ tests covering all required components."""

from __future__ import annotations

import math

import pytest
import torch

from src.alignment.grpo import (
    GRPOConfig,
    GRPOTrainer,
    # legacy names (kept for backward-compat tests)
    compute_sequence_log_probs,
    group_relative_advantages,
    grpo_policy_loss,
    sample_completions,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


@pytest.fixture
def grpo_cfg():
    return GRPOConfig(
        n_samples=4,
        beta=0.01,
        clip_ratio=0.2,
        kl_coef=0.1,
        max_new_tokens=4,
        temperature=1.0,
    )


@pytest.fixture
def prompt_ids():
    torch.manual_seed(42)
    return torch.randint(0, 256, (1, 8))


# ---------------------------------------------------------------------------
# 1. GRPOConfig defaults
# ---------------------------------------------------------------------------


def test_grpo_config_defaults():
    """GRPOConfig must have the correct default field values."""
    cfg = GRPOConfig()
    assert cfg.n_samples == 8
    assert cfg.beta == pytest.approx(0.01)
    assert cfg.clip_ratio == pytest.approx(0.2)
    assert cfg.kl_coef == pytest.approx(0.1)
    assert cfg.max_new_tokens == 64
    assert cfg.temperature == pytest.approx(1.0)


def test_grpo_config_custom():
    """GRPOConfig must accept custom values."""
    cfg = GRPOConfig(
        n_samples=16, beta=0.05, clip_ratio=0.1, kl_coef=0.05, max_new_tokens=128, temperature=0.7
    )
    assert cfg.n_samples == 16
    assert cfg.beta == pytest.approx(0.05)
    assert cfg.clip_ratio == pytest.approx(0.1)
    assert cfg.kl_coef == pytest.approx(0.05)
    assert cfg.max_new_tokens == 128
    assert cfg.temperature == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 2. group_relative_advantages
# ---------------------------------------------------------------------------


def test_group_relative_advantages_zero_mean():
    """Output advantages must have approximately zero mean."""
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    adv = group_relative_advantages(rewards)
    assert abs(adv.mean().item()) < 1e-5


def test_group_relative_advantages_unit_std():
    """Output advantages must have std close to 1."""
    rewards = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])
    adv = group_relative_advantages(rewards)
    assert abs(adv.std().item() - 1.0) < 0.15


def test_group_relative_advantages_identical_rewards_zeros():
    """When all rewards are identical, advantages must be all zeros."""
    rewards = torch.tensor([3.0, 3.0, 3.0, 3.0])
    adv = group_relative_advantages(rewards)
    assert adv.shape == rewards.shape
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6)


def test_group_relative_advantages_single_element():
    """A single-element group must return zero advantage."""
    rewards = torch.tensor([5.0])
    adv = group_relative_advantages(rewards)
    assert adv.shape == (1,)
    assert adv[0].item() == pytest.approx(0.0, abs=1e-6)


def test_group_relative_advantages_output_shape():
    """Output shape must match input shape."""
    rewards = torch.tensor([1.0, -1.0, 0.5, -0.5, 2.0])
    adv = group_relative_advantages(rewards)
    assert adv.shape == rewards.shape


# ---------------------------------------------------------------------------
# 3. grpo_policy_loss
# ---------------------------------------------------------------------------


def test_grpo_policy_loss_no_op_ratio1_adv0():
    """When ratio=1 and advantages=0, loss should be ~0."""
    log_probs = torch.tensor([-1.0, -2.0, -1.5, -2.5])
    advantages = torch.zeros(4)
    loss = grpo_policy_loss(log_probs, log_probs.detach(), advantages)
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_grpo_policy_loss_equals_negative_mean_when_unclipped():
    """With ratio=1 and large clip window, loss = -mean(advantages)."""
    log_probs = torch.tensor([-1.0, -2.0, -3.0])
    advantages = torch.tensor([1.0, -1.0, 2.0])
    # clip_ratio=10 effectively disables clipping
    loss = grpo_policy_loss(log_probs, log_probs.detach(), advantages, clip_ratio=10.0)
    expected = -advantages.mean().item()
    assert abs(loss.item() - expected) < 1e-5


def test_grpo_policy_loss_clipping_upper():
    """Ratio above 1+clip_ratio must be clipped (large positive advantage).

    With ratio >> 1 and advantage = +1:
      - unclipped objective = ratio * 1.0  (large positive)
      - clipped objective   = 1.2 * 1.0 = 1.2
      - min(unclipped, clipped) = 1.2  (clipping binds)
      - loss = -1.2
    """
    log_probs_old = torch.tensor([-5.0])
    log_probs_new = torch.tensor([-1.0])  # ratio = exp(4) >> 1
    advantages = torch.tensor([1.0])
    clip_ratio = 0.2

    loss = grpo_policy_loss(log_probs_new, log_probs_old.detach(), advantages, clip_ratio)
    # loss = -min(ratio*adv, clipped*adv) = -min(exp(4), 1.2) = -1.2
    assert loss.item() == pytest.approx(-1.2, abs=1e-5)


def test_grpo_policy_loss_clipping_lower():
    """Ratio below 1-clip_ratio must be clipped (large negative advantage)."""
    log_probs_old = torch.tensor([-1.0])
    log_probs_new = torch.tensor([-5.0])  # ratio = exp(-4) << 1
    advantages = torch.tensor([-1.0])
    clip_ratio = 0.2

    loss = grpo_policy_loss(log_probs_new, log_probs_old.detach(), advantages, clip_ratio)
    # Clipped: ratio clamped to 0.8; unclipped ratio*adv = exp(-4)*(-1) ~ 0
    # min(unclipped, clipped) with neg adv: clipped * adv = 0.8 * -1 = -0.8
    # loss = -(-0.8) = 0.8
    assert loss.item() == pytest.approx(0.8, abs=1e-5)


def test_grpo_policy_loss_returns_scalar():
    """grpo_policy_loss must return a 0-d finite tensor."""
    log_probs_new = torch.tensor([-2.0, -3.0, -1.5, -2.5])
    log_probs_old = log_probs_new.clone().detach()
    advantages = torch.tensor([1.0, -0.5, 0.8, -1.3])
    loss = grpo_policy_loss(log_probs_new, log_probs_old, advantages)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 4. compute_sequence_log_probs
# ---------------------------------------------------------------------------


def test_compute_sequence_log_probs_returns_scalar(small_model):
    """compute_sequence_log_probs must return a 0-d tensor."""
    ids = torch.randint(0, 256, (1, 16))
    lp = compute_sequence_log_probs(small_model, ids)
    assert lp.ndim == 0


def test_compute_sequence_log_probs_non_positive(small_model):
    """Log probs of a valid sequence must be <= 0."""
    ids = torch.randint(0, 256, (1, 16))
    lp = compute_sequence_log_probs(small_model, ids)
    assert lp.item() <= 0.0


def test_compute_sequence_log_probs_finite(small_model):
    """compute_sequence_log_probs must return a finite value."""
    ids = torch.randint(0, 256, (1, 16))
    lp = compute_sequence_log_probs(small_model, ids)
    assert torch.isfinite(lp)


def test_compute_sequence_log_probs_with_response_start(small_model):
    """With response_start, must return finite scalar <= 0."""
    ids = torch.randint(0, 256, (1, 16))
    lp = compute_sequence_log_probs(small_model, ids, response_start=8)
    assert lp.ndim == 0
    assert torch.isfinite(lp)
    assert lp.item() <= 0.0


# ---------------------------------------------------------------------------
# 5. sample_completions
# ---------------------------------------------------------------------------


def test_sample_completions_count(small_model, prompt_ids):
    """sample_completions must return exactly n_samples tensors."""
    n_samples = 4
    completions = sample_completions(
        small_model, prompt_ids, n_samples, max_new_tokens=4, temperature=1.0
    )
    assert len(completions) == n_samples


def test_sample_completions_shape(small_model, prompt_ids):
    """Each completion must be a 2-D tensor starting with the prompt length."""
    completions = sample_completions(
        small_model, prompt_ids, n_samples=3, max_new_tokens=4, temperature=1.0
    )
    for c in completions:
        assert c.ndim == 2
        assert c.shape[0] == 1
        assert c.shape[1] >= prompt_ids.shape[1]  # at least prompt tokens


def test_sample_completions_returns_tensors(small_model, prompt_ids):
    """Completions must be torch.Tensor objects."""
    completions = sample_completions(
        small_model, prompt_ids, n_samples=2, max_new_tokens=4, temperature=1.0
    )
    for c in completions:
        assert isinstance(c, torch.Tensor)


# ---------------------------------------------------------------------------
# 6. GRPOTrainer.train_step
# ---------------------------------------------------------------------------


@pytest.fixture
def trainer_and_prompt(small_model, grpo_cfg, prompt_ids):
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)

    # Reward: constant 1.0 per completion token
    def reward_fn(completion_ids):
        return 1.0

    trainer = GRPOTrainer(small_model, None, grpo_cfg, optimizer, reward_fn)
    return trainer, prompt_ids


def test_train_step_returns_dict(trainer_and_prompt):
    """train_step must return a dict."""
    trainer, prompt_ids = trainer_and_prompt
    result = trainer.train_step(prompt_ids)
    assert isinstance(result, dict)


def test_train_step_has_required_keys(trainer_and_prompt):
    """train_step dict must contain 'loss', 'mean_reward', 'advantage_std'."""
    trainer, prompt_ids = trainer_and_prompt
    result = trainer.train_step(prompt_ids)
    assert "loss" in result
    assert "mean_reward" in result
    assert "advantage_std" in result


def test_train_step_loss_finite(trainer_and_prompt):
    """train_step loss must be finite."""
    trainer, prompt_ids = trainer_and_prompt
    result = trainer.train_step(prompt_ids)
    assert math.isfinite(result["loss"])


def test_train_step_mean_reward_matches_reward_fn(small_model, grpo_cfg, prompt_ids):
    """mean_reward must match what the reward_fn returns for uniform rewards."""
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
    fixed_reward = 0.42

    def reward_fn(c):
        return fixed_reward

    trainer = GRPOTrainer(small_model, None, grpo_cfg, optimizer, reward_fn)
    result = trainer.train_step(prompt_ids)
    assert result["mean_reward"] == pytest.approx(fixed_reward, abs=1e-5)


def test_train_step_advantage_std_zero_for_uniform_rewards(small_model, grpo_cfg, prompt_ids):
    """When all rewards are identical, advantage_std should be ~0."""
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)

    def reward_fn(c):
        return 1.0

    trainer = GRPOTrainer(small_model, None, grpo_cfg, optimizer, reward_fn)
    result = trainer.train_step(prompt_ids)
    assert result["advantage_std"] == pytest.approx(0.0, abs=1e-5)


def test_train_step_updates_weights(small_model, prompt_ids):
    """train_step must update at least some model parameters."""
    cfg = GRPOConfig(n_samples=4, max_new_tokens=4, kl_coef=0.1)
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
    # Vary rewards so gradient is non-trivial
    counter = [0]

    def reward_fn(c):
        counter[0] += 1
        return float(counter[0] % 3)  # 1, 2, 0, 1 — non-uniform

    trainer = GRPOTrainer(small_model, None, cfg, optimizer, reward_fn)

    before = {n: p.clone() for n, p in small_model.named_parameters()}
    trainer.train_step(prompt_ids)
    changed = any(
        not torch.equal(before[n], p) for n, p in small_model.named_parameters() if p.requires_grad
    )
    assert changed, "No weights updated after GRPOTrainer.train_step"
