"""Tests for grpo_trainer.py — GRPOConfig, GroupSample, sample_group,
compute_group_advantages, compute_grpo_loss, and GRPOTrainer."""

import math

import pytest
import torch

from src.alignment.grpo_trainer import (
    GroupSample,
    GRPOConfig,
    GRPOTrainer,
    compute_group_advantages,
    compute_grpo_loss,
    sample_group,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
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
    return [10, 20, 30]


@pytest.fixture
def grpo_cfg():
    return GRPOConfig(n_group=2, max_new_tokens=4)


@pytest.fixture
def simple_reward_fn():
    return lambda response_ids: 1.0


# ---------------------------------------------------------------------------
# 1. GRPOConfig defaults
# ---------------------------------------------------------------------------


def test_grpo_config_defaults():
    cfg = GRPOConfig()
    assert cfg.n_group == 8
    assert cfg.clip_ratio == pytest.approx(0.2)
    assert cfg.kl_coeff == pytest.approx(0.04)
    assert cfg.max_new_tokens == 128
    assert cfg.temperature == pytest.approx(1.0)
    assert cfg.normalize_rewards is True
    assert cfg.entropy_bonus == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 2. GroupSample fields
# ---------------------------------------------------------------------------


def test_group_sample_fields():
    s = GroupSample(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        log_prob=-2.5,
        reward=0.8,
        advantage=0.3,
    )
    assert s.prompt_ids == [1, 2, 3]
    assert s.response_ids == [4, 5]
    assert s.log_prob == pytest.approx(-2.5)
    assert s.reward == pytest.approx(0.8)
    assert s.advantage == pytest.approx(0.3)


def test_group_sample_default_advantage():
    s = GroupSample(prompt_ids=[1], response_ids=[2], log_prob=-1.0, reward=0.5)
    assert s.advantage == 0.0


# ---------------------------------------------------------------------------
# 3. sample_group returns n_group samples
# ---------------------------------------------------------------------------


def test_sample_group_count(policy_model, prompt_ids, grpo_cfg):
    samples = sample_group(
        policy_model,
        prompt_ids,
        n_group=grpo_cfg.n_group,
        max_new_tokens=grpo_cfg.max_new_tokens,
        temperature=grpo_cfg.temperature,
    )
    assert len(samples) == grpo_cfg.n_group


# ---------------------------------------------------------------------------
# 4. sample_group each sample has non-empty response_ids
# ---------------------------------------------------------------------------


def test_sample_group_nonempty_responses(policy_model, prompt_ids, grpo_cfg):
    samples = sample_group(
        policy_model,
        prompt_ids,
        n_group=grpo_cfg.n_group,
        max_new_tokens=grpo_cfg.max_new_tokens,
        temperature=grpo_cfg.temperature,
    )
    for s in samples:
        assert len(s.response_ids) > 0, "response_ids should be non-empty"


# ---------------------------------------------------------------------------
# 5. sample_group log_prob is negative float
# ---------------------------------------------------------------------------


def test_sample_group_log_prob_negative(policy_model, prompt_ids, grpo_cfg):
    samples = sample_group(
        policy_model,
        prompt_ids,
        n_group=grpo_cfg.n_group,
        max_new_tokens=grpo_cfg.max_new_tokens,
        temperature=grpo_cfg.temperature,
    )
    for s in samples:
        assert isinstance(s.log_prob, float), "log_prob should be a float"
        assert s.log_prob < 0.0, f"log_prob should be negative, got {s.log_prob}"


# ---------------------------------------------------------------------------
# 6. compute_group_advantages mean advantage ~= 0 (normalized)
# ---------------------------------------------------------------------------


def test_compute_group_advantages_mean_zero_normalized():
    rewards = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4]
    samples = [
        GroupSample(prompt_ids=[1], response_ids=[2], log_prob=-1.0, reward=r) for r in rewards
    ]
    result = compute_group_advantages(samples, normalize=True)
    advantages = [s.advantage for s in result]
    mean_adv = sum(advantages) / len(advantages)
    assert abs(mean_adv) < 1e-5, f"Expected mean ≈ 0, got {mean_adv}"


# ---------------------------------------------------------------------------
# 7. compute_group_advantages std advantage ~= 1 (normalized)
# ---------------------------------------------------------------------------


def test_compute_group_advantages_std_one_normalized():
    rewards = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4]
    samples = [
        GroupSample(prompt_ids=[1], response_ids=[2], log_prob=-1.0, reward=r) for r in rewards
    ]
    result = compute_group_advantages(samples, normalize=True)
    advantages = [s.advantage for s in result]
    n = len(advantages)
    mean_adv = sum(advantages) / n
    # population std (ddof=0)
    variance = sum((a - mean_adv) ** 2 for a in advantages) / n
    std_adv = math.sqrt(variance)
    assert abs(std_adv - 1.0) < 0.1, f"Expected std ≈ 1, got {std_adv}"


# ---------------------------------------------------------------------------
# 8. compute_group_advantages without normalize: advantage = reward - mean_reward
# ---------------------------------------------------------------------------


def test_compute_group_advantages_unnormalized():
    rewards = [1.0, 2.0, 3.0, 4.0]
    mean_r = sum(rewards) / len(rewards)
    samples = [
        GroupSample(prompt_ids=[1], response_ids=[2], log_prob=-1.0, reward=r) for r in rewards
    ]
    result = compute_group_advantages(samples, normalize=False)
    for s, r in zip(result, rewards):
        expected = r - mean_r
        assert s.advantage == pytest.approx(expected, abs=1e-6), (
            f"Expected advantage={expected}, got {s.advantage}"
        )


# ---------------------------------------------------------------------------
# 9. compute_grpo_loss returns tensor + dict
# ---------------------------------------------------------------------------


def test_compute_grpo_loss_returns_types(policy_model, ref_model, grpo_cfg, prompt_ids):
    samples = sample_group(
        policy_model,
        prompt_ids,
        n_group=grpo_cfg.n_group,
        max_new_tokens=grpo_cfg.max_new_tokens,
        temperature=grpo_cfg.temperature,
    )
    for i, s in enumerate(samples):
        samples[i] = GroupSample(
            prompt_ids=s.prompt_ids,
            response_ids=s.response_ids,
            log_prob=s.log_prob,
            reward=float(i),
            advantage=float(i) - 0.5,
        )
    loss, metrics = compute_grpo_loss(policy_model, ref_model, samples, grpo_cfg)
    assert isinstance(loss, torch.Tensor), "loss should be a Tensor"
    assert isinstance(metrics, dict), "metrics should be a dict"


# ---------------------------------------------------------------------------
# 10. compute_grpo_loss dict has required keys
# ---------------------------------------------------------------------------


def test_compute_grpo_loss_metric_keys(policy_model, ref_model, grpo_cfg, prompt_ids):
    samples = sample_group(
        policy_model,
        prompt_ids,
        n_group=grpo_cfg.n_group,
        max_new_tokens=grpo_cfg.max_new_tokens,
        temperature=grpo_cfg.temperature,
    )
    for i, s in enumerate(samples):
        samples[i] = GroupSample(
            prompt_ids=s.prompt_ids,
            response_ids=s.response_ids,
            log_prob=s.log_prob,
            reward=float(i),
            advantage=float(i) - 0.5,
        )
    _, metrics = compute_grpo_loss(policy_model, ref_model, samples, grpo_cfg)
    for key in ("policy_loss", "kl", "entropy", "clip_fraction"):
        assert key in metrics, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 11. compute_grpo_loss total loss is finite
# ---------------------------------------------------------------------------


def test_compute_grpo_loss_finite(policy_model, ref_model, grpo_cfg, prompt_ids):
    samples = sample_group(
        policy_model,
        prompt_ids,
        n_group=grpo_cfg.n_group,
        max_new_tokens=grpo_cfg.max_new_tokens,
        temperature=grpo_cfg.temperature,
    )
    samples = compute_group_advantages(samples, normalize=True)
    loss, _ = compute_grpo_loss(policy_model, ref_model, samples, grpo_cfg)
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"


# ---------------------------------------------------------------------------
# 12. GRPOTrainer.train_step returns dict with "loss" key
# ---------------------------------------------------------------------------


def test_trainer_train_step_has_loss(policy_model, ref_model, grpo_cfg, prompt_ids):
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    trainer = GRPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        reward_fn=lambda ids: 1.0,
        config=grpo_cfg,
        optimizer=optimizer,
    )
    metrics = trainer.train_step(prompt_ids)
    assert "loss" in metrics, f"Expected 'loss' in metrics, got {list(metrics.keys())}"


# ---------------------------------------------------------------------------
# 13. GRPOTrainer.train_step returns dict with "mean_reward" key
# ---------------------------------------------------------------------------


def test_trainer_train_step_has_mean_reward(policy_model, ref_model, grpo_cfg, prompt_ids):
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    trainer = GRPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        reward_fn=lambda ids: 0.5,
        config=grpo_cfg,
        optimizer=optimizer,
    )
    metrics = trainer.train_step(prompt_ids)
    assert "mean_reward" in metrics
    assert metrics["mean_reward"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 14. GRPOTrainer.evaluate returns dict with "mean_reward" key
# ---------------------------------------------------------------------------


def test_trainer_evaluate_has_mean_reward(policy_model, ref_model, grpo_cfg, prompt_ids):
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    trainer = GRPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        reward_fn=lambda ids: 0.7,
        config=grpo_cfg,
        optimizer=optimizer,
    )
    result = trainer.evaluate(prompts=[prompt_ids], n_eval=2)
    assert "mean_reward" in result
    assert result["mean_reward"] == pytest.approx(0.7, abs=1e-5)


# ---------------------------------------------------------------------------
# 15. compute_group_advantages single sample gets advantage 0
# ---------------------------------------------------------------------------


def test_compute_group_advantages_single_sample_zero():
    samples = [GroupSample(prompt_ids=[1], response_ids=[2], log_prob=-1.0, reward=5.0)]
    result = compute_group_advantages(samples, normalize=True)
    assert len(result) == 1
    assert result[0].advantage == pytest.approx(0.0, abs=1e-7)
