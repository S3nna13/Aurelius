"""Tests for src/alignment/reward_shaping.py — composite reward pipeline."""

import pytest
import torch

from src.alignment.reward_shaping import (
    RewardShapingConfig,
    RunningMeanStd,
    clip_rewards,
    kl_reward_penalty,
    length_penalty,
    diversity_bonus,
    CompositeRewardFunction,
)

B, T = 4, 8


# ---------------------------------------------------------------------------
# RewardShapingConfig
# ---------------------------------------------------------------------------

def test_reward_shaping_config_defaults():
    cfg = RewardShapingConfig()
    assert cfg.reward_clip == 5.0
    assert cfg.normalize_rewards is True
    assert cfg.kl_penalty_coeff == 0.1
    assert cfg.length_penalty_coeff == 0.0
    assert cfg.diversity_bonus_coeff == 0.0
    assert cfg.running_stats_alpha == 0.01


# ---------------------------------------------------------------------------
# clip_rewards
# ---------------------------------------------------------------------------

def test_clip_rewards_clips_out_of_range():
    rewards = torch.tensor([-10.0, -6.0, 0.0, 6.0, 10.0])
    clipped = clip_rewards(rewards, clip=5.0)
    assert clipped.max().item() <= 5.0
    assert clipped.min().item() >= -5.0
    assert clipped[0].item() == pytest.approx(-5.0)
    assert clipped[-1].item() == pytest.approx(5.0)


def test_clip_rewards_does_not_modify_in_range():
    rewards = torch.tensor([-3.0, -1.0, 0.0, 2.5, 4.9])
    clipped = clip_rewards(rewards, clip=5.0)
    assert torch.allclose(clipped, rewards, atol=1e-6)


# ---------------------------------------------------------------------------
# RunningMeanStd
# ---------------------------------------------------------------------------

def test_running_mean_std_update_and_properties():
    rms = RunningMeanStd(alpha=1.0)  # alpha=1 → fully replace on first update
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    rms.update(x)
    # After single update with alpha=1, mean should equal batch mean
    assert abs(rms.mean - x.mean().item()) < 1e-5
    assert rms.std >= 0.0


def test_running_mean_std_normalize_reduces_scale():
    rms = RunningMeanStd(alpha=1.0)
    # Large-valued tensor
    x = torch.tensor([100.0, 200.0, 300.0, 400.0])
    rms.update(x)
    # After update, normalizing should bring values much closer to 0
    normalized = rms.normalize(x)
    assert normalized.abs().max().item() < x.abs().max().item()


# ---------------------------------------------------------------------------
# kl_reward_penalty
# ---------------------------------------------------------------------------

def test_kl_reward_penalty_shape():
    log_probs_policy = torch.randn(B, T)
    log_probs_ref = torch.randn(B, T)
    penalty = kl_reward_penalty(log_probs_policy, log_probs_ref, coeff=0.1)
    assert penalty.shape == (B,)


def test_kl_reward_penalty_zero_when_same():
    log_probs = torch.randn(B, T)
    penalty = kl_reward_penalty(log_probs, log_probs, coeff=0.1)
    assert torch.allclose(penalty, torch.zeros(B), atol=1e-6)


# ---------------------------------------------------------------------------
# length_penalty
# ---------------------------------------------------------------------------

def test_length_penalty_shape():
    seq_lengths = torch.randint(1, T + 1, (B,))
    pen = length_penalty(seq_lengths, max_len=T, coeff=1.0)
    assert pen.shape == (B,)


def test_length_penalty_negative_coeff_gives_negative():
    # All sequences with length > 0 should get negative penalty with negative coeff
    seq_lengths = torch.tensor([4, 6, 8, 2])
    pen = length_penalty(seq_lengths, max_len=8, coeff=-1.0)
    assert pen.shape == (B,)
    assert (pen <= 0).all(), "Negative coeff should produce non-positive penalties"
    # Longer sequences should have more negative penalty
    assert pen[2].item() < pen[3].item()


# ---------------------------------------------------------------------------
# diversity_bonus
# ---------------------------------------------------------------------------

def test_diversity_bonus_shape():
    ids_batch = [torch.randint(0, 100, (T,)) for _ in range(B)]
    bonus = diversity_bonus(ids_batch, coeff=1.0)
    assert bonus.shape == (B,)


def test_diversity_bonus_identical_sequences_zero():
    # All sequences are the same → Jaccard similarity = 1 → bonus = coeff * (1 - 1) = 0
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    ids_batch = [ids.clone() for _ in range(B)]
    bonus = diversity_bonus(ids_batch, coeff=1.0)
    assert torch.allclose(bonus, torch.zeros(B), atol=1e-6)


# ---------------------------------------------------------------------------
# CompositeRewardFunction
# ---------------------------------------------------------------------------

def test_composite_reward_function_returns_shape():
    cfg = RewardShapingConfig(
        reward_clip=5.0,
        normalize_rewards=True,
        kl_penalty_coeff=0.1,
        length_penalty_coeff=0.0,
        diversity_bonus_coeff=0.0,
    )
    fn = CompositeRewardFunction(cfg)

    # Seed running stats so normalization doesn't divide by zero
    fn.update_stats(torch.randn(100))

    base_rewards = torch.randn(B)
    log_probs_policy = torch.randn(B, T)
    log_probs_ref = torch.randn(B, T)
    generated_ids = [torch.randint(0, 50, (T,)) for _ in range(B)]

    shaped = fn(base_rewards, log_probs_policy, log_probs_ref, generated_ids)
    assert shaped.shape == (B,)
