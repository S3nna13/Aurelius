"""Tests for src/training/reward_shaping.py"""
from __future__ import annotations

import pytest
import torch

from src.training.reward_shaping import (
    RewardShapingConfig,
    RewardShaper,
    EWMARewardNormalizer,
    normalize_rewards,
    normalize_rewards_zscore,
    normalize_rewards_minmax,
    normalize_rewards_rank,
    compute_discounted_returns,
    apply_potential_shaping,
    length_penalty,
    repetition_penalty,
)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------

def test_reward_shaping_config_defaults():
    cfg = RewardShapingConfig()
    assert cfg.normalize_method == "zscore"
    assert cfg.clip_range == 5.0
    assert cfg.gamma == 1.0
    assert cfg.use_potential_shaping is False
    assert cfg.ewma_alpha == 0.01
    assert cfg.penalty_coeff == 0.1


# ---------------------------------------------------------------------------
# 2. Z-score normalization
# ---------------------------------------------------------------------------

def test_normalize_zscore_mean_std():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    normalized = normalize_rewards_zscore(rewards)
    assert abs(normalized.mean().item()) < 1e-5
    assert abs(normalized.std().item() - 1.0) < 0.1


# ---------------------------------------------------------------------------
# 3. Min-max normalization
# ---------------------------------------------------------------------------

def test_normalize_minmax_range():
    rewards = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
    normalized = normalize_rewards_minmax(rewards)
    assert normalized.min().item() >= 0.0 - 1e-6
    assert normalized.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 4. Rank normalization
# ---------------------------------------------------------------------------

def test_normalize_rank_range():
    rewards = torch.tensor([10.0, -5.0, 3.0, 1.0, 0.0])
    normalized = normalize_rewards_rank(rewards)
    assert normalized.min().item() >= -1.0 - 1e-6
    assert normalized.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 5. "none" normalization unchanged
# ---------------------------------------------------------------------------

def test_normalize_none_unchanged():
    rewards = torch.tensor([3.0, -1.5, 0.0, 7.2])
    out = normalize_rewards(rewards, method="none")
    assert torch.allclose(out, rewards)


# ---------------------------------------------------------------------------
# 6. Discounted returns gamma=1
# ---------------------------------------------------------------------------

def test_discounted_returns_gamma1():
    # With gamma=1, G_t = sum of all rewards from t onward
    rewards = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
    returns = compute_discounted_returns(rewards, gamma=1.0)
    # G_0 = 1+2+3=6, G_1=2+3=5, G_2=3
    expected = torch.tensor([[6.0, 5.0, 3.0]])
    assert torch.allclose(returns, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Discounted returns gamma=0
# ---------------------------------------------------------------------------

def test_discounted_returns_gamma0():
    # With gamma=0, G_t = r_t (no future)
    rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    returns = compute_discounted_returns(rewards, gamma=0.0)
    assert torch.allclose(returns, rewards, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. Potential shaping shape
# ---------------------------------------------------------------------------

def test_potential_shaping_shape():
    B, T = 4, 6
    rewards = torch.randn(B, T)
    potentials = torch.randn(B, T + 1)
    shaped = apply_potential_shaping(rewards, potentials, gamma=0.99)
    assert shaped.shape == (B, T)


# ---------------------------------------------------------------------------
# 9. Length penalty: in range -> 0
# ---------------------------------------------------------------------------

def test_length_penalty_in_range():
    # 15 non-pad tokens per sequence, range [10, 200]
    token_ids = torch.ones(4, 15, dtype=torch.long)  # no padding
    penalties = length_penalty(token_ids, min_length=10, max_length=200, coeff=0.1)
    assert torch.all(penalties == 0.0)


# ---------------------------------------------------------------------------
# 10. Repetition penalty: no repeats -> 0
# ---------------------------------------------------------------------------

def test_repetition_penalty_unique():
    # All unique tokens -> all bigrams unique -> penalty=0
    B, S = 3, 20
    # Each sequence: consecutive integers (no token ID = 0, so pad won't be stripped)
    token_ids = torch.arange(1, S + 1).unsqueeze(0).expand(B, -1).clone()
    penalties = repetition_penalty(token_ids, window=20, coeff=0.1)
    assert torch.all(penalties < 1e-6)


# ---------------------------------------------------------------------------
# 11. EWMA normalizer updates
# ---------------------------------------------------------------------------

def test_ewma_normalizer_updates():
    norm = EWMARewardNormalizer(alpha=0.5)
    initial_mean = norm._mean  # 0.0

    # Feed large positive rewards
    rewards = torch.tensor([10.0, 10.0, 10.0, 10.0])
    norm.update(rewards)

    # Mean should have moved toward 10
    assert norm._mean > initial_mean


# ---------------------------------------------------------------------------
# 12. RewardShaper output shape
# ---------------------------------------------------------------------------

def test_reward_shaper_shape_output():
    cfg = RewardShapingConfig(normalize_method="zscore", clip_range=5.0)
    shaper = RewardShaper(cfg)

    B = 8
    base_rewards = torch.randn(B)
    token_ids = torch.randint(1, 50, (B, 30))  # no padding

    shaped = shaper.shape(base_rewards, token_ids=token_ids)
    assert shaped.shape == (B,)
