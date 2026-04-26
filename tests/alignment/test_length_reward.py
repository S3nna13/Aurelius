"""Unit tests for LengthReward (src/alignment/length_reward.py).

Tests: 15 total covering config defaults, scalar compute, batch, tensor, and statistics.
"""

import pytest
import torch

from src.alignment.length_reward import LengthReward, LengthRewardConfig

# ---------------------------------------------------------------------------
# Test 1: Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = LengthRewardConfig()
    assert cfg.token_budget == 4096
    assert cfg.length_penalty_weight == 0.1
    assert cfg.length_bonus_weight == 0.05
    assert cfg.correct_base_reward == 1.0
    assert cfg.incorrect_base_reward == 0.0
    assert cfg.min_length_ratio == 0.1
    assert cfg.max_length_ratio == 3.0


# ---------------------------------------------------------------------------
# Test 2: Correct within budget → reward > correct_base (bonus applied)
# ---------------------------------------------------------------------------


def test_correct_within_budget():
    lr = LengthReward()
    # 2048 tokens, budget 4096 — clearly within budget and above min ratio
    reward = lr.compute(is_correct=True, n_tokens=2048)
    assert reward > lr.config.correct_base_reward, (
        f"Expected bonus reward > {lr.config.correct_base_reward}, got {reward}"
    )


# ---------------------------------------------------------------------------
# Test 3: Correct over budget → reward < correct_base (penalty applied)
# ---------------------------------------------------------------------------


def test_correct_over_budget():
    lr = LengthReward()
    # 6000 tokens, budget 4096 — over budget
    reward = lr.compute(is_correct=True, n_tokens=6000)
    assert reward < lr.config.correct_base_reward, (
        f"Expected penalty reward < {lr.config.correct_base_reward}, got {reward}"
    )


# ---------------------------------------------------------------------------
# Test 4: Correct at exact budget → reward == correct_base
# ---------------------------------------------------------------------------


def test_correct_exact_budget():
    lr = LengthReward()
    reward = lr.compute(is_correct=True, n_tokens=lr.config.token_budget)
    assert reward == pytest.approx(lr.config.correct_base_reward), (
        f"Expected {lr.config.correct_base_reward} at exact budget, got {reward}"
    )


# ---------------------------------------------------------------------------
# Test 5: Incorrect rollout → incorrect_base_reward regardless of length
# ---------------------------------------------------------------------------


def test_incorrect():
    lr = LengthReward()
    # Short incorrect
    r_short = lr.compute(is_correct=False, n_tokens=500)
    # Long incorrect
    r_long = lr.compute(is_correct=False, n_tokens=10000)
    assert r_short == pytest.approx(lr.config.incorrect_base_reward)
    assert r_long == pytest.approx(lr.config.incorrect_base_reward)


# ---------------------------------------------------------------------------
# Test 6: Too short (below min_length_ratio * budget) → treated as incorrect
# ---------------------------------------------------------------------------


def test_too_short_penalized():
    lr = LengthReward()
    # min_length_ratio=0.1, budget=4096 → threshold=409.6
    # Use 100 tokens — far below threshold
    reward = lr.compute(is_correct=True, n_tokens=100)
    assert reward == pytest.approx(lr.config.incorrect_base_reward), (
        f"Too-short correct should receive incorrect_base, got {reward}"
    )


# ---------------------------------------------------------------------------
# Test 7: Length penalty scales with excess tokens
# ---------------------------------------------------------------------------


def test_length_penalty_scales():
    lr = LengthReward()
    # 2× budget vs 1.1× budget — more excess should mean more penalty
    r_2x = lr.compute(is_correct=True, n_tokens=lr.config.token_budget * 2)
    r_1_1x = lr.compute(is_correct=True, n_tokens=int(lr.config.token_budget * 1.1))
    assert r_2x < r_1_1x, f"2× over budget ({r_2x}) should be penalized more than 1.1× ({r_1_1x})"


# ---------------------------------------------------------------------------
# Test 8: Length bonus scales — 50% of budget gets more bonus than 90%
# ---------------------------------------------------------------------------


def test_length_bonus_scales():
    lr = LengthReward()
    r_50pct = lr.compute(is_correct=True, n_tokens=int(lr.config.token_budget * 0.5))
    r_90pct = lr.compute(is_correct=True, n_tokens=int(lr.config.token_budget * 0.9))
    assert r_50pct > r_90pct, (
        f"50% of budget ({r_50pct}) should have more bonus than 90% ({r_90pct})"
    )


# ---------------------------------------------------------------------------
# Test 9: Max length cap — very long sequence capped at max_length_ratio
# ---------------------------------------------------------------------------


def test_max_length_cap():
    lr = LengthReward()
    # Extremely long: 100× budget
    r_extreme = lr.compute(is_correct=True, n_tokens=lr.config.token_budget * 100)
    # At max_length_ratio=3.0: factor capped at 2.0, penalty = 0.1 * 2.0 = 0.2
    expected_max_penalty = lr.config.length_penalty_weight * (lr.config.max_length_ratio - 1.0)
    expected_floor = lr.config.correct_base_reward - expected_max_penalty
    assert r_extreme == pytest.approx(expected_floor), (
        f"Expected capped reward {expected_floor}, got {r_extreme}"
    )


# ---------------------------------------------------------------------------
# Test 10: compute_batch returns list of correct length
# ---------------------------------------------------------------------------


def test_compute_batch_length():
    lr = LengthReward()
    correctness = [True, False, True]
    token_counts = [2000, 5000, 4096]
    results = lr.compute_batch(correctness, token_counts)
    assert isinstance(results, list), "compute_batch should return a list"
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all(isinstance(r, float) for r in results), "All rewards should be floats"


# ---------------------------------------------------------------------------
# Test 11: compute_tensor shape — [B] input → [B] output
# ---------------------------------------------------------------------------


def test_compute_tensor_shape():
    lr = LengthReward()
    B = 6
    correctness = torch.tensor([True, False, True, True, False, True])
    token_counts = torch.tensor([1000, 5000, 4096, 6000, 100, 2000])
    rewards = lr.compute_tensor(correctness, token_counts)
    assert rewards.shape == (B,), f"Expected shape ({B},), got {rewards.shape}"
    assert rewards.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 12: statistics returns correct keys
# ---------------------------------------------------------------------------


def test_statistics_keys():
    lr = LengthReward()
    correctness = [True, False, True]
    token_counts = [2000, 5000, 4096]
    stats = lr.statistics(correctness, token_counts)
    required_keys = {"mean_reward", "mean_tokens", "n_correct", "n_penalized", "n_bonus"}
    assert required_keys == set(stats.keys()), f"Missing/extra keys. Got: {set(stats.keys())}"


# ---------------------------------------------------------------------------
# Test 13: statistics n_correct counts correctly
# ---------------------------------------------------------------------------


def test_statistics_n_correct():
    lr = LengthReward()
    correctness = [True, False, True, True, False]
    token_counts = [2000, 5000, 4096, 6000, 100]
    stats = lr.statistics(correctness, token_counts)
    assert stats["n_correct"] == 3, f"Expected n_correct=3, got {stats['n_correct']}"


# ---------------------------------------------------------------------------
# Test 14: statistics n_penalized counts over-budget correct rollouts
# ---------------------------------------------------------------------------


def test_statistics_penalized():
    lr = LengthReward()
    budget = lr.config.token_budget
    # 2 correct over-budget, 1 correct within-budget, 1 incorrect over-budget
    correctness = [True, True, True, False]
    token_counts = [budget + 1000, budget + 2000, budget - 500, budget + 500]
    stats = lr.statistics(correctness, token_counts)
    assert stats["n_penalized"] == 2, f"Expected n_penalized=2, got {stats['n_penalized']}"


# ---------------------------------------------------------------------------
# Test 15: statistics n_bonus counts within-budget correct (not too short)
# ---------------------------------------------------------------------------


def test_statistics_bonus():
    lr = LengthReward()
    budget = lr.config.token_budget
    min_tokens = int(budget * lr.config.min_length_ratio)
    # 2 correct within budget above min, 1 correct too short, 1 correct over budget
    correctness = [True, True, True, True]
    token_counts = [
        min_tokens + 100,  # bonus
        budget - 100,  # bonus
        min_tokens - 10,  # too short, no bonus
        budget + 100,  # penalized, no bonus
    ]
    stats = lr.statistics(correctness, token_counts)
    assert stats["n_bonus"] == 2, f"Expected n_bonus=2, got {stats['n_bonus']}"
