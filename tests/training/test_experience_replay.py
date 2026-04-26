"""Tests for src/training/experience_replay.py."""

from __future__ import annotations

import math
import random

import pytest

from src.training.experience_replay import (
    Experience,
    OnlinePPOBuffer,
    ReplayBuffer,
    RewardNormalizer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_exp(reward: float = 1.0, step: int = 0) -> Experience:
    return Experience(
        prompt_ids=[1, 2, 3],
        completion_ids=[4, 5],
        reward=reward,
        log_prob=-0.5,
        step=step,
    )


# ---------------------------------------------------------------------------
# ReplayBuffer tests
# ---------------------------------------------------------------------------


def test_replay_buffer_add_and_len():
    buf = ReplayBuffer(capacity=100, prioritized=False)
    for i in range(5):
        buf.add(make_exp(reward=float(i)))
    assert len(buf) == 5


def test_replay_buffer_capacity_overflow():
    capacity = 10
    buf = ReplayBuffer(capacity=capacity, prioritized=False)
    for i in range(capacity + 3):
        buf.add(make_exp(reward=float(i)))
    assert len(buf) == capacity


def test_replay_buffer_uniform_sample():
    buf = ReplayBuffer(capacity=50, prioritized=False)
    for i in range(20):
        buf.add(make_exp(reward=float(i)))
    batch_size = 8
    experiences, weights = buf.sample(batch_size)
    assert len(experiences) == batch_size
    assert len(weights) == batch_size
    assert all(w == 1.0 for w in weights)


def test_replay_buffer_prioritized_sample():
    """High-reward experiences should be sampled more often with prioritized=True."""
    random.seed(42)
    buf = ReplayBuffer(capacity=1000, prioritized=True, alpha=0.6, beta=0.4)
    # Add 1 high-reward experience and 99 low-reward experiences
    buf.add(make_exp(reward=100.0, step=0))
    for i in range(99):
        buf.add(make_exp(reward=0.01, step=i + 1))

    n_samples = 5000
    high_count = 0
    for _ in range(n_samples):
        exps, _ = buf.sample(1)
        if exps[0].reward == 100.0:
            high_count += 1

    # With priority ∝ |reward|^0.6, the high-reward item should appear far more
    # than its 1% base rate
    fraction = high_count / n_samples
    assert fraction > 0.10, f"High-reward fraction too low: {fraction:.3f}"


def test_replay_buffer_importance_weights_max_1():
    buf = ReplayBuffer(capacity=100, prioritized=True, alpha=0.6, beta=0.4)
    for i in range(20):
        buf.add(make_exp(reward=float(i + 1)))
    _, weights = buf.sample(10)
    assert all(w <= 1.0 + 1e-6 for w in weights), f"Max weight exceeded 1: {max(weights)}"


def test_replay_buffer_stats():
    buf = ReplayBuffer(capacity=100, prioritized=False)
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    for r in rewards:
        buf.add(make_exp(reward=r))
    s = buf.stats()
    assert s["size"] == 5
    assert s["capacity"] == 100
    assert math.isfinite(s["mean_reward"])
    assert math.isfinite(s["max_reward"])
    assert math.isfinite(s["min_reward"])
    assert abs(s["mean_reward"] - 3.0) < 1e-6
    assert s["max_reward"] == 5.0
    assert s["min_reward"] == 1.0


# ---------------------------------------------------------------------------
# OnlinePPOBuffer tests
# ---------------------------------------------------------------------------


def test_ppo_buffer_add_and_compute_advantages():
    buf = OnlinePPOBuffer(gamma=1.0, lam=0.95)
    for i in range(3):
        buf.add(make_exp(reward=1.0, step=i), value=0.5)
    advantages = buf.compute_advantages()
    assert len(advantages) == 3


def test_ppo_buffer_advantage_reduces_with_value():
    """Higher value estimate should yield a lower advantage."""
    buf_low = OnlinePPOBuffer(gamma=1.0, lam=0.95)
    buf_high = OnlinePPOBuffer(gamma=1.0, lam=0.95)

    reward = 2.0
    buf_low.add(make_exp(reward=reward), value=0.1)
    buf_high.add(make_exp(reward=reward), value=1.5)

    adv_low = buf_low.compute_advantages()[0]
    adv_high = buf_high.compute_advantages()[0]

    assert adv_low > adv_high, f"Expected adv_low ({adv_low:.4f}) > adv_high ({adv_high:.4f})"


def test_ppo_buffer_clear_empties():
    buf = OnlinePPOBuffer()
    for i in range(5):
        buf.add(make_exp(reward=1.0, step=i), value=0.5)
    buf.clear()
    assert len(buf.experiences) == 0
    assert len(buf.values) == 0


# ---------------------------------------------------------------------------
# RewardNormalizer tests
# ---------------------------------------------------------------------------


def test_reward_normalizer_welford():
    """After 100 standard-normal rewards, normalized samples should have approx mean≈0, std≈1."""
    rng = random.Random(0)
    normalizer = RewardNormalizer(eps=1e-8)

    raw_rewards = [rng.gauss(0.0, 1.0) for _ in range(100)]

    # Feed all rewards into the normalizer first
    for r in raw_rewards:
        normalizer.update(r)

    # Normalize them all
    normalized = [normalizer.normalize(r) for r in raw_rewards]

    mean_norm = sum(normalized) / len(normalized)
    std_norm = math.sqrt(sum((x - mean_norm) ** 2 for x in normalized) / (len(normalized) - 1))

    assert abs(mean_norm) < 0.1, f"Normalized mean too far from 0: {mean_norm:.4f}"
    assert abs(std_norm - 1.0) < 0.1, f"Normalized std too far from 1: {std_norm:.4f}"


def test_reward_normalizer_single_sample():
    """With n=1, normalize should return the raw reward unchanged."""
    normalizer = RewardNormalizer(eps=1e-8)
    normalizer.update(3.14)
    assert normalizer.n == 1
    result = normalizer.normalize(3.14)
    assert result == pytest.approx(3.14), f"Expected raw reward, got {result}"
