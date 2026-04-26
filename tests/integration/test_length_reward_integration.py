"""Integration test for LengthReward — length-conditional reward for efficient thinking chains.

Verifies:
- End-to-end batch computation across mixed rollouts (correct/incorrect, short/long).
- statistics() sums correctly: n_correct + n_incorrect == total.
- compute_tensor() returns expected shape and plausible values.
- ALIGNMENT_REGISTRY["length_reward"] is wired to LengthReward.
"""

import pytest
import torch

from src.alignment import ALIGNMENT_REGISTRY
from src.alignment.length_reward import LengthReward, LengthRewardConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lr():
    cfg = LengthRewardConfig(
        token_budget=2048,
        length_penalty_weight=0.1,
        length_bonus_weight=0.05,
        correct_base_reward=1.0,
        incorrect_base_reward=0.0,
        min_length_ratio=0.1,
        max_length_ratio=3.0,
    )
    return LengthReward(cfg)


# Five rollouts: mix of correct/incorrect, short/at-budget/long/too-short
CORRECTNESS = [True, False, True, True, False]
TOKEN_COUNTS = [
    1024,  # correct, within budget → bonus
    3000,  # incorrect, over budget → incorrect_base
    2048,  # correct, exact budget → correct_base
    4000,  # correct, over budget → penalty
    200,  # incorrect, short → incorrect_base
]


# ---------------------------------------------------------------------------
# Integration test 1: batch rewards have expected types and count
# ---------------------------------------------------------------------------


def test_batch_rewards_types_and_count(lr):
    rewards = lr.compute_batch(CORRECTNESS, TOKEN_COUNTS)
    assert isinstance(rewards, list), "compute_batch should return a list"
    assert len(rewards) == 5, f"Expected 5 rewards, got {len(rewards)}"
    for i, r in enumerate(rewards):
        assert isinstance(r, float), f"Reward {i} should be float, got {type(r)}"


# ---------------------------------------------------------------------------
# Integration test 2: individual reward values match expectations
# ---------------------------------------------------------------------------


def test_batch_reward_values(lr):
    rewards = lr.compute_batch(CORRECTNESS, TOKEN_COUNTS)
    cfg = lr.config

    # rollout 0: correct, 1024 < 2048 → bonus
    assert rewards[0] > cfg.correct_base_reward, (
        f"Rollout 0 (correct, within budget) should have bonus. Got {rewards[0]}"
    )

    # rollout 1: incorrect → incorrect_base
    assert rewards[1] == pytest.approx(cfg.incorrect_base_reward), (
        f"Rollout 1 (incorrect) should be {cfg.incorrect_base_reward}. Got {rewards[1]}"
    )

    # rollout 2: correct, exact budget → correct_base
    assert rewards[2] == pytest.approx(cfg.correct_base_reward), (
        f"Rollout 2 (correct, exact budget) should be {cfg.correct_base_reward}. Got {rewards[2]}"
    )

    # rollout 3: correct, 4000 > 2048 → penalty
    assert rewards[3] < cfg.correct_base_reward, (
        f"Rollout 3 (correct, over budget) should have penalty. Got {rewards[3]}"
    )

    # rollout 4: incorrect → incorrect_base
    assert rewards[4] == pytest.approx(cfg.incorrect_base_reward), (
        f"Rollout 4 (incorrect) should be {cfg.incorrect_base_reward}. Got {rewards[4]}"
    )


# ---------------------------------------------------------------------------
# Integration test 3: statistics sums — n_correct + n_incorrect == total
# ---------------------------------------------------------------------------


def test_statistics_sum(lr):
    stats = lr.statistics(CORRECTNESS, TOKEN_COUNTS)
    total = len(CORRECTNESS)
    n_correct = stats["n_correct"]
    n_incorrect = total - n_correct
    assert n_correct + n_incorrect == total, (
        f"n_correct ({n_correct}) + n_incorrect ({n_incorrect}) != total ({total})"
    )
    # Verify the expected breakdown: 3 correct, 2 incorrect
    assert n_correct == 3, f"Expected n_correct=3, got {n_correct}"
    assert n_incorrect == 2, f"Expected n_incorrect=2, got {n_incorrect}"


# ---------------------------------------------------------------------------
# Integration test 4: statistics n_penalized and n_bonus are consistent
# ---------------------------------------------------------------------------


def test_statistics_penalized_bonus_consistent(lr):
    stats = lr.statistics(CORRECTNESS, TOKEN_COUNTS)

    # Only rollout 3 is correct and over budget → n_penalized=1
    assert stats["n_penalized"] == 1, f"Expected n_penalized=1, got {stats['n_penalized']}"

    # Rollouts 0 (within budget, above min) and 2 (exact budget, above min) → n_bonus=2
    assert stats["n_bonus"] == 2, f"Expected n_bonus=2, got {stats['n_bonus']}"

    # n_penalized + n_bonus <= n_correct (correct rollouts partitioned by budget)
    assert stats["n_penalized"] + stats["n_bonus"] <= stats["n_correct"]


# ---------------------------------------------------------------------------
# Integration test 5: statistics mean_reward is in plausible range
# ---------------------------------------------------------------------------


def test_statistics_mean_reward_range(lr):
    stats = lr.statistics(CORRECTNESS, TOKEN_COUNTS)
    cfg = lr.config
    # Mean reward should be between incorrect_base and max_possible_reward
    max_possible = cfg.correct_base_reward + cfg.length_bonus_weight
    assert cfg.incorrect_base_reward <= stats["mean_reward"] <= max_possible + 1e-6, (
        f"mean_reward {stats['mean_reward']} out of expected range "
        f"[{cfg.incorrect_base_reward}, {max_possible}]"
    )


# ---------------------------------------------------------------------------
# Integration test 6: compute_tensor output shape and type
# ---------------------------------------------------------------------------


def test_tensor_output_shape_and_type(lr):
    correctness_t = torch.tensor(CORRECTNESS)
    token_counts_t = torch.tensor(TOKEN_COUNTS)
    rewards_t = lr.compute_tensor(correctness_t, token_counts_t)
    assert rewards_t.shape == (5,), f"Expected shape (5,), got {rewards_t.shape}"
    assert rewards_t.dtype == torch.float32, f"Expected float32, got {rewards_t.dtype}"


# ---------------------------------------------------------------------------
# Integration test 7: compute_tensor values match compute_batch
# ---------------------------------------------------------------------------


def test_tensor_matches_batch(lr):
    rewards_list = lr.compute_batch(CORRECTNESS, TOKEN_COUNTS)
    correctness_t = torch.tensor(CORRECTNESS)
    token_counts_t = torch.tensor(TOKEN_COUNTS)
    rewards_t = lr.compute_tensor(correctness_t, token_counts_t)

    for i, (r_list, r_tensor) in enumerate(zip(rewards_list, rewards_t.tolist())):
        assert abs(r_list - r_tensor) < 1e-5, (
            f"Rollout {i}: batch={r_list}, tensor={r_tensor} — mismatch"
        )


# ---------------------------------------------------------------------------
# Integration test 8: ALIGNMENT_REGISTRY["length_reward"] is wired correctly
# ---------------------------------------------------------------------------


def test_registry_wired():
    assert "length_reward" in ALIGNMENT_REGISTRY, "ALIGNMENT_REGISTRY missing 'length_reward' key"
    assert ALIGNMENT_REGISTRY["length_reward"] is LengthReward, (
        f"Registry entry should be LengthReward class, got {ALIGNMENT_REGISTRY['length_reward']}"
    )


# ---------------------------------------------------------------------------
# Integration test 9: Registry entry is instantiable with default config
# ---------------------------------------------------------------------------


def test_registry_instantiable():
    cls = ALIGNMENT_REGISTRY["length_reward"]
    instance = cls()
    assert isinstance(instance, LengthReward)
    reward = instance.compute(is_correct=True, n_tokens=2000)
    assert isinstance(reward, float)
