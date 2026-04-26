"""Tests for src/simulation/rollout_buffer.py."""

from __future__ import annotations

import random

import pytest

from src.simulation.rollout_buffer import (
    ROLLOUT_BUFFER_REGISTRY,
    Experience,
    RolloutBuffer,
)


def _exp(reward: float = 0.0, done: bool = False, action: int = 0) -> Experience:
    return Experience(
        state=[0.0, 0.0],
        action=action,
        reward=reward,
        next_state=[0.0, 0.0],
        done=done,
    )


def test_experience_is_frozen():
    e = _exp()
    with pytest.raises(Exception):
        e.action = 5  # type: ignore[misc]


def test_buffer_starts_empty():
    b = RolloutBuffer()
    assert len(b) == 0


def test_default_maxlen_is_10000():
    b = RolloutBuffer()
    assert b.maxlen == 10000


def test_push_increments_len():
    b = RolloutBuffer(maxlen=10)
    b.push(_exp())
    assert len(b) == 1
    b.push(_exp())
    assert len(b) == 2


def test_invalid_maxlen_raises():
    with pytest.raises(ValueError):
        RolloutBuffer(maxlen=0)


def test_maxlen_enforced():
    b = RolloutBuffer(maxlen=3)
    for i in range(5):
        b.push(_exp(reward=float(i)))
    assert len(b) == 3


def test_clear_empties_buffer():
    b = RolloutBuffer(maxlen=5)
    b.push(_exp())
    b.push(_exp())
    b.clear()
    assert len(b) == 0


def test_sample_returns_correct_size():
    b = RolloutBuffer()
    for i in range(10):
        b.push(_exp(reward=float(i)))
    sample = b.sample(4)
    assert len(sample) == 4


def test_sample_returns_list_of_experiences():
    b = RolloutBuffer()
    for _ in range(5):
        b.push(_exp())
    sample = b.sample(3)
    assert all(isinstance(e, Experience) for e in sample)


def test_sample_raises_value_error_when_insufficient():
    b = RolloutBuffer()
    b.push(_exp())
    with pytest.raises(ValueError):
        b.sample(5)


def test_sample_raises_on_nonpositive_batch():
    b = RolloutBuffer()
    b.push(_exp())
    with pytest.raises(ValueError):
        b.sample(0)


def test_can_sample_true_when_enough():
    b = RolloutBuffer()
    for _ in range(5):
        b.push(_exp())
    assert b.can_sample(5) is True
    assert b.can_sample(3) is True


def test_can_sample_false_when_insufficient():
    b = RolloutBuffer()
    b.push(_exp())
    assert b.can_sample(2) is False


def test_can_sample_false_for_empty():
    b = RolloutBuffer()
    assert b.can_sample(1) is False


def test_can_sample_false_for_zero():
    b = RolloutBuffer()
    b.push(_exp())
    assert b.can_sample(0) is False


def test_compute_episode_returns_discounted():
    b = RolloutBuffer()
    experiences = [_exp(reward=1.0), _exp(reward=1.0), _exp(reward=1.0)]
    returns = b.compute_episode_returns(experiences, gamma=0.5)
    assert returns == pytest.approx([1.75, 1.5, 1.0])


def test_compute_episode_returns_default_gamma():
    b = RolloutBuffer()
    experiences = [_exp(reward=1.0), _exp(reward=0.0)]
    returns = b.compute_episode_returns(experiences)
    # gamma default 0.99: G1=0, G0=1+0.99*0=1
    assert returns[1] == pytest.approx(0.0)
    assert returns[0] == pytest.approx(1.0)


def test_compute_episode_returns_empty():
    b = RolloutBuffer()
    assert b.compute_episode_returns([]) == []


def test_compute_episode_returns_zero_gamma():
    b = RolloutBuffer()
    experiences = [_exp(reward=1.0), _exp(reward=2.0)]
    assert b.compute_episode_returns(experiences, gamma=0.0) == [1.0, 2.0]


def test_mean_reward_empty():
    b = RolloutBuffer()
    assert b.mean_reward() == 0.0


def test_mean_reward_computed():
    b = RolloutBuffer()
    for r in [1.0, 2.0, 3.0]:
        b.push(_exp(reward=r))
    assert b.mean_reward() == pytest.approx(2.0)


def test_total_episodes_counts_dones():
    b = RolloutBuffer()
    b.push(_exp(done=False))
    b.push(_exp(done=True))
    b.push(_exp(done=False))
    b.push(_exp(done=True))
    assert b.total_episodes() == 2


def test_total_episodes_none_done():
    b = RolloutBuffer()
    for _ in range(3):
        b.push(_exp(done=False))
    assert b.total_episodes() == 0


def test_total_episodes_empty():
    b = RolloutBuffer()
    assert b.total_episodes() == 0


def test_as_arrays_keys():
    b = RolloutBuffer()
    b.push(_exp())
    out = b.as_arrays()
    assert set(out.keys()) == {
        "states",
        "actions",
        "rewards",
        "next_states",
        "dones",
    }


def test_as_arrays_values():
    b = RolloutBuffer()
    b.push(
        Experience(
            state=[1.0, 2.0],
            action=3,
            reward=0.5,
            next_state=[1.1, 2.1],
            done=True,
        )
    )
    out = b.as_arrays()
    assert out["states"] == [[1.0, 2.0]]
    assert out["actions"] == [3]
    assert out["rewards"] == [0.5]
    assert out["next_states"] == [[1.1, 2.1]]
    assert out["dones"] == [True]


def test_as_arrays_empty():
    b = RolloutBuffer()
    out = b.as_arrays()
    for v in out.values():
        assert v == []


def test_sample_random_with_seed():
    b = RolloutBuffer()
    for i in range(20):
        b.push(_exp(reward=float(i)))
    random.seed(123)
    s1 = b.sample(5)
    random.seed(123)
    s2 = b.sample(5)
    assert [e.reward for e in s1] == [e.reward for e in s2]


def test_registry_has_default():
    assert "default" in ROLLOUT_BUFFER_REGISTRY
    assert ROLLOUT_BUFFER_REGISTRY["default"] is RolloutBuffer


def test_push_then_clear_then_push():
    b = RolloutBuffer(maxlen=5)
    b.push(_exp())
    b.clear()
    b.push(_exp(reward=9.0))
    assert len(b) == 1
    assert b.mean_reward() == 9.0
