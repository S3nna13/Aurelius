"""Tests for src/simulation/agent_policy.py."""

from __future__ import annotations

import random

import pytest

from src.simulation.agent_policy import (
    AGENT_POLICY_REGISTRY,
    EpsilonGreedyPolicy,
    PolicyConfig,
    PolicyStats,
)


def _policy(**overrides) -> EpsilonGreedyPolicy:
    cfg = PolicyConfig(**overrides)
    return EpsilonGreedyPolicy(n_actions=4, config=cfg)


def test_config_defaults():
    c = PolicyConfig()
    assert c.learning_rate == 3e-4
    assert c.gamma == 0.99
    assert c.epsilon_start == 1.0
    assert c.epsilon_end == 0.01
    assert c.epsilon_decay == 0.995
    assert c.clip_ratio == 0.2


def test_epsilon_starts_at_epsilon_start():
    p = _policy(epsilon_start=0.7)
    assert p.epsilon == 0.7


def test_invalid_n_actions_raises():
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(n_actions=0, config=PolicyConfig())


def test_decay_epsilon_reduces():
    p = _policy(epsilon_start=1.0, epsilon_decay=0.5, epsilon_end=0.0)
    p.decay_epsilon()
    assert p.epsilon == 0.5


def test_decay_epsilon_clamped_to_end():
    p = _policy(epsilon_start=0.02, epsilon_decay=0.1, epsilon_end=0.01)
    p.decay_epsilon()
    assert p.epsilon == 0.01


def test_decay_epsilon_returns_value():
    p = _policy(epsilon_start=1.0, epsilon_decay=0.9, epsilon_end=0.0)
    eps = p.decay_epsilon()
    assert eps == p.epsilon


def test_select_action_argmax_on_zero_epsilon():
    p = _policy(epsilon_start=0.0)
    p.epsilon = 0.0
    assert p.select_action([0.1, 0.5, 0.2, 0.4]) == 1


def test_select_action_argmax_first_index_on_tie():
    p = _policy(epsilon_start=0.0)
    p.epsilon = 0.0
    assert p.select_action([0.5, 0.5, 0.5, 0.5]) == 0


def test_select_action_random_when_epsilon_one():
    p = _policy(epsilon_start=1.0)
    p.epsilon = 1.0
    random.seed(0)
    for _ in range(20):
        a = p.select_action([1.0, 0.0, 0.0, 0.0])
        assert 0 <= a < 4


def test_select_action_length_mismatch_raises():
    p = _policy()
    with pytest.raises(ValueError):
        p.select_action([0.0, 1.0])


def test_compute_returns_simple():
    p = _policy(gamma=0.99)
    returns = p.compute_returns([1.0, 0.0, 0.0])
    assert returns[2] == pytest.approx(0.0)
    assert returns[1] == pytest.approx(0.0)
    assert returns[0] == pytest.approx(1.0)


def test_compute_returns_discounted():
    p = _policy(gamma=0.5)
    returns = p.compute_returns([1.0, 1.0, 1.0])
    # G2=1, G1=1+0.5*1=1.5, G0=1+0.5*1.5=1.75
    assert returns == pytest.approx([1.75, 1.5, 1.0])


def test_compute_returns_override_gamma():
    p = _policy(gamma=0.99)
    returns = p.compute_returns([1.0, 1.0], gamma=0.0)
    assert returns == [1.0, 1.0]


def test_compute_returns_empty():
    p = _policy()
    assert p.compute_returns([]) == []


def test_compute_returns_zero_gamma():
    p = _policy(gamma=0.0)
    assert p.compute_returns([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


def test_compute_returns_length_preserved():
    p = _policy()
    assert len(p.compute_returns([0.1] * 10)) == 10


def test_td_error_formula_not_done():
    p = _policy(gamma=0.9)
    # r + gamma * next_q - current_q
    err = p.td_error(reward=1.0, next_q=2.0, current_q=0.5, done=False)
    assert err == pytest.approx(1.0 + 0.9 * 2.0 - 0.5)


def test_td_error_formula_done():
    p = _policy(gamma=0.9)
    err = p.td_error(reward=1.0, next_q=2.0, current_q=0.5, done=True)
    assert err == pytest.approx(1.0 - 0.5)


def test_td_error_zero_when_consistent():
    p = _policy(gamma=1.0)
    err = p.td_error(reward=0.0, next_q=1.0, current_q=1.0, done=False)
    assert err == pytest.approx(0.0)


def test_update_stats_returns_policy_stats():
    p = _policy()
    s = p.update_stats(episode=1, total_reward=5.0, steps=10)
    assert isinstance(s, PolicyStats)


def test_update_stats_values():
    p = _policy(epsilon_start=0.5)
    s = p.update_stats(episode=3, total_reward=42.0, steps=7)
    assert s.episode == 3
    assert s.total_reward == 42.0
    assert s.steps == 7
    assert s.epsilon == 0.5
    assert s.loss == 0.0


def test_policy_stats_is_frozen():
    s = PolicyStats(episode=0, total_reward=0.0, steps=0, epsilon=1.0)
    with pytest.raises(Exception):
        s.episode = 1  # type: ignore[misc]


def test_registry_has_default():
    assert "default" in AGENT_POLICY_REGISTRY
    assert AGENT_POLICY_REGISTRY["default"] is EpsilonGreedyPolicy


def test_decay_multiple_times_monotonic():
    p = _policy(epsilon_start=1.0, epsilon_decay=0.9, epsilon_end=0.0)
    prev = p.epsilon
    for _ in range(5):
        p.decay_epsilon()
        assert p.epsilon <= prev
        prev = p.epsilon


def test_select_action_all_negative_q():
    p = _policy(epsilon_start=0.0)
    p.epsilon = 0.0
    assert p.select_action([-5.0, -1.0, -3.0, -10.0]) == 1


def test_compute_returns_negative_rewards():
    p = _policy(gamma=1.0)
    returns = p.compute_returns([-1.0, -1.0, -1.0])
    assert returns == [-3.0, -2.0, -1.0]
