"""Tests for src/simulation/agent_harness.py — ~50 tests."""

from __future__ import annotations

import pytest

from src.simulation.agent_harness import (
    AGENT_HARNESS,
    AgentHarness,
    Trajectory,
)
from src.simulation.environment import EnvAction, EnvState, GridWorldEnv

# ---------------------------------------------------------------------------
# Trajectory dataclass
# ---------------------------------------------------------------------------


class TestTrajectory:
    def _make_traj(self, env_name="gridworld", steps=None, total_reward=0.0, success=False):
        return Trajectory(
            env_name=env_name,
            steps=steps or [],
            total_reward=total_reward,
            success=success,
        )

    def test_env_name_stored(self):
        t = self._make_traj(env_name="myenv")
        assert t.env_name == "myenv"

    def test_steps_stored(self):
        t = self._make_traj(steps=[])
        assert t.steps == []

    def test_total_reward_stored(self):
        t = self._make_traj(total_reward=3.5)
        assert t.total_reward == pytest.approx(3.5)

    def test_success_stored(self):
        t = self._make_traj(success=True)
        assert t.success is True

    def test_success_false_default(self):
        t = self._make_traj()
        assert t.success is False


# ---------------------------------------------------------------------------
# AgentHarness construction
# ---------------------------------------------------------------------------


class TestAgentHarnessInit:
    def test_default_env_is_gridworld(self):
        harness = AgentHarness()
        assert isinstance(harness.env, GridWorldEnv)

    def test_custom_env_stored(self):
        env = GridWorldEnv(width=3, height=3, goal=(2, 2))
        harness = AgentHarness(env=env)
        assert harness.env is env

    def test_agent_harness_singleton_exists(self):
        assert AGENT_HARNESS is not None

    def test_agent_harness_is_instance(self):
        assert isinstance(AGENT_HARNESS, AgentHarness)


# ---------------------------------------------------------------------------
# run_episode
# ---------------------------------------------------------------------------


class TestRunEpisode:
    def test_returns_trajectory(self):
        harness = AgentHarness()
        traj = harness.run_episode(AgentHarness.random_policy, max_steps=10)
        assert isinstance(traj, Trajectory)

    def test_trajectory_env_name(self):
        harness = AgentHarness()
        traj = harness.run_episode(AgentHarness.random_policy, max_steps=10)
        assert traj.env_name == "gridworld"

    def test_step_count_at_least_one(self):
        harness = AgentHarness()
        traj = harness.run_episode(AgentHarness.random_policy, max_steps=10)
        assert len(traj.steps) >= 1

    def test_max_steps_respected(self):
        harness = AgentHarness()
        traj = harness.run_episode(AgentHarness.random_policy, max_steps=5)
        assert len(traj.steps) <= 5

    def test_total_reward_is_sum(self):
        harness = AgentHarness()
        traj = harness.run_episode(AgentHarness.random_policy, max_steps=10)
        expected = sum(s.next_state.reward for s in traj.steps)
        assert traj.total_reward == pytest.approx(expected)

    def test_success_is_bool(self):
        harness = AgentHarness()
        traj = harness.run_episode(AgentHarness.random_policy, max_steps=10)
        assert isinstance(traj.success, bool)

    def test_steps_are_env_steps(self):
        from src.simulation.environment import EnvStep

        harness = AgentHarness()
        traj = harness.run_episode(AgentHarness.random_policy, max_steps=5)
        for s in traj.steps:
            assert isinstance(s, EnvStep)

    def test_step_numbers_sequential(self):
        harness = AgentHarness()
        traj = harness.run_episode(AgentHarness.random_policy, max_steps=10)
        for i, s in enumerate(traj.steps, start=1):
            assert s.step_number == i


# ---------------------------------------------------------------------------
# greedy_policy
# ---------------------------------------------------------------------------


class TestGreedyPolicy:
    def test_reaches_goal_in_few_steps(self):
        env = GridWorldEnv(width=5, height=5, goal=(4, 4))
        harness = AgentHarness(env=env)
        policy = AgentHarness.greedy_policy((4, 4))
        traj = harness.run_episode(policy, max_steps=20)
        assert traj.success is True

    def test_reaches_goal_in_at_most_eight_steps(self):
        env = GridWorldEnv(width=5, height=5, goal=(4, 4))
        harness = AgentHarness(env=env)
        policy = AgentHarness.greedy_policy((4, 4))
        traj = harness.run_episode(policy, max_steps=20)
        assert len(traj.steps) <= 8

    def test_greedy_policy_callable(self):
        policy = AgentHarness.greedy_policy((4, 4))
        assert callable(policy)

    def test_greedy_policy_returns_action(self):
        policy = AgentHarness.greedy_policy((4, 4))
        state = EnvState(obs={"x": 0, "y": 0, "goal": (4, 4)})
        actions = [EnvAction(a) for a in ("up", "down", "left", "right")]
        result = policy(state, actions)
        assert isinstance(result, EnvAction)

    def test_greedy_moves_toward_goal_x(self):
        """From (0,0) with goal (4,4), greedy should move right or up."""
        policy = AgentHarness.greedy_policy((4, 4))
        state = EnvState(obs={"x": 0, "y": 0, "goal": (4, 4)})
        actions = [EnvAction(a) for a in ("up", "down", "left", "right")]
        action = policy(state, actions)
        assert action.action_id in {"right", "up"}

    def test_greedy_small_grid(self):
        env = GridWorldEnv(width=3, height=3, goal=(2, 2))
        harness = AgentHarness(env=env)
        policy = AgentHarness.greedy_policy((2, 2))
        traj = harness.run_episode(policy, max_steps=10)
        assert traj.success is True


# ---------------------------------------------------------------------------
# run_n_episodes
# ---------------------------------------------------------------------------


class TestRunNEpisodes:
    def test_returns_n_trajectories(self):
        harness = AgentHarness()
        trajs = harness.run_n_episodes(AgentHarness.random_policy, n=5, max_steps=20)
        assert len(trajs) == 5

    def test_returns_list(self):
        harness = AgentHarness()
        trajs = harness.run_n_episodes(AgentHarness.random_policy, n=3, max_steps=10)
        assert isinstance(trajs, list)

    def test_each_element_is_trajectory(self):
        harness = AgentHarness()
        trajs = harness.run_n_episodes(AgentHarness.random_policy, n=3, max_steps=10)
        for t in trajs:
            assert isinstance(t, Trajectory)

    def test_zero_episodes(self):
        harness = AgentHarness()
        trajs = harness.run_n_episodes(AgentHarness.random_policy, n=0)
        assert trajs == []


# ---------------------------------------------------------------------------
# success_rate
# ---------------------------------------------------------------------------


class TestSuccessRate:
    def _traj(self, success: bool) -> Trajectory:
        return Trajectory(env_name="gridworld", steps=[], total_reward=0.0, success=success)

    def test_all_success(self):
        trajs = [self._traj(True), self._traj(True)]
        assert AgentHarness.success_rate(trajs) == pytest.approx(1.0)

    def test_none_success(self):
        trajs = [self._traj(False), self._traj(False)]
        assert AgentHarness.success_rate(trajs) == pytest.approx(0.0)

    def test_empty_list_returns_zero(self):
        assert AgentHarness.success_rate([]) == pytest.approx(0.0)

    def test_mixed(self):
        trajs = [self._traj(True), self._traj(False), self._traj(True), self._traj(False)]
        assert AgentHarness.success_rate(trajs) == pytest.approx(0.5)

    def test_single_success(self):
        assert AgentHarness.success_rate([self._traj(True)]) == pytest.approx(1.0)

    def test_single_failure(self):
        assert AgentHarness.success_rate([self._traj(False)]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# random_policy
# ---------------------------------------------------------------------------


class TestRandomPolicy:
    def test_returns_env_action(self):
        state = EnvState(obs={"x": 0, "y": 0, "goal": (4, 4)})
        actions = [EnvAction(a) for a in ("up", "down", "left", "right")]
        result = AgentHarness.random_policy(state, actions)
        assert isinstance(result, EnvAction)

    def test_returns_one_of_valid_actions(self):
        state = EnvState(obs={"x": 0, "y": 0, "goal": (4, 4)})
        actions = [EnvAction(a) for a in ("up", "down", "left", "right")]
        result = AgentHarness.random_policy(state, actions)
        assert result in actions
