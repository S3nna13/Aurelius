"""Tests for src/simulation/environment.py — ~50 tests."""
from __future__ import annotations

import pytest

from src.simulation.environment import (
    ENV_REGISTRY,
    EnvAction,
    EnvState,
    EnvStep,
    Environment,
    GridWorldEnv,
)


# ---------------------------------------------------------------------------
# EnvState
# ---------------------------------------------------------------------------

class TestEnvState:
    def test_auto_generates_state_id(self):
        s = EnvState(obs={})
        assert isinstance(s.state_id, str)
        assert len(s.state_id) == 8

    def test_two_states_have_different_ids(self):
        s1 = EnvState(obs={})
        s2 = EnvState(obs={})
        assert s1.state_id != s2.state_id

    def test_default_reward_zero(self):
        s = EnvState(obs={})
        assert s.reward == 0.0

    def test_default_done_false(self):
        s = EnvState(obs={})
        assert s.done is False

    def test_default_info_empty_dict(self):
        s = EnvState(obs={})
        assert s.info == {}

    def test_info_not_shared_between_instances(self):
        s1 = EnvState(obs={})
        s2 = EnvState(obs={})
        s1.info["k"] = 1
        assert "k" not in s2.info

    def test_obs_stored_correctly(self):
        obs = {"x": 3, "y": 7}
        s = EnvState(obs=obs)
        assert s.obs == obs

    def test_custom_reward(self):
        s = EnvState(obs={}, reward=1.5)
        assert s.reward == 1.5

    def test_custom_done(self):
        s = EnvState(obs={}, done=True)
        assert s.done is True

    def test_custom_state_id(self):
        s = EnvState(obs={}, state_id="abcd1234")
        assert s.state_id == "abcd1234"


# ---------------------------------------------------------------------------
# EnvAction
# ---------------------------------------------------------------------------

class TestEnvAction:
    def test_action_id_stored(self):
        a = EnvAction(action_id="up")
        assert a.action_id == "up"

    def test_default_params_empty(self):
        a = EnvAction(action_id="left")
        assert a.params == {}

    def test_params_not_shared(self):
        a1 = EnvAction(action_id="up")
        a2 = EnvAction(action_id="up")
        a1.params["speed"] = 5
        assert "speed" not in a2.params

    def test_custom_params(self):
        a = EnvAction(action_id="right", params={"force": 2})
        assert a.params["force"] == 2


# ---------------------------------------------------------------------------
# EnvStep
# ---------------------------------------------------------------------------

class TestEnvStep:
    def _make_step(self, step_number=1):
        prev = EnvState(obs={"x": 0, "y": 0})
        action = EnvAction(action_id="right")
        nxt = EnvState(obs={"x": 1, "y": 0})
        return EnvStep(prev_state=prev, action=action, next_state=nxt, step_number=step_number)

    def test_prev_state_stored(self):
        step = self._make_step()
        assert step.prev_state.obs == {"x": 0, "y": 0}

    def test_action_stored(self):
        step = self._make_step()
        assert step.action.action_id == "right"

    def test_next_state_stored(self):
        step = self._make_step()
        assert step.next_state.obs == {"x": 1, "y": 0}

    def test_step_number_stored(self):
        step = self._make_step(step_number=5)
        assert step.step_number == 5


# ---------------------------------------------------------------------------
# GridWorldEnv
# ---------------------------------------------------------------------------

class TestGridWorldEnvReset:
    def test_reset_returns_env_state(self):
        env = GridWorldEnv()
        state = env.reset()
        assert isinstance(state, EnvState)

    def test_reset_x_zero(self):
        env = GridWorldEnv()
        state = env.reset()
        assert state.obs["x"] == 0

    def test_reset_y_zero(self):
        env = GridWorldEnv()
        state = env.reset()
        assert state.obs["y"] == 0

    def test_reset_goal_in_obs(self):
        env = GridWorldEnv(goal=(3, 3))
        state = env.reset()
        assert state.obs["goal"] == (3, 3)

    def test_reset_done_false(self):
        env = GridWorldEnv()
        state = env.reset()
        assert state.done is False


class TestGridWorldEnvStep:
    def test_step_right_increases_x(self):
        env = GridWorldEnv()
        state = env.reset()
        step = env.step(state, EnvAction("right"))
        assert step.next_state.obs["x"] == 1

    def test_step_up_increases_y(self):
        env = GridWorldEnv()
        state = env.reset()
        step = env.step(state, EnvAction("up"))
        assert step.next_state.obs["y"] == 1

    def test_step_down_from_zero_clamped(self):
        env = GridWorldEnv()
        state = env.reset()
        step = env.step(state, EnvAction("down"))
        assert step.next_state.obs["y"] == 0

    def test_step_left_from_zero_clamped(self):
        env = GridWorldEnv()
        state = env.reset()
        step = env.step(state, EnvAction("left"))
        assert step.next_state.obs["x"] == 0

    def test_step_right_at_wall_clamped(self):
        env = GridWorldEnv(width=5)
        state = EnvState(obs={"x": 4, "y": 0, "goal": (4, 4)})
        step = env.step(state, EnvAction("right"))
        assert step.next_state.obs["x"] == 4

    def test_step_up_at_wall_clamped(self):
        env = GridWorldEnv(height=5)
        state = EnvState(obs={"x": 0, "y": 4, "goal": (4, 4)})
        step = env.step(state, EnvAction("up"))
        assert step.next_state.obs["y"] == 4

    def test_step_to_goal_reward_one(self):
        env = GridWorldEnv(width=5, height=5, goal=(1, 0))
        state = env.reset()
        step = env.step(state, EnvAction("right"))
        assert step.next_state.reward == 1.0

    def test_step_to_goal_done_true(self):
        env = GridWorldEnv(width=5, height=5, goal=(1, 0))
        state = env.reset()
        step = env.step(state, EnvAction("right"))
        assert step.next_state.done is True

    def test_step_non_goal_reward_negative(self):
        env = GridWorldEnv()
        state = env.reset()
        step = env.step(state, EnvAction("right"))
        assert step.next_state.reward == pytest.approx(-0.01)

    def test_step_non_goal_done_false(self):
        env = GridWorldEnv()
        state = env.reset()
        step = env.step(state, EnvAction("right"))
        assert step.next_state.done is False

    def test_step_returns_env_step(self):
        env = GridWorldEnv()
        state = env.reset()
        step = env.step(state, EnvAction("up"))
        assert isinstance(step, EnvStep)

    def test_step_prev_state_matches(self):
        env = GridWorldEnv()
        state = env.reset()
        step = env.step(state, EnvAction("up"))
        assert step.prev_state is state


class TestGridWorldEnvValidActions:
    def test_returns_four_actions(self):
        env = GridWorldEnv()
        state = env.reset()
        actions = env.valid_actions(state)
        assert len(actions) == 4

    def test_all_action_ids_present(self):
        env = GridWorldEnv()
        state = env.reset()
        ids = {a.action_id for a in env.valid_actions(state)}
        assert ids == {"up", "down", "left", "right"}

    def test_valid_actions_returns_env_actions(self):
        env = GridWorldEnv()
        state = env.reset()
        for a in env.valid_actions(state):
            assert isinstance(a, EnvAction)


# ---------------------------------------------------------------------------
# ENV_REGISTRY
# ---------------------------------------------------------------------------

class TestEnvRegistry:
    def test_has_gridworld_key(self):
        assert "gridworld" in ENV_REGISTRY

    def test_gridworld_maps_to_class(self):
        assert ENV_REGISTRY["gridworld"] is GridWorldEnv

    def test_registry_is_dict(self):
        assert isinstance(ENV_REGISTRY, dict)


# ---------------------------------------------------------------------------
# Environment base class
# ---------------------------------------------------------------------------

class TestEnvironmentBase:
    def test_step_raises_not_implemented(self):
        env = Environment()
        state = EnvState(obs={})
        with pytest.raises(NotImplementedError):
            env.step(state, EnvAction("up"))

    def test_valid_actions_raises_not_implemented(self):
        env = Environment()
        state = EnvState(obs={})
        with pytest.raises(NotImplementedError):
            env.valid_actions(state)

    def test_reset_raises_not_implemented(self):
        env = Environment()
        with pytest.raises(NotImplementedError):
            env.reset()

    def test_name_attribute(self):
        assert Environment.name == "base"

    def test_gridworld_name(self):
        assert GridWorldEnv.name == "gridworld"
