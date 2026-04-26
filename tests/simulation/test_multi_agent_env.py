"""Tests for src/simulation/multi_agent_env.py — 10+ tests."""

import pytest

from src.simulation.multi_agent_env import (
    _DOWN,
    _STAY,
    _UP,
    AgentAction,
    MultiAgentEnv,
    MultiAgentStep,
    SimpleCooperativeEnv,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    return SimpleCooperativeEnv(grid_size=5, goal=(4, 4))


@pytest.fixture
def small_env():
    return SimpleCooperativeEnv(grid_size=3, goal=(2, 2))


# ---------------------------------------------------------------------------
# 1. Reset
# ---------------------------------------------------------------------------


def test_reset_returns_dict_with_agent_ids(env):
    obs = env.reset()
    assert "agent_0" in obs
    assert "agent_1" in obs


def test_reset_positions_differ(env):
    obs = env.reset()
    assert obs["agent_0"]["position"] != obs["agent_1"]["position"]


def test_reset_goal_in_obs(env):
    obs = env.reset()
    assert "goal" in obs["agent_0"]
    assert obs["agent_0"]["goal"] == [4, 4]


# ---------------------------------------------------------------------------
# 2. Step — basic movement
# ---------------------------------------------------------------------------


def test_step_returns_multi_agent_step(env):
    env.reset()
    actions = [
        AgentAction(agent_id="agent_0", action=_STAY),
        AgentAction(agent_id="agent_1", action=_STAY),
    ]
    result = env.step(actions)
    assert isinstance(result, MultiAgentStep)


def test_step_rewards_dict_has_all_agents(env):
    env.reset()
    actions = [
        AgentAction(agent_id="agent_0", action=_STAY),
        AgentAction(agent_id="agent_1", action=_STAY),
    ]
    result = env.step(actions)
    assert "agent_0" in result.rewards
    assert "agent_1" in result.rewards


def test_step_dones_false_before_goal(env):
    env.reset()
    actions = [
        AgentAction(agent_id="agent_0", action=_STAY),
        AgentAction(agent_id="agent_1", action=_STAY),
    ]
    result = env.step(actions)
    assert result.dones["agent_0"] is False
    assert result.dones["agent_1"] is False


def test_step_negative_reward_before_goal(env):
    env.reset()
    actions = [
        AgentAction(agent_id="agent_0", action=_STAY),
        AgentAction(agent_id="agent_1", action=_STAY),
    ]
    result = env.step(actions)
    assert result.rewards["agent_0"] == pytest.approx(-0.01)


def test_movement_changes_position(env):
    env.reset()
    actions = [
        AgentAction(agent_id="agent_0", action=_DOWN),
        AgentAction(agent_id="agent_1", action=_STAY),
    ]
    result = env.step(actions)
    # agent_0 starts at [0,0], moves down → [1,0]
    assert result.observations["agent_0"]["position"] == [1, 0]


def test_wall_clamp_prevents_going_out_of_bounds(small_env):
    small_env.reset()
    # agent_0 starts at [0,0]; move UP which is (-1,0) → clamped to [0,0]
    actions = [
        AgentAction(agent_id="agent_0", action=_UP),
        AgentAction(agent_id="agent_1", action=_STAY),
    ]
    result = small_env.step(actions)
    assert result.observations["agent_0"]["position"][0] >= 0


# ---------------------------------------------------------------------------
# 3. Cooperative goal
# ---------------------------------------------------------------------------


def test_shared_reward_when_both_at_goal(small_env):
    """Drive both agents to (2,2) and confirm shared +1 reward and done."""
    small_env.reset()
    # Force positions directly
    small_env._positions["agent_0"] = [2, 2]
    small_env._positions["agent_1"] = [2, 2]
    actions = [
        AgentAction(agent_id="agent_0", action=_STAY),
        AgentAction(agent_id="agent_1", action=_STAY),
    ]
    result = small_env.step(actions)
    assert result.rewards["agent_0"] == pytest.approx(1.0)
    assert result.rewards["agent_1"] == pytest.approx(1.0)
    assert result.dones["agent_0"] is True
    assert result.dones["agent_1"] is True


def test_no_shared_reward_if_only_one_at_goal(small_env):
    small_env.reset()
    small_env._positions["agent_0"] = [2, 2]
    small_env._positions["agent_1"] = [0, 0]
    actions = [
        AgentAction(agent_id="agent_0", action=_STAY),
        AgentAction(agent_id="agent_1", action=_STAY),
    ]
    result = small_env.step(actions)
    assert result.rewards["agent_0"] == pytest.approx(-0.01)


# ---------------------------------------------------------------------------
# 4. Render
# ---------------------------------------------------------------------------


def test_render_returns_string(env):
    env.reset()
    s = env.render()
    assert isinstance(s, str)
    assert len(s) > 0


# ---------------------------------------------------------------------------
# 5. AgentAction dataclass
# ---------------------------------------------------------------------------


def test_agent_action_defaults():
    a = AgentAction(agent_id="x", action=0)
    assert a.metadata == {}


# ---------------------------------------------------------------------------
# 6. MultiAgentEnv base raises NotImplementedError
# ---------------------------------------------------------------------------


def test_base_env_reset_raises():
    env = MultiAgentEnv()
    with pytest.raises(NotImplementedError):
        env.reset()


def test_base_env_step_raises():
    env = MultiAgentEnv()
    with pytest.raises(NotImplementedError):
        env.step([])
