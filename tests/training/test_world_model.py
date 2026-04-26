"""Tests for src/training/world_model.py — model-based RL world model."""

import math

import pytest
import torch
import torch.optim as optim

from src.training.world_model import (
    RewardModel,
    TransitionModel,
    WorldModelConfig,
    WorldModelTrainer,
    imagine_trajectory,
    world_model_loss,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------
B = 2
STATE_DIM = 16
ACTION_DIM = 8
HORIZON = 3


@pytest.fixture()
def cfg():
    return WorldModelConfig(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=32,
        horizon=HORIZON,
        imagination_steps=2,
    )


@pytest.fixture()
def transition(cfg):
    return TransitionModel(cfg)


@pytest.fixture()
def reward(cfg):
    return RewardModel(cfg)


@pytest.fixture()
def states():
    return torch.randn(B, STATE_DIM)


@pytest.fixture()
def actions():
    return torch.randn(B, ACTION_DIM)


@pytest.fixture()
def actions_seq():
    return torch.randn(B, HORIZON, ACTION_DIM)


@pytest.fixture()
def next_states():
    return torch.randn(B, STATE_DIM)


@pytest.fixture()
def rewards_gt():
    return torch.randn(B, 1)


@pytest.fixture()
def trainer(transition, reward, cfg):
    params = list(transition.parameters()) + list(reward.parameters())
    opt = optim.Adam(params, lr=1e-3)
    return WorldModelTrainer(transition, reward, cfg, opt)


# ---------------------------------------------------------------------------
# 1. WorldModelConfig defaults
# ---------------------------------------------------------------------------
def test_world_model_config_defaults():
    cfg = WorldModelConfig()
    assert cfg.state_dim == 64
    assert cfg.action_dim == 16
    assert cfg.hidden_dim == 128
    assert cfg.horizon == 5
    assert cfg.imagination_steps == 3


# ---------------------------------------------------------------------------
# 2. TransitionModel output shapes
# ---------------------------------------------------------------------------
def test_transition_model_next_state_shape(transition, states, actions):
    next_state, log_var = transition(states, actions)
    assert next_state.shape == (B, STATE_DIM)


def test_transition_model_log_var_shape(transition, states, actions):
    next_state, log_var = transition(states, actions)
    assert log_var.shape == (B, STATE_DIM)


# ---------------------------------------------------------------------------
# 3. TransitionModel is differentiable
# ---------------------------------------------------------------------------
def test_transition_model_differentiable(transition, states, actions):
    states = states.requires_grad_(True)
    next_state, log_var = transition(states, actions)
    loss = next_state.sum() + log_var.sum()
    loss.backward()
    assert states.grad is not None


# ---------------------------------------------------------------------------
# 4. RewardModel output shape
# ---------------------------------------------------------------------------
def test_reward_model_output_shape(reward, states, actions):
    r = reward(states, actions)
    assert r.shape == (B, 1)


# ---------------------------------------------------------------------------
# 5. RewardModel is differentiable
# ---------------------------------------------------------------------------
def test_reward_model_differentiable(reward, states, actions):
    states = states.requires_grad_(True)
    r = reward(states, actions)
    r.sum().backward()
    assert states.grad is not None


# ---------------------------------------------------------------------------
# 6-8. imagine_trajectory
# ---------------------------------------------------------------------------
def test_imagine_trajectory_returns_dict(transition, reward, states, actions_seq):
    result = imagine_trajectory(transition, reward, states, actions_seq)
    assert isinstance(result, dict)
    assert "states" in result
    assert "rewards" in result
    assert "total_reward" in result


def test_imagine_trajectory_states_shape(transition, reward, states, actions_seq):
    result = imagine_trajectory(transition, reward, states, actions_seq)
    assert result["states"].shape == (B, HORIZON + 1, STATE_DIM)


def test_imagine_trajectory_rewards_shape(transition, reward, states, actions_seq):
    result = imagine_trajectory(transition, reward, states, actions_seq)
    assert result["rewards"].shape == (B, HORIZON)


# ---------------------------------------------------------------------------
# 9-10. world_model_loss
# ---------------------------------------------------------------------------
def test_world_model_loss_keys(transition, reward, states, actions, next_states, rewards_gt):
    losses = world_model_loss(transition, reward, states, actions, next_states, rewards_gt)
    assert "state_loss" in losses
    assert "reward_loss" in losses
    assert "total_loss" in losses


def test_world_model_loss_total_is_scalar(
    transition, reward, states, actions, next_states, rewards_gt
):
    losses = world_model_loss(transition, reward, states, actions, next_states, rewards_gt)
    total = losses["total_loss"]
    assert isinstance(total, torch.Tensor)
    assert total.ndim == 0  # scalar


# ---------------------------------------------------------------------------
# 11-12. WorldModelTrainer.train_step
# ---------------------------------------------------------------------------
def test_train_step_returns_required_keys(trainer, states, actions, next_states, rewards_gt):
    result = trainer.train_step(states, actions, next_states, rewards_gt)
    assert "state_loss" in result
    assert "reward_loss" in result
    assert "total_loss" in result


def test_train_step_loss_is_finite(trainer, states, actions, next_states, rewards_gt):
    result = trainer.train_step(states, actions, next_states, rewards_gt)
    # total_loss may already be detached after backward; use float()
    total = result["total_loss"]
    val = total.item() if isinstance(total, torch.Tensor) else float(total)
    assert math.isfinite(val)


# ---------------------------------------------------------------------------
# 13. plan returns valid index
# ---------------------------------------------------------------------------
def test_plan_returns_valid_index(trainer, states, actions_seq):
    n_candidates = 4
    candidates = [torch.randn(B, HORIZON, ACTION_DIM) for _ in range(n_candidates)]
    idx = trainer.plan(states, candidates)
    assert isinstance(idx, int)
    assert 0 <= idx < n_candidates


# ---------------------------------------------------------------------------
# 14. TransitionModel log_var values are finite
# ---------------------------------------------------------------------------
def test_transition_log_var_finite(transition, states, actions):
    _, log_var = transition(states, actions)
    assert torch.isfinite(log_var).all()
