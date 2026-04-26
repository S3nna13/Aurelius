"""Model-based RL with a learned world model for planning and imagination.

Components:
1. TransitionModel    -- predicts next state given current state + action
2. RewardModel        -- predicts scalar reward given state + action
3. imagine_trajectory -- rolls out a multi-step trajectory in latent space
4. world_model_loss   -- MSE losses for state and reward reconstruction
5. WorldModelTrainer  -- trains both models and exposes a planning interface
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class WorldModelConfig:
    state_dim: int = 64
    action_dim: int = 16
    hidden_dim: int = 128
    horizon: int = 5
    imagination_steps: int = 3


# ---------------------------------------------------------------------------
# TransitionModel
# ---------------------------------------------------------------------------


class TransitionModel(nn.Module):
    """Predicts the next latent state (and its log-variance) from state + action."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        in_dim = config.state_dim + config.action_dim

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(config.hidden_dim, config.state_dim)
        self.log_var_head = nn.Linear(config.hidden_dim, config.state_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Args:
            state:  (B, state_dim)
            action: (B, action_dim)
        Returns:
            next_state: (B, state_dim)
            log_var:    (B, state_dim)
        """
        x = torch.cat([state, action], dim=-1)
        h = self.trunk(x)
        next_state = self.mean_head(h)
        log_var = self.log_var_head(h)
        return next_state, log_var


# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------


class RewardModel(nn.Module):
    """Predicts a scalar reward from state + action."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        in_dim = config.state_dim + config.action_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:  (B, state_dim)
            action: (B, action_dim)
        Returns:
            reward: (B, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Trajectory imagination
# ---------------------------------------------------------------------------


def imagine_trajectory(
    transition_model: TransitionModel,
    reward_model: RewardModel,
    init_state: torch.Tensor,
    actions: torch.Tensor,
) -> dict:
    """Roll out a trajectory in imagination.

    Args:
        transition_model: trained TransitionModel
        reward_model:     trained RewardModel
        init_state:       (B, state_dim) initial latent state
        actions:          (B, horizon, action_dim) action sequence

    Returns:
        {
            "states":       (B, horizon+1, state_dim),
            "rewards":      (B, horizon),
            "total_reward": (B,),
        }
    """
    B, horizon, _ = actions.shape
    state = init_state  # (B, state_dim)

    all_states = [state]
    all_rewards = []

    for t in range(horizon):
        action_t = actions[:, t, :]  # (B, action_dim)
        next_state, _ = transition_model(state, action_t)
        reward_t = reward_model(state, action_t)  # (B, 1)
        all_states.append(next_state)
        all_rewards.append(reward_t)
        state = next_state

    states = torch.stack(all_states, dim=1)  # (B, horizon+1, state_dim)
    rewards = torch.cat(all_rewards, dim=-1)  # (B, horizon)
    total_reward = rewards.sum(dim=-1)  # (B,)

    return {"states": states, "rewards": rewards, "total_reward": total_reward}


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def world_model_loss(
    transition_model: TransitionModel,
    reward_model: RewardModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
) -> dict:
    """Compute reconstruction losses for the world model.

    Args:
        states:      (B, state_dim)
        actions:     (B, action_dim)
        next_states: (B, state_dim)  ground-truth next states
        rewards:     (B, 1) or (B,)  ground-truth rewards

    Returns:
        {
            "state_loss":  float,
            "reward_loss": float,
            "total_loss":  Tensor (scalar),
        }
    """
    pred_next_state, _ = transition_model(states, actions)
    state_loss = F.mse_loss(pred_next_state, next_states)

    pred_reward = reward_model(states, actions)  # (B, 1)
    target_reward = rewards.view_as(pred_reward)
    reward_loss = F.mse_loss(pred_reward, target_reward)

    total_loss = state_loss + reward_loss

    return {
        "state_loss": state_loss.item(),
        "reward_loss": reward_loss.item(),
        "total_loss": total_loss,
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class WorldModelTrainer:
    """Trains the world model on collected (s, a, s', r) transitions.

    Args:
        transition_model: TransitionModel
        reward_model:     RewardModel
        config:           WorldModelConfig
        optimizer:        any torch Optimizer covering both models' params
    """

    def __init__(
        self,
        transition_model: TransitionModel,
        reward_model: RewardModel,
        config: WorldModelConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.config = config
        self.optimizer = optimizer

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
    ) -> dict:
        """One gradient step on a batch of transitions.

        Returns dict with keys: state_loss, reward_loss, total_loss.
        """
        self.transition_model.train()
        self.reward_model.train()
        self.optimizer.zero_grad()

        losses = world_model_loss(
            self.transition_model,
            self.reward_model,
            states,
            actions,
            next_states,
            rewards,
        )
        losses["total_loss"].backward()
        self.optimizer.step()
        return losses

    def plan(
        self,
        init_state: torch.Tensor,
        candidate_actions_list: list[torch.Tensor],
    ) -> int:
        """Select the best action sequence by imagining each candidate.

        Args:
            init_state:            (B, state_dim) or (state_dim,) initial state
            candidate_actions_list: list of N tensors each (B, horizon, action_dim)
                                    or (horizon, action_dim) if unbatched

        Returns:
            Index of the candidate with the highest mean total reward.
        """
        self.transition_model.train(False)
        self.reward_model.train(False)

        if init_state.dim() == 1:
            init_state = init_state.unsqueeze(0)  # (1, state_dim)

        best_idx = 0
        best_reward = float("-inf")

        with torch.no_grad():
            for i, actions in enumerate(candidate_actions_list):
                if actions.dim() == 2:
                    actions = actions.unsqueeze(0)  # (1, horizon, action_dim)
                B = actions.shape[0]
                state = init_state.expand(B, -1)
                traj = imagine_trajectory(self.transition_model, self.reward_model, state, actions)
                total = traj["total_reward"].mean().item()
                if total > best_reward:
                    best_reward = total
                    best_idx = i

        return best_idx
