"""Generalized Advantage Estimation (GAE) for PPO.

Implements GAE-Lambda (Schulman et al., 2016) for computing advantages
and value targets used in Proximal Policy Optimization training.
"""

import torch
from dataclasses import dataclass


@dataclass
class GAEConfig:
    """Configuration for Generalized Advantage Estimation."""

    gamma: float = 0.99  # discount factor
    lam: float = 0.95  # GAE lambda (0=TD, 1=Monte Carlo)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    next_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    Args:
        rewards: (T,) rewards at each timestep
        values: (T,) value estimates V(s_t) for t=0..T-1
        dones: (T,) float/bool, 1.0 (or True) if episode ended at t
        gamma: discount factor
        lam: GAE lambda
        next_value: V(s_T) bootstrap value (0 if terminal)

    Returns:
        (advantages, returns) both of shape (T,)
        - advantages: GAE advantages, NOT normalized (caller normalizes)
        - returns: value targets = advantages + values
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, device=rewards.device)
    dones_float = dones.float()

    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value * (1.0 - dones_float[t])
        else:
            next_val = values[t + 1] * (1.0 - dones_float[t])

        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * (1.0 - dones_float[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def compute_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    next_value: float = 0.0,
) -> torch.Tensor:
    """Compute Monte Carlo discounted returns (GAE with lambda=1, no value baseline).

    Args:
        rewards: (T,) rewards
        dones: (T,) episode termination flags
        gamma: discount factor
        next_value: bootstrap value for non-terminal last state

    Returns:
        (T,) discounted returns G_t = r_t + gamma * G_{t+1}
    """
    T = rewards.shape[0]
    returns = torch.zeros(T, device=rewards.device)
    dones_float = dones.float()

    G = next_value
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G * (1.0 - dones_float[t])
        returns[t] = G
    return returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to zero mean and unit variance."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)
