"""Token-level reward shaping for RLHF.

Distributes a scalar end-of-sequence reward across individual tokens using
KL-penalized rewards (Ziegler et al. 2019) and potential-based shaping
(Ng et al. 1999).
"""

import torch
from dataclasses import dataclass
from torch import Tensor


@dataclass
class RewardShapingConfig:
    kl_coef: float = 0.1
    gamma: float = 1.0
    whiten: bool = True
    potential_fn: str = "uniform"


def kl_penalized_rewards(
    log_probs_policy: Tensor,
    log_probs_ref: Tensor,
    scalar_reward: Tensor,
    kl_coef: float = 0.1,
    response_mask: Tensor | None = None,
) -> Tensor:
    """Compute per-token shaped rewards with KL penalty (Ziegler et al. 2019).

    Args:
        log_probs_policy: (B, T) log probs under current policy
        log_probs_ref: (B, T) log probs under reference policy
        scalar_reward: (B,) end-of-sequence reward
        kl_coef: coefficient for KL penalty
        response_mask: (B, T) 1 for response tokens, 0 for prompt tokens

    Returns:
        shaped_rewards: (B, T) per-token shaped rewards
    """
    # r_t = -kl_coef * (log_pi(a_t|s_t) - log_pi_ref(a_t|s_t))
    kl_penalty = -kl_coef * (log_probs_policy - log_probs_ref)
    shaped_rewards = kl_penalty.clone()

    B, T = shaped_rewards.shape

    if response_mask is not None:
        # Find last response token for each batch element
        # Last 1 in mask along T dimension
        # Flip mask, find first 1, convert back to original index
        flipped = response_mask.flip(dims=[1])
        last_indices = T - 1 - flipped.long().argmax(dim=1)  # (B,)
        # Add scalar reward at last response token
        for b in range(B):
            shaped_rewards[b, last_indices[b]] += scalar_reward[b]
        # Zero out prompt tokens
        shaped_rewards = shaped_rewards * response_mask
    else:
        # Add scalar reward at the last token (T-1) for each batch element
        shaped_rewards[:, -1] += scalar_reward

    return shaped_rewards


def potential_based_shaping(
    rewards: Tensor,
    gamma: float = 1.0,
    potential_fn: str = "uniform",
) -> Tensor:
    """Apply potential-based shaping bonus (Ng et al. 1999).

    F_t = gamma * Phi(s_{t+1}) - Phi(s_t)

    Args:
        rewards: (B, T) base per-token rewards
        gamma: discount factor
        potential_fn: "uniform" | "linear_decay" | "exponential_decay"

    Returns:
        shaped_rewards: (B, T) = rewards + shaping_bonus
    """
    B, T = rewards.shape
    device = rewards.device

    t_indices = torch.arange(T, dtype=torch.float32, device=device)

    if potential_fn == "uniform":
        # Phi(t) = 1.0 for all t
        phi = torch.ones(T, device=device)
    elif potential_fn == "linear_decay":
        # Phi(t) = 1 - t/T  (earlier = higher potential)
        phi = 1.0 - t_indices / T
    elif potential_fn == "exponential_decay":
        # Phi(t) = gamma^(T-t)  (later = lower potential)
        phi = gamma ** (T - t_indices)
    else:
        raise ValueError(f"Unknown potential_fn: {potential_fn!r}")

    # F_t = gamma * Phi(t+1) - Phi(t)
    # For t < T-1: F_t = gamma * phi[t+1] - phi[t]
    # For t = T-1: F_t = gamma * 0 - phi[T-1]  (terminal state has Phi=0)
    phi_next = torch.cat([phi[1:], torch.zeros(1, device=device)])
    shaping_bonus = gamma * phi_next - phi  # (T,)

    # Broadcast over batch
    shaped_rewards = rewards + shaping_bonus.unsqueeze(0)
    return shaped_rewards


def whiten_rewards(rewards: Tensor, response_mask: Tensor | None = None) -> Tensor:
    """Normalize rewards to zero mean and unit std per batch.

    Args:
        rewards: (B, T) rewards
        response_mask: (B, T) 1 for response tokens; if provided, stats
            are computed only over response tokens

    Returns:
        whitened rewards, same shape as input
    """
    eps = 1e-8

    if response_mask is not None:
        mask_bool = response_mask.bool()
        response_vals = rewards[mask_bool]
        mean = response_vals.mean()
        std = response_vals.std()
    else:
        mean = rewards.mean()
        std = rewards.std()

    return (rewards - mean) / (std + eps)


def discount_cumsum(rewards: Tensor, gamma: float) -> Tensor:
    """Compute discounted cumulative sum R_t = r_t + gamma*r_{t+1} + ...

    Args:
        rewards: (B, T) or (T,) rewards
        gamma: discount factor

    Returns:
        discounted returns, same shape as rewards
    """
    squeeze = rewards.dim() == 1
    if squeeze:
        rewards = rewards.unsqueeze(0)

    B, T = rewards.shape
    returns = torch.zeros_like(rewards)
    running_add = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        running_add = rewards[:, t] + gamma * running_add
        returns[:, t] = running_add

    if squeeze:
        returns = returns.squeeze(0)

    return returns
