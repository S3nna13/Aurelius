"""RLHF utilities: KL penalty, reward processing, GAE, PPO losses, and trainer.

Pure PyTorch implementation — no external ML frameworks required.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RLHFConfig:
    """Hyperparameters for RLHF / PPO training."""

    kl_coef: float = 0.1  # coefficient for KL penalty term
    clip_ratio: float = 0.2  # PPO epsilon for ratio clipping
    vf_coef: float = 0.1  # value function loss coefficient
    entropy_coef: float = 0.01  # entropy bonus coefficient
    gamma: float = 1.0  # discount factor
    lam: float = 0.95  # GAE-Lambda coefficient
    reward_scale: float = 1.0  # scale applied to raw rewards
    reward_clip: float | None = None  # symmetric clip value for rewards


# ---------------------------------------------------------------------------
# KL penalty
# ---------------------------------------------------------------------------


def compute_kl_penalty(log_probs: Tensor, ref_log_probs: Tensor) -> Tensor:
    """Per-token KL divergence estimate: log_probs - ref_log_probs.

    Args:
        log_probs:     (B, T) log-probabilities from the current policy.
        ref_log_probs: (B, T) log-probabilities from the reference policy.

    Returns:
        (B, T) per-token KL estimate (positive means current > reference).
    """
    return log_probs - ref_log_probs


# ---------------------------------------------------------------------------
# Reward processing
# ---------------------------------------------------------------------------


def clip_rewards(rewards: Tensor, clip_val: float | None = None) -> Tensor:
    """Symmetrically clip rewards to [-clip_val, clip_val].

    Args:
        rewards:  Any-shape reward tensor.
        clip_val: If None, rewards are returned unchanged.

    Returns:
        Clipped reward tensor of the same shape.
    """
    if clip_val is None:
        return rewards
    return rewards.clamp(-clip_val, clip_val)


def whiten_rewards(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize rewards to zero mean and unit variance across the batch.

    Args:
        rewards: (B, T) reward tensor.
        eps:     Small constant for numerical stability.

    Returns:
        (B, T) whitened reward tensor.
    """
    mean = rewards.mean()
    std = rewards.std().clamp(min=eps)
    return (rewards - mean) / std


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def compute_returns(rewards: Tensor, gamma: float = 1.0) -> Tensor:
    """Compute discounted cumulative returns G_t = r_t + gamma * G_{t+1}.

    Args:
        rewards: (B, T) reward tensor; index 0 is the first timestep.
        gamma:   Discount factor.

    Returns:
        (B, T) returns tensor.
    """
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)
    g = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        g = rewards[:, t] + gamma * g
        returns[:, t] = g
    return returns


# ---------------------------------------------------------------------------
# GAE advantages
# ---------------------------------------------------------------------------


def compute_advantages_gae(
    rewards: Tensor,
    values: Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tensor:
    """Generalized Advantage Estimation (GAE-Lambda).

    delta_t = r_t + gamma * V_{t+1} - V_t
    A_t     = delta_t + gamma * lam * A_{t+1}

    The value beyond the last timestep is assumed to be 0 (terminal).

    Args:
        rewards: (B, T) rewards.
        values:  (B, T) value estimates V(s_t).
        gamma:   Discount factor.
        lam:     GAE lambda.

    Returns:
        (B, T) advantage estimates.
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    next_value = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        next_advantage = delta + gamma * lam * next_advantage
        advantages[:, t] = next_advantage
        next_value = values[:, t]

    return advantages


# ---------------------------------------------------------------------------
# PPO losses
# ---------------------------------------------------------------------------


def ppo_policy_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    clip_ratio: float = 0.2,
) -> Tensor:
    """Clipped PPO surrogate policy loss (negated for gradient ascent).

    L = -mean( min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) )

    Args:
        log_probs:     (B, T) log-probabilities from the current policy.
        old_log_probs: (B, T) log-probabilities collected during rollout.
        advantages:    (B, T) advantage estimates.
        clip_ratio:    PPO epsilon.

    Returns:
        Scalar loss tensor.
    """
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
    return -surrogate.mean()


def ppo_value_loss(
    values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip_ratio: float = 0.2,
) -> Tensor:
    """Clipped value function loss.

    L = mean( max( (V - R)^2, (clip(V, V_old - eps, V_old + eps) - R)^2 ) )

    Args:
        values:     (B, T) current value estimates.
        old_values: (B, T) value estimates collected during rollout.
        returns:    (B, T) discounted returns (regression targets).
        clip_ratio: Symmetric clip range around old_values.

    Returns:
        Scalar loss tensor.
    """
    clipped_values = old_values + (values - old_values).clamp(-clip_ratio, clip_ratio)
    loss_unclipped = (values - returns) ** 2
    loss_clipped = (clipped_values - returns) ** 2
    return torch.max(loss_unclipped, loss_clipped).mean()


# ---------------------------------------------------------------------------
# Entropy bonus
# ---------------------------------------------------------------------------


def entropy_bonus(log_probs: Tensor) -> Tensor:
    """Approximate entropy: -mean(log_probs) across all non-padded tokens.

    This is the sample-based entropy estimate H ≈ -E[log p].

    Args:
        log_probs: (B, T) log-probabilities of the sampled tokens.

    Returns:
        Scalar entropy estimate (always non-negative for valid log_probs ≤ 0).
    """
    return -log_probs.mean()


# ---------------------------------------------------------------------------
# RLHFTrainer
# ---------------------------------------------------------------------------


class RLHFTrainer:
    """High-level trainer that orchestrates PPO-style RLHF updates.

    Args:
        policy:     The policy model being optimised. Must accept input_ids
                    and return logits of shape (B, T, V).
        ref_policy: Frozen reference policy with the same signature.
        critic:     Value model; must accept input_ids and return (B, T, 1)
                    or (B, T).
        config:     RLHFConfig instance.
        optimizer:  Optimizer bound to the policy (and optionally critic).
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        critic: nn.Module,
        config: RLHFConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.policy = policy
        self.ref_policy = ref_policy
        self.critic = critic
        self.config = config
        self.optimizer = optimizer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_log_probs(self, model: nn.Module, input_ids: Tensor, response_ids: Tensor) -> Tensor:
        """Compute per-token log-probabilities for *response_ids* tokens.

        We concatenate input_ids and response_ids, run the model, take the
        logits at the response positions, and gather the log-probs of the
        actual response tokens.

        Args:
            model:        Language model returning logits (B, T_full, V).
            input_ids:    (B, T_in) prompt token ids.
            response_ids: (B, T_resp) response token ids.

        Returns:
            (B, T_resp) log-probabilities.
        """
        B, T_in = input_ids.shape
        T_resp = response_ids.shape[1]

        full_ids = torch.cat([input_ids, response_ids], dim=1)  # (B, T_in+T_resp)
        logits = model(full_ids)  # (B, T_in+T_resp, V)

        # Logits at positions T_in-1 .. T_in+T_resp-2 predict the response.
        resp_logits = logits[:, T_in - 1 : T_in - 1 + T_resp, :]  # (B, T_resp, V)
        log_probs_all = F.log_softmax(resp_logits, dim=-1)  # (B, T_resp, V)

        # Gather the log-prob of the actual token chosen.
        gathered = log_probs_all.gather(
            dim=-1,
            index=response_ids.unsqueeze(-1),
        ).squeeze(-1)  # (B, T_resp)

        return gathered

    def _get_values(self, input_ids: Tensor, response_ids: Tensor) -> Tensor:
        """Compute per-token scalar value estimates for response positions.

        Args:
            input_ids:    (B, T_in) prompt token ids.
            response_ids: (B, T_resp) response token ids.

        Returns:
            (B, T_resp) value estimates.
        """
        B, T_in = input_ids.shape
        T_resp = response_ids.shape[1]

        full_ids = torch.cat([input_ids, response_ids], dim=1)
        raw = self.critic(full_ids)  # (B, T_in+T_resp, 1) or (B, T_in+T_resp)
        if raw.dim() == 3:
            raw = raw.squeeze(-1)  # (B, T_in+T_resp)

        return raw[:, T_in - 1 : T_in - 1 + T_resp]  # (B, T_resp)

    # ------------------------------------------------------------------
    # Main compute_loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        input_ids: Tensor,
        response_ids: Tensor,
        rewards: Tensor,
    ) -> dict[str, Tensor]:
        """Run a single RLHF loss computation.

        Steps:
          1. Compute policy log-probs and detached reference log-probs.
          2. Compute value estimates (detached as old_values).
          3. Apply reward scaling and optional clipping.
          4. Subtract KL penalty from rewards → KL-penalised rewards.
          5. Compute discounted returns and GAE advantages.
          6. Compute PPO policy loss, value loss, entropy bonus.
          7. Combine into total_loss.

        Args:
            input_ids:    (B, T_in) prompt token ids.
            response_ids: (B, T_resp) response token ids.
            rewards:      (B, T_resp) per-token reward signals.

        Returns:
            Dict with keys: total_loss, policy_loss, value_loss,
                            kl_penalty, entropy.
        """
        cfg = self.config

        # --- log-probs ---------------------------------------------------
        log_probs = self._get_log_probs(self.policy, input_ids, response_ids)

        with torch.no_grad():
            ref_log_probs = self._get_log_probs(self.ref_policy, input_ids, response_ids)
            old_values = self._get_values(input_ids, response_ids)

        # --- KL penalty --------------------------------------------------
        kl = compute_kl_penalty(log_probs, ref_log_probs)  # (B, T)
        kl_penalty_scalar = kl.mean()

        # --- Reward processing -------------------------------------------
        scaled_rewards = rewards * cfg.reward_scale
        clipped_rewards = clip_rewards(scaled_rewards, cfg.reward_clip)

        # Subtract per-token KL penalty from rewards (detach KL for reward).
        penalised_rewards = clipped_rewards - cfg.kl_coef * kl.detach()

        # --- Returns & advantages ----------------------------------------
        returns = compute_returns(penalised_rewards, gamma=cfg.gamma)
        advantages = compute_advantages_gae(
            penalised_rewards,
            old_values,
            gamma=cfg.gamma,
            lam=cfg.lam,
        )

        # --- Current values for value loss --------------------------------
        values = self._get_values(input_ids, response_ids)

        # --- PPO losses ---------------------------------------------------
        policy_loss = ppo_policy_loss(
            log_probs,
            ref_log_probs,  # use ref as "old" for simplicity (one update step)
            advantages.detach(),
            clip_ratio=cfg.clip_ratio,
        )

        value_loss = ppo_value_loss(
            values,
            old_values,
            returns.detach(),
            clip_ratio=cfg.clip_ratio,
        )

        ent = entropy_bonus(log_probs)

        total_loss = policy_loss + cfg.vf_coef * value_loss - cfg.entropy_coef * ent

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "kl_penalty": kl_penalty_scalar,
            "entropy": ent,
        }
