"""PPO (Proximal Policy Optimization) trainer for RLHF.

Implements KL-penalized PPO with mini-batch updates:
- PPOConfig: hyperparameters
- RolloutBuffer: trajectory storage with GAE computation
- PPOLoss: clipped surrogate + value + KL losses
- PPOTrainer: full training loop with freeze_ref support
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
class PPOConfig:
    """Hyperparameters for KL-penalized PPO training."""

    clip_ratio: float = 0.2
    kl_coef: float = 0.1
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 1.0
    gae_lambda: float = 0.95
    n_epochs: int = 4
    minibatch_size: int = 8


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """Stores trajectories and computes GAE advantages/returns."""

    def __init__(self) -> None:
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._ref_log_probs: list[torch.Tensor] = []
        self._masks: list[torch.Tensor] = []

    def add(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        ref_log_probs: torch.Tensor,
        masks: torch.Tensor,
    ) -> None:
        """Append one step to the buffer.

        Args:
            log_probs:     scalar or (B,) log-probs under current policy
            rewards:       scalar or (B,) reward at this step
            values:        scalar or (B,) value estimate V(s_t)
            ref_log_probs: scalar or (B,) log-probs under reference policy
            masks:         scalar or (B,) 1=valid, 0=padded/done
        """
        self._log_probs.append(log_probs)
        self._rewards.append(rewards)
        self._values.append(values)
        self._ref_log_probs.append(ref_log_probs)
        self._masks.append(masks)

    def compute_advantages(
        self, gamma: float, gae_lambda: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns.

        GAE: delta_t = r_t + gamma*V_{t+1} - V_t
             A_t     = sum_{l>=0} (gamma*lambda)^l * delta_{t+l}

        Returns:
            (advantages, returns) each of shape (T,) or (T, B) depending on
            how tensors were stored.
        """
        T = len(self._rewards)
        rewards = torch.stack(self._rewards)  # (T, ...)
        values = torch.stack(self._values)  # (T, ...)
        masks = torch.stack(self._masks)  # (T, ...)

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[0])

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros_like(values[0])
            else:
                next_value = values[t + 1] * masks[t]

            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * masks[t] * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def clear(self) -> None:
        """Empty the buffer."""
        self._log_probs.clear()
        self._rewards.clear()
        self._values.clear()
        self._ref_log_probs.clear()
        self._masks.clear()

    def size(self) -> int:
        """Number of steps stored."""
        return len(self._rewards)


# ---------------------------------------------------------------------------
# PPO Loss
# ---------------------------------------------------------------------------


class PPOLoss:
    """Computes PPO losses: clipped policy, value function, KL penalty."""

    def __init__(self, config: PPOConfig) -> None:
        self.config = config

    def policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Clipped surrogate policy loss (to be minimized).

        Args:
            log_probs:     (B,) log-probs under current policy
            old_log_probs: (B,) log-probs under old policy (detached)
            advantages:    (B,) advantage estimates

        Returns:
            Scalar loss.  Negative when policy is improving.
        """
        ratio = torch.exp(log_probs - old_log_probs)
        clipped = ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
        loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        return loss

    def value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """MSE value function loss.

        Args:
            values:  (B,) current value estimates
            returns: (B,) GAE returns (targets)

        Returns:
            Scalar non-negative loss.
        """
        return F.mse_loss(values, returns)

    def kl_loss(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward KL approximation: mean(log_probs - ref_log_probs).

        Args:
            log_probs:     (B,) current policy log-probs
            ref_log_probs: (B,) reference policy log-probs

        Returns:
            Scalar KL estimate.
        """
        return (log_probs - ref_log_probs).mean()

    def total_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Combined PPO loss.

        L = L_CLIP + vf_coef * L_VF + kl_coef * L_KL

        Returns:
            (total_loss, {"policy_loss": ..., "value_loss": ..., "kl_loss": ...})
        """
        cfg = self.config
        l_clip = self.policy_loss(log_probs, old_log_probs, advantages)
        l_vf = self.value_loss(values, returns)
        l_kl = self.kl_loss(log_probs, ref_log_probs)

        total = l_clip + cfg.vf_coef * l_vf + cfg.kl_coef * l_kl
        metrics = {
            "policy_loss": l_clip,
            "value_loss": l_vf,
            "kl_loss": l_kl,
        }
        return total, metrics


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """PPO trainer for RLHF fine-tuning.

    Args:
        policy_model: Trainable policy nn.Module
        value_model:  Trainable value-head nn.Module
        ref_model:    Reference (frozen) policy nn.Module
        optimizer:    PyTorch optimizer covering policy + value parameters
        config:       PPOConfig
        loss_fn:      PPOLoss instance
    """

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        ref_model: nn.Module,
        optimizer,
        config: PPOConfig,
        loss_fn: PPOLoss,
    ) -> None:
        self.policy_model = policy_model
        self.value_model = value_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.config = config
        self.loss_fn = loss_fn

    def freeze_ref(self) -> None:
        """Freeze all parameters of the reference model."""
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

    def ppo_step(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> dict[str, float]:
        """Single gradient update step.

        Args:
            log_probs:     (B,) current policy log-probs (with grad)
            old_log_probs: (B,) old policy log-probs (detached)
            advantages:    (B,) GAE advantages (detached)
            values:        (B,) current value estimates (with grad)
            returns:       (B,) GAE returns (detached)
            ref_log_probs: (B,) reference log-probs (detached)

        Returns:
            Dict with float metrics: 'total_loss', 'policy_loss', 'value_loss', 'kl_loss'
        """
        self.optimizer.zero_grad()

        total, metrics = self.loss_fn.total_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs.detach(),
            advantages=advantages.detach(),
            values=values,
            returns=returns.detach(),
            ref_log_probs=ref_log_probs.detach(),
        )

        total.backward()
        self.optimizer.step()

        result = {"total_loss": total.item()}
        result.update({k: v.item() for k, v in metrics.items()})
        return result

    def compute_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns for a single rollout.

        Args:
            rewards: (T,) rewards at each timestep
            values:  (T+1,) value estimates including bootstrap value at T

        Returns:
            (advantages, returns) each of shape (T,)
        """
        T = rewards.shape[0]
        gamma = self.config.gamma
        lam = self.config.gae_lambda

        advantages = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)
        gae = 0.0

        for t in reversed(range(T)):
            next_value = float(values[t + 1])
            delta = float(rewards[t]) + gamma * next_value - float(values[t])
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        returns = advantages + values[:T]
        return advantages, returns
