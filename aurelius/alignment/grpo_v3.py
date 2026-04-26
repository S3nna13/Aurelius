"""GRPO v3: Group Relative Policy Optimization (Shao et al. 2024 / DeepSeek-Math).

Eliminates the value model by using group-relative rewards: for a group of G
completions per prompt, normalise rewards within the group to compute advantages.
Applies an asymmetric PPO-clip loss plus a KL penalty against a frozen reference.

Reference: arXiv:2402.03300 — "DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    """Hyperparameters for GRPO training (Shao et al. 2024)."""

    beta: float = 0.04
    """KL penalty coefficient."""

    group_size: int = 8
    """G: number of completions sampled per prompt."""

    clip_ratio: float = 0.2
    """Symmetric PPO clipping epsilon (used when epsilon_low == epsilon_high)."""

    epsilon_low: float = 0.2
    """Lower asymmetric clipping bound — clips ratio below (1 - epsilon_low)."""

    epsilon_high: float = 0.2
    """Upper asymmetric clipping bound — clips ratio above (1 + epsilon_high)."""


# ---------------------------------------------------------------------------
# GroupRewardNormalizer
# ---------------------------------------------------------------------------


class GroupRewardNormalizer:
    """Compute group-relative advantages from flat reward vectors.

    Given B prompts each with G completions, the flat reward tensor has
    shape (B*G,).  Rewards are normalised *within* each group of G to
    produce zero-mean, unit-variance advantages.

    Args:
        group_size: G, number of completions per prompt.
    """

    def __init__(self, group_size: int) -> None:
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.group_size = group_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, rewards: Tensor) -> Tensor:
        """Normalise a flat (B*G,) reward tensor to per-group advantages.

        Args:
            rewards: Shape (B*G,).  All G entries for prompt i must be
                     contiguous (rows i*G .. (i+1)*G - 1).

        Returns:
            Advantages tensor with the same shape (B*G,), zero mean and
            approximately unit std within each group of G.

        Raises:
            ValueError: if len(rewards) is not divisible by group_size.
        """
        G = self.group_size
        total = rewards.shape[0]
        if total % G != 0:
            raise ValueError(f"rewards length {total} is not divisible by group_size {G}")
        B = total // G
        rewards_2d = rewards.reshape(B, G)
        advantages_2d = self._normalize_2d(rewards_2d)
        return advantages_2d.reshape(B * G)

    def normalize_batch(self, rewards: Tensor) -> Tensor:
        """Normalise a (B, G) reward tensor to per-group advantages.

        Args:
            rewards: Shape (B, G).

        Returns:
            Advantages tensor of shape (B, G).
        """
        return self._normalize_2d(rewards)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_2d(rewards_2d: Tensor) -> Tensor:
        """Normalise each row of a (B, G) tensor independently."""
        mean = rewards_2d.mean(dim=1, keepdim=True)  # (B, 1)
        std = rewards_2d.std(dim=1, keepdim=True)  # (B, 1)  unbiased by default
        advantages = (rewards_2d - mean) / (std + 1e-8)
        return advantages


# ---------------------------------------------------------------------------
# GRPOLoss
# ---------------------------------------------------------------------------


class GRPOLoss(nn.Module):
    """Policy gradient loss for GRPO with asymmetric clipping and KL penalty.

    Args:
        config: GRPOConfig instance.
    """

    def __init__(self, config: GRPOConfig) -> None:
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------
    # Component methods (public so they can be tested individually)
    # ------------------------------------------------------------------

    def clip_ratio(self, ratio: Tensor) -> Tensor:
        """Asymmetric PPO clip: clamp ratio to [1-epsilon_low, 1+epsilon_high].

        Args:
            ratio: Importance-weight tensor of any shape.

        Returns:
            Clipped tensor with the same shape.
        """
        lo = 1.0 - self.config.epsilon_low
        hi = 1.0 + self.config.epsilon_high
        return ratio.clamp(lo, hi)

    def policy_loss(
        self,
        log_probs: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
    ) -> Tensor:
        """Clipped surrogate policy loss (negative, to be minimised).

        L = -E[min(r * A, clip(r) * A)]

        Args:
            log_probs:     Shape (B,) — log-probs under the current policy.
            old_log_probs: Shape (B,) — log-probs under the behaviour policy.
            advantages:    Shape (B,) — per-sample advantage estimates.

        Returns:
            Scalar loss tensor.
        """
        ratio = torch.exp(log_probs - old_log_probs)  # (B,)
        clipped = self.clip_ratio(ratio)  # (B,)
        surr1 = ratio * advantages
        surr2 = clipped * advantages
        loss = -torch.min(surr1, surr2).mean()
        return loss

    def kl_penalty(self, log_probs: Tensor, ref_log_probs: Tensor) -> Tensor:
        """Approximated reverse KL(policy || ref) per token.

        Uses the unbiased low-variance approximation:
            KL ≈ exp(log_p - log_ref) - (log_p - log_ref) - 1

        which is non-negative in expectation and equals zero when the two
        distributions are identical.

        Args:
            log_probs:     Shape (B,) — log-probs under the current policy.
            ref_log_probs: Shape (B,) — log-probs under the frozen reference.

        Returns:
            Scalar non-negative KL estimate.
        """
        diff = log_probs - ref_log_probs  # (B,)
        kl = (torch.exp(diff) - diff - 1.0).mean()
        return kl

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        log_probs: Tensor,
        old_log_probs: Tensor,
        ref_log_probs: Tensor,
        advantages: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute total GRPO loss = policy_loss + beta * kl_penalty.

        Args:
            log_probs:     Shape (B,) — current policy log-probs.
            old_log_probs: Shape (B,) — behaviour policy log-probs (no grad).
            ref_log_probs: Shape (B,) — frozen reference log-probs (no grad).
            advantages:    Shape (B,) — per-sample advantages.

        Returns:
            Tuple of:
              - total_loss: scalar Tensor (differentiable through log_probs).
              - metrics: dict with keys "policy_loss", "kl_loss", "total_loss",
                         "mean_advantage".
        """
        p_loss = self.policy_loss(log_probs, old_log_probs, advantages)
        kl_loss = self.kl_penalty(log_probs, ref_log_probs)
        total_loss = p_loss + self.config.beta * kl_loss

        metrics: dict[str, float] = {
            "policy_loss": p_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": total_loss.item(),
            "mean_advantage": advantages.mean().item(),
        }
        return total_loss, metrics


# ---------------------------------------------------------------------------
# GRPOTrainer
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """Orchestrates a single GRPO update step.

    Combines reward normalisation, loss computation, and an optimiser step
    into a single ``grpo_step`` call.

    Args:
        policy_model: The trainable policy (nn.Module).
        ref_model:    Frozen reference model (nn.Module).
        optimizer:    PyTorch optimiser bound to policy_model.parameters().
        config:       GRPOConfig.
        loss_fn:      GRPOLoss instance.
        normalizer:   GroupRewardNormalizer instance.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: GRPOConfig,
        loss_fn: GRPOLoss,
        normalizer: GroupRewardNormalizer,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.config = config
        self.loss_fn = loss_fn
        self.normalizer = normalizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def freeze_ref(self) -> None:
        """Freeze all parameters of the reference model (requires_grad = False)."""
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

    def grpo_step(
        self,
        log_probs: Tensor,
        old_log_probs: Tensor,
        ref_log_probs: Tensor,
        rewards: Tensor,
    ) -> dict[str, float]:
        """Perform one GRPO backward pass and optimiser update.

        Args:
            log_probs:     Shape (B*G,) — current policy log-probs (has grad).
            old_log_probs: Shape (B*G,) — behaviour policy log-probs (no grad).
            ref_log_probs: Shape (B*G,) — reference log-probs (no grad).
            rewards:       Shape (B*G,) — scalar rewards for each completion.

        Returns:
            Metrics dict with keys: "policy_loss", "kl_loss", "total_loss",
            "mean_advantage".
        """
        advantages = self.normalizer.normalize(rewards)

        self.optimizer.zero_grad()
        loss, metrics = self.loss_fn(log_probs, old_log_probs, ref_log_probs, advantages)
        loss.backward()
        self.optimizer.step()

        return metrics
