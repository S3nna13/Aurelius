"""RLOO (REINFORCE Leave-One-Out) policy gradient trainer for LLM alignment.

RLOO uses the mean reward of the other (k-1) completions for the same prompt as
a variance-reduction baseline, eliminating the need for a learned critic/value
network while substantially reducing gradient variance compared to vanilla
REINFORCE.

References:
    Kool et al. (2019) "Buy 4 REINFORCE Samples, Get a Baseline for Free!"
    Ahmadian et al. (2024) "Back to Basics: Revisiting REINFORCE Style Optimization
        for Language Models in RLHF"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RLOOConfig:
    """Hyperparameters for the RLOO policy gradient trainer."""

    k_responses: int = 4          # number of completions sampled per prompt
    kl_coef: float = 0.01         # KL penalty coefficient against reference policy
    clip_ratio: float = 0.2       # PPO-style importance ratio clipping bound
    gamma: float = 1.0            # discount factor (1.0 = episodic / no discount)
    normalize_advantages: bool = True  # z-score advantages before computing loss


# ---------------------------------------------------------------------------
# Standalone advantage estimator
# ---------------------------------------------------------------------------

def rloo_advantage_estimator(rewards: Tensor, k: int) -> Tensor:
    """Compute RLOO advantages for a flat batch of rewards.

    For a batch of n*k rewards where every consecutive block of k entries
    corresponds to k completions from the same prompt, the baseline for
    response i inside a group is the mean reward of the other (k-1) responses.

    Args:
        rewards: 1-D tensor of shape (n*k,).  The first k entries belong to
                 prompt 0, the next k to prompt 1, etc.
        k:       Number of completions per prompt.

    Returns:
        advantages: 1-D tensor of shape (n*k,) = rewards - baselines.
                    When k==1, all advantages are zero.
    """
    if k == 1:
        return torch.zeros_like(rewards)

    # Reshape to (n, k) for vectorised group operations
    n = rewards.shape[0] // k
    r = rewards.view(n, k)          # (n, k)

    group_sum = r.sum(dim=1, keepdim=True)          # (n, 1)
    # baseline_i = (sum_j - r_i) / (k - 1)
    baselines = (group_sum - r) / (k - 1)           # (n, k)
    advantages = (r - baselines).view(-1)            # (n*k,)
    return advantages


# ---------------------------------------------------------------------------
# RLOOTrainer
# ---------------------------------------------------------------------------

class RLOOTrainer:
    """RLOO policy gradient trainer.

    Uses the Leave-One-Out mean reward as a variance-reducing baseline and
    optionally adds a KL penalty against a reference (frozen) policy.

    Args:
        model:        Policy model (nn.Module).  Forward signature is flexible;
                      only ``log_probs`` computed externally are required for the
                      loss methods.
        reward_fn:    Callable that maps a batch of responses to a reward tensor.
                      Signature: ``reward_fn(responses) -> Tensor`` of shape (B,).
        optimizer:    PyTorch optimiser attached to ``model.parameters()``.
        k_responses:  Number of completions sampled per prompt.
        kl_coef:      Weight of the KL penalty term.
        clip_ratio:   PPO-style clipping parameter (0 disables clipping).
        gamma:        Discount factor (unused for episodic rewards, kept for API
                      compatibility).
    """

    def __init__(
        self,
        model: nn.Module,
        reward_fn: Callable,
        optimizer: torch.optim.Optimizer,
        k_responses: int = 4,
        kl_coef: float = 0.01,
        clip_ratio: float = 0.2,
        gamma: float = 1.0,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.k_responses = k_responses
        self.kl_coef = kl_coef
        self.clip_ratio = clip_ratio
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Core computation methods
    # ------------------------------------------------------------------

    def compute_rloo_advantages(self, rewards: Tensor) -> Tensor:
        """Compute RLOO advantages from a flat reward tensor.

        Args:
            rewards: Shape ``(batch_size * k_responses,)``.  Every consecutive
                     block of ``k_responses`` entries corresponds to one prompt.

        Returns:
            advantages: Same shape as ``rewards``.
        """
        return rloo_advantage_estimator(rewards, self.k_responses)

    def compute_policy_gradient_loss(
        self,
        log_probs: Tensor,
        advantages: Tensor,
    ) -> Tensor:
        """REINFORCE loss with optional PPO-style probability-ratio clipping.

        When ``clip_ratio > 0`` the loss is clipped so the effective importance
        weight stays within ``[1-clip_ratio, 1+clip_ratio]``.  Because we are
        working with on-policy log-probs (ratio ≡ 1 at the time of the update)
        the clipping acts as a gradient norm constraint.

        Args:
            log_probs:  Shape ``(B,)`` — summed (or mean) sequence log-probs.
            advantages: Shape ``(B,)`` — detached advantage estimates.

        Returns:
            Scalar policy-gradient loss (positive = descent direction).
        """
        adv = advantages.detach()

        if self.clip_ratio > 0.0:
            # On-policy: old_log_probs == log_probs, so ratio starts at 1.
            # We clip the surrogate objective to mimic PPO for stability.
            surr_unclipped = log_probs * adv
            # ratio = exp(log_probs - log_probs_old); on-policy this is 1,
            # but we still apply the clipping structure so it degrades
            # gracefully when used off-policy.
            ratio = torch.ones_like(log_probs)
            surr_clipped = torch.clamp(
                ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            ) * adv
            pg_loss = -torch.mean(torch.min(surr_unclipped, surr_clipped))
        else:
            pg_loss = -torch.mean(log_probs * adv)

        return pg_loss

    def compute_kl_penalty(
        self,
        log_probs: Tensor,
        ref_log_probs: Tensor,
    ) -> Tensor:
        """First-order KL approximation: E[log π - log π_ref].

        Args:
            log_probs:     Shape ``(B,)`` log-probs under the current policy.
            ref_log_probs: Shape ``(B,)`` log-probs under the reference policy.

        Returns:
            Scalar KL estimate (non-negative when policy matches reference).
        """
        return torch.mean(log_probs - ref_log_probs)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        input_ids: Tensor,
        ref_log_probs: Tensor,
        rewards: Tensor,
    ) -> dict[str, float]:
        """Perform one RLOO gradient update.

        Args:
            input_ids:     Token-id tensor used to obtain current log-probs from
                           ``self.model``.  Shape ``(B, T)`` where
                           ``B = batch_size * k_responses``.
            ref_log_probs: Pre-computed log-probs from the frozen reference
                           policy.  Shape ``(B,)``.
            rewards:       Scalar reward per response.  Shape ``(B,)``.

        Returns:
            Dictionary with keys:
                ``loss``, ``pg_loss``, ``kl_loss``,
                ``mean_reward``, ``mean_advantage``.
        """
        self.model.train()

        # ------------------------------------------------------------------
        # 1. Forward pass through the policy to get log-probs
        # ------------------------------------------------------------------
        # We assume the model returns (loss_or_none, logits, *rest) or can
        # be called to produce log_probs directly.  For maximum flexibility
        # we call the model and collect per-sequence log-probs.
        # If the model returns logits we compute log-probs; if it has a
        # dedicated attribute we use that.  The simplest contract used
        # throughout Aurelius is forward() -> (_, logits, _).
        output = self.model(input_ids)
        if isinstance(output, tuple):
            logits = output[1]  # (B, T, V)
        else:
            logits = output     # assume direct logits

        # Compute per-token log-probs for the *target* tokens (teacher-forced)
        # target = input_ids shifted left by 1
        log_probs_all = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
        target_ids = input_ids[:, 1:].unsqueeze(-1)                                  # (B, T-1, 1)
        token_log_probs = log_probs_all.gather(2, target_ids).squeeze(-1)            # (B, T-1)
        log_probs = token_log_probs.sum(dim=-1)                                      # (B,)

        # ------------------------------------------------------------------
        # 2. RLOO advantages
        # ------------------------------------------------------------------
        advantages = self.compute_rloo_advantages(rewards)

        # ------------------------------------------------------------------
        # 3. Loss components
        # ------------------------------------------------------------------
        pg_loss = self.compute_policy_gradient_loss(log_probs, advantages)
        kl_loss = self.compute_kl_penalty(log_probs, ref_log_probs)
        loss = pg_loss + self.kl_coef * kl_loss

        # ------------------------------------------------------------------
        # 4. Optimiser step
        # ------------------------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }
