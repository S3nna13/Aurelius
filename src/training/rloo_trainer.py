"""RLOO (REINFORCE Leave-One-Out) Trainer.

Implements the unbiased group-baseline policy gradient from:
  Ahmadian et al. 2024 — "Back to Basics: Revisiting REINFORCE Simple Algorithm"

Instead of a learned value-function baseline, RLOO uses the mean reward of the
*other* k-1 responses in the same prompt group as the baseline for each response:

    baseline_i = (sum_{j≠i} r_j) / (k - 1)
    advantage_i = r_i - baseline_i

This yields an unbiased, low-variance policy gradient with no separate value
network required.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RLOOConfig:
    """Configuration for RLOOTrainer.

    Attributes:
        k:                    Number of responses sampled per prompt. Must be >= 2.
        kl_coeff:             Weight for the per-token KL penalty term.
        normalize_advantages: If True, z-score advantages across the whole batch.
        eps:                  Small constant for numerical stability.
    """

    k: int = 8
    kl_coeff: float = 0.05
    normalize_advantages: bool = True
    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


@dataclass
class RLOOBatch:
    """A single RLOO training batch.

    All tensors carry rows in prompt-major order: rows [i*k : (i+1)*k] belong
    to prompt i, so the total batch dimension is B*k where B is the number of
    distinct prompts.

    Attributes:
        log_probs:      [B*k, T] per-token log-probabilities under the current policy.
        ref_log_probs:  [B*k, T] per-token log-probabilities under the reference policy.
        rewards:        [B*k]    scalar reward for each response.
        attention_mask: [B*k, T] 1 for real tokens, 0 for padding.
    """

    log_probs: Tensor
    ref_log_probs: Tensor
    rewards: Tensor
    attention_mask: Tensor


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class RLOOTrainer:
    """REINFORCE Leave-One-Out policy gradient trainer.

    Args:
        config: RLOOConfig instance controlling k, kl_coeff, and normalisation.
    """

    def __init__(self, config: RLOOConfig) -> None:
        if config.k < 2:
            raise ValueError(f"k must be >= 2, got {config.k}")
        self.config = config

    # ------------------------------------------------------------------
    # Core: leave-one-out advantages
    # ------------------------------------------------------------------

    def compute_loo_advantages(self, rewards: Tensor) -> Tensor:
        """Compute RLOO advantages for a flat [B*k] reward tensor.

        For each group of k responses belonging to the same prompt, the
        baseline for response i is the mean reward of the other k-1 responses:

            baseline_i = (sum_group - r_i) / (k - 1)
            advantage_i = r_i - baseline_i

        If ``config.normalize_advantages`` is True the output is z-scored
        (mean 0, std 1) across the entire batch.

        Args:
            rewards: [B*k] float tensor.

        Returns:
            advantages: [B*k] float tensor, same shape as input.
        """
        k = self.config.k
        total = rewards.shape[0]
        if total % k != 0:
            raise ValueError(f"rewards.shape[0]={total} is not divisible by k={k}")
        b = total // k

        r = rewards.view(b, k)  # [B, k]
        sum_r = r.sum(dim=1, keepdim=True)  # [B, 1]
        loo_baseline = (sum_r - r) / (k - 1)  # [B, k]
        adv = (r - loo_baseline).view(-1)  # [B*k]

        if self.config.normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + self.config.eps)

        return adv

    # ------------------------------------------------------------------
    # KL penalty
    # ------------------------------------------------------------------

    def compute_kl_penalty(
        self,
        log_probs: Tensor,
        ref_log_probs: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """One-directional per-token KL approximation averaged over masked tokens.

        KL(policy || ref) ≈ log_prob - ref_log_prob  (first-order approximation)

        Args:
            log_probs:     [B*k, T] current policy log-probs.
            ref_log_probs: [B*k, T] reference policy log-probs.
            mask:          [B*k, T] binary mask (1 = real token).

        Returns:
            Scalar mean KL over all unmasked token positions.
        """
        per_token_kl = log_probs - ref_log_probs  # [B*k, T]
        # Average only over tokens that are not padding
        masked_kl = per_token_kl * mask
        n_tokens = mask.sum().clamp(min=1.0)
        return masked_kl.sum() / n_tokens

    # ------------------------------------------------------------------
    # Policy loss
    # ------------------------------------------------------------------

    def compute_policy_loss(self, batch: RLOOBatch) -> Tensor:
        """Policy gradient loss using RLOO advantages.

        PG loss = -mean_{masked tokens}( advantage_i * log_prob_{i,t} )

        Args:
            batch: RLOOBatch with shapes described in the dataclass.

        Returns:
            Scalar policy-gradient loss (to be minimised).
        """
        adv = self.compute_loo_advantages(batch.rewards)  # [B*k]
        # Broadcast advantage to token dimension
        adv_expanded = adv.unsqueeze(1) * batch.attention_mask  # [B*k, T]
        pg = -adv_expanded * batch.log_probs  # [B*k, T]
        n_tokens = batch.attention_mask.sum().clamp(min=1.0)
        return pg.sum() / n_tokens

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def total_loss(self, batch: RLOOBatch) -> dict[str, Tensor]:
        """Compute the total RLOO loss and associated diagnostics.

        Args:
            batch: RLOOBatch.

        Returns:
            dict with keys:
                ``loss``           — total loss (pg_loss + kl_coeff * kl_loss)
                ``pg_loss``        — policy-gradient component
                ``kl_loss``        — KL penalty component
                ``mean_advantage`` — mean advantage (scalar tensor)
        """
        pg_loss = self.compute_policy_loss(batch)
        kl_loss = self.compute_kl_penalty(
            batch.log_probs, batch.ref_log_probs, batch.attention_mask
        )
        loss = pg_loss + self.config.kl_coeff * kl_loss

        adv = self.compute_loo_advantages(batch.rewards)

        return {
            "loss": loss,
            "pg_loss": pg_loss,
            "kl_loss": kl_loss,
            "mean_advantage": adv.mean(),
        }

    # ------------------------------------------------------------------
    # Statistics (detached floats for logging)
    # ------------------------------------------------------------------

    def statistics(self, batch: RLOOBatch) -> dict[str, float]:
        """Return a dict of Python floats for logging/monitoring.

        Args:
            batch: RLOOBatch.

        Returns:
            dict with keys: mean_advantage, std_advantage, mean_kl, mean_reward.
        """
        with torch.no_grad():
            adv = self.compute_loo_advantages(batch.rewards)
            kl = self.compute_kl_penalty(batch.log_probs, batch.ref_log_probs, batch.attention_mask)
        return {
            "mean_advantage": adv.mean().item(),
            "std_advantage": adv.std().item(),
            "mean_kl": kl.item(),
            "mean_reward": batch.rewards.mean().item(),
        }


# ---------------------------------------------------------------------------
# Register in TRAINING_REGISTRY
# ---------------------------------------------------------------------------

from src.training import TRAINING_REGISTRY  # noqa: E402

TRAINING_REGISTRY["rloo"] = RLOOTrainer
