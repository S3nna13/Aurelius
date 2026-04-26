"""Dr. GRPO: Bias-Free Group Relative Policy Optimization.

Implements the corrected advantage estimation and sequence-level loss from:
  "Dr. GRPO: Decomposing GRPO's Biases for Better Mathematical Reasoning" (2025)

Standard GRPO has two biases that Dr. GRPO eliminates:
  1. Std-normalization bias: dividing advantages by std(rewards) suppresses
     gradient signal when the group is near-uniform (low variance rewards).
  2. Length bias: token-level mean normalization treats shorter sequences
     differently from longer ones, creating an implicit length reward.

Dr. GRPO fixes both:
  - Advantage: adv_i = r_i - mean(r)  — NO std division, mean-centering only.
  - Loss: sequence-level mean (average over sequences), with each sequence's
    gradient weighted by 1/T_i (divide by its own length), not the global
    token count. This removes the length bias.

Pure PyTorch implementation — no transformers, scipy, sklearn, or other
external ML libraries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DrGRPOConfig:
    """Configuration for DrGRPOTrainer.

    Attributes:
        group_size: Number of responses G sampled per prompt.
        clip_eps: PPO probability-ratio clipping epsilon ε.
        kl_coeff: Weight λ on the reference-KL penalty term.
        eps: Small constant used to guard against division by zero.
        normalize_sequence_length: When True (default), each sequence's
            policy-gradient contribution is divided by its token count
            (1/T_i weighting).  Setting False reverts to the biased
            token-level-mean GRPO behaviour.
    """

    group_size: int = 8
    clip_eps: float = 0.2
    kl_coeff: float = 0.01
    eps: float = 1e-8
    normalize_sequence_length: bool = True


# ---------------------------------------------------------------------------
# Batch container
# ---------------------------------------------------------------------------


@dataclass
class DrGRPOBatch:
    """A group of G responses for a single prompt.

    Attributes:
        log_probs: Current-policy per-token log-probabilities, shape [G, T].
        ref_log_probs: Reference-policy per-token log-probabilities, [G, T].
        rewards: Scalar reward per response, shape [G].
        attention_mask: 1 for real tokens, 0 for padding, shape [G, T].
    """

    log_probs: Tensor  # [G, T]
    ref_log_probs: Tensor  # [G, T]
    rewards: Tensor  # [G]
    attention_mask: Tensor  # [G, T]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class DrGRPOTrainer:
    """Bias-free GRPO trainer.

    Key algorithmic differences from standard GRPO:
      * No std normalization in advantages — prevents suppressing gradient
        signal when all responses receive similar rewards.
      * Sequence-level (not token-level) loss averaging — removes the implicit
        length bias where longer responses dominate gradient updates.

    Args:
        config: A :class:`DrGRPOConfig` instance.
    """

    def __init__(self, config: DrGRPOConfig | None = None) -> None:
        self.config = config if config is not None else DrGRPOConfig()

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def compute_advantages(self, rewards: Tensor) -> Tensor:
        """Mean-center rewards to produce advantages — NO std normalization.

        Standard GRPO divides by ``std(rewards) + eps``, which shrinks gradient
        signal when the group is near-uniform.  Dr. GRPO omits this step.

        Args:
            rewards: Shape [G] — one scalar reward per response in the group.

        Returns:
            advantages: Shape [G] — mean-centered rewards.  Always sums to 0.
        """
        return rewards - rewards.mean()

    # ------------------------------------------------------------------
    # Sequence-level policy-gradient loss
    # ------------------------------------------------------------------

    def compute_sequence_loss(self, batch: DrGRPOBatch) -> Tensor:
        """Compute the sequence-level PPO-clip policy-gradient loss.

        Steps:
          1. Importance ratio  r_t = exp(log π_θ - log π_ref)  per token.
          2. Broadcast group advantages [G] → [G, T].
          3. Clipped PPO surrogate per token, zeroed on padding.
          4. Per-sequence loss: sum tokens / sequence length  (removes length bias).
          5. Return mean over G sequences.

        Args:
            batch: :class:`DrGRPOBatch` with shapes [G, T].

        Returns:
            Scalar policy-gradient loss (to be minimised).
        """
        cfg = self.config
        mask = batch.attention_mask.float()  # [G, T]

        # 1. Importance ratios — clamp log-ratio before exp() for stability.
        log_ratio = batch.log_probs - batch.ref_log_probs.detach()  # [G, T]
        log_ratio = log_ratio.clamp(-20.0, 20.0)
        ratio = log_ratio.exp()  # [G, T]

        # 2. Advantages [G], broadcast to [G, T].
        adv = self.compute_advantages(batch.rewards)  # [G]
        adv_expanded = adv.unsqueeze(1).expand_as(ratio)  # [G, T]

        # 3. Clipped PPO surrogate.
        surr1 = ratio * adv_expanded  # [G, T]
        surr2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_expanded
        pg = -torch.min(surr1, surr2) * mask  # [G, T]

        # 4. Per-sequence loss: divide each sequence's token sum by its length.
        if cfg.normalize_sequence_length:
            seq_lengths = mask.sum(dim=1).clamp(min=1.0)  # [G]
            per_seq_loss = pg.sum(dim=1) / seq_lengths  # [G]
        else:
            # Biased token-level mean (standard GRPO behaviour).
            per_seq_loss = pg.sum(dim=1)  # [G]

        # 5. Mean over group.
        return per_seq_loss.mean()

    # ------------------------------------------------------------------
    # KL penalty
    # ------------------------------------------------------------------

    def compute_kl_loss(self, batch: DrGRPOBatch) -> Tensor:
        """Estimate forward KL divergence KL(π_ref ‖ π_θ) averaged over tokens.

        Uses the one-sample approximation:
            KL ≈ (log π_ref - log π_θ) averaged over valid tokens.

        Args:
            batch: :class:`DrGRPOBatch` with shapes [G, T].

        Returns:
            Scalar non-negative KL estimate.
        """
        mask = batch.attention_mask.float()
        kl = (batch.ref_log_probs - batch.log_probs) * mask  # [G, T]
        n_tokens = mask.sum().clamp(min=1.0)
        return kl.sum() / n_tokens

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def total_loss(self, batch: DrGRPOBatch) -> dict[str, Tensor]:
        """Compute the combined Dr. GRPO training loss and diagnostics.

        Args:
            batch: :class:`DrGRPOBatch` with shapes [G, T].

        Returns:
            Dictionary with keys:
              ``"loss"``           — total loss (pg_loss + kl_coeff * kl_loss).
              ``"pg_loss"``        — policy-gradient component.
              ``"kl_loss"``        — KL-divergence component.
              ``"mean_advantage"`` — mean advantage over the group (diagnostic).
        """
        pg_loss = self.compute_sequence_loss(batch)
        kl_loss = self.compute_kl_loss(batch)
        loss = pg_loss + self.config.kl_coeff * kl_loss

        adv = self.compute_advantages(batch.rewards)
        return {
            "loss": loss,
            "pg_loss": pg_loss,
            "kl_loss": kl_loss,
            "mean_advantage": adv.mean(),
        }

    # ------------------------------------------------------------------
    # Statistics (no-grad diagnostics)
    # ------------------------------------------------------------------

    def statistics(self, batch: DrGRPOBatch) -> dict[str, float]:
        """Compute training diagnostics without building a computation graph.

        Args:
            batch: :class:`DrGRPOBatch` with shapes [G, T].

        Returns:
            Dictionary with float values:
              ``"clip_fraction"``  — fraction of valid tokens where the ratio
                                     was outside [1-ε, 1+ε].
              ``"mean_ratio"``     — mean importance ratio over valid tokens.
              ``"mean_advantage"`` — mean advantage over the group.
              ``"std_advantage"``  — std of advantages over the group.
              ``"mean_kl"``        — mean per-token KL divergence.
        """
        cfg = self.config
        with torch.no_grad():
            mask = batch.attention_mask.float()

            log_ratio = (batch.log_probs - batch.ref_log_probs).clamp(-20.0, 20.0)
            ratio = log_ratio.exp()  # [G, T]

            # Clip fraction over valid tokens.
            was_clipped = (ratio < 1.0 - cfg.clip_eps) | (ratio > 1.0 + cfg.clip_eps)
            n_valid = mask.sum().clamp(min=1.0)
            clip_fraction = (was_clipped.float() * mask).sum() / n_valid

            # Mean ratio over valid tokens.
            mean_ratio = (ratio * mask).sum() / n_valid

            # Advantage statistics.
            adv = self.compute_advantages(batch.rewards)
            mean_adv = adv.mean()
            std_adv = adv.std(unbiased=True) if adv.numel() > 1 else torch.zeros(1)

            # Mean KL over valid tokens.
            kl = (batch.ref_log_probs - batch.log_probs) * mask
            mean_kl = kl.sum() / n_valid

        return {
            "clip_fraction": clip_fraction.item(),
            "mean_ratio": mean_ratio.item(),
            "mean_advantage": mean_adv.item(),
            "std_advantage": std_adv.item(),
            "mean_kl": mean_kl.item(),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.alignment import ALIGNMENT_REGISTRY  # noqa: E402

ALIGNMENT_REGISTRY["dr_grpo"] = DrGRPOTrainer
