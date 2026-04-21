"""Aurelius -- KTO Trainer (Kahneman-Tversky Optimization).

KTO (Ethayarajh et al. 2024, "KTO: Model Alignment as Prospect Theoretic
Optimization", arXiv:2402.01306) aligns language models using binary desirability
labels on individual responses rather than paired (chosen, rejected) comparisons.

Inspired by Kahneman-Tversky prospect theory:
  - Humans are more averse to losses than motivated by equivalent gains
  - Desirable outputs are rewarded via sigmoid(β · (log_ratio - z_ref))
  - Undesirable outputs are penalised via sigmoid(β · (z_ref - log_ratio))

Algorithm:
    z_ref   = clamp(mean(log π_θ(y|x) - log π_ref(y|x)), min=0)   # KL anchor

    For desirable outputs:
        L_d = 1 - sigmoid(β · (log_ratio - z_ref))

    For undesirable outputs:
        L_u = 1 - sigmoid(β · (z_ref - log_ratio))

    Total: mean(L_d) + λ_U · mean(L_u)

References:
    Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic
    Optimization", arXiv:2402.01306, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KTOConfig:
    """Hyperparameters for KTO training.

    Args:
        beta: Temperature scaling for the sigmoid logistic function.
        lambda_u: Weight applied to the undesirable loss component.
        kl_num_samples: Number of batch samples used to estimate z_ref.
            If 0, all samples in the batch are used.
        eps: Small constant for numerical stability guards.
    """

    beta: float = 0.1
    lambda_u: float = 1.0
    kl_num_samples: int = 8
    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Batch container
# ---------------------------------------------------------------------------


@dataclass
class KTOBatch:
    """A single training batch for KTO.

    Args:
        log_probs: Current policy token log-probs, shape [B, T].
        ref_log_probs: Reference policy token log-probs, shape [B, T].
        attention_mask: 1 for real tokens, 0 for padding, shape [B, T].
        desirable: Boolean tensor of shape [B]; True iff the response is
            considered desirable/good for this prompt.
    """

    log_probs: Tensor        # [B, T]
    ref_log_probs: Tensor    # [B, T]
    attention_mask: Tensor   # [B, T]
    desirable: Tensor        # [B] bool


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class KTOTrainer:
    """Pure-PyTorch KTO trainer for binary desirability preference learning.

    Unlike DPO this trainer does not require paired (chosen, rejected) examples.
    Each response in the batch carries a scalar binary label (desirable=True/False)
    and the loss is computed independently per response with a shared KL anchor
    z_ref that grounds the implicit reward in the reference distribution.

    Usage::

        config  = KTOConfig(beta=0.1, lambda_u=1.0)
        trainer = KTOTrainer(config)
        result  = trainer.total_loss(batch)
        result["loss"].backward()
    """

    def __init__(self, config: KTOConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Core components
    # ------------------------------------------------------------------

    def sequence_log_ratio(
        self,
        log_probs: Tensor,
        ref_log_probs: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Per-sequence mean log-ratio between current and reference policy.

        Computes mean( (log_probs - ref_log_probs) * mask, dim=1 ) while
        correctly handling variable-length sequences via the attention mask.

        Args:
            log_probs: [B, T] token log-probabilities under current policy.
            ref_log_probs: [B, T] token log-probabilities under reference policy.
            mask: [B, T] float/bool attention mask (1 = real token).

        Returns:
            Tensor of shape [B] — per-sequence mean log-ratio.
        """
        mask = mask.float()
        diff = (log_probs - ref_log_probs) * mask
        token_counts = mask.sum(dim=1).clamp(min=self.config.eps)
        return diff.sum(dim=1) / token_counts

    def estimate_kl(
        self,
        log_probs: Tensor,
        ref_log_probs: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Estimate the KL-divergence anchor z_ref.

        Uses a subset of at most kl_num_samples sequences from the batch to
        compute a Monte Carlo estimate of KL(π_θ || π_ref).  The result is
        clamped to be non-negative (KL is always ≥ 0).

        Args:
            log_probs: [B, T] current policy token log-probs.
            ref_log_probs: [B, T] reference policy token log-probs.
            mask: [B, T] attention mask.

        Returns:
            Scalar tensor — clamp(mean_log_ratio, min=0).
        """
        B = log_probs.shape[0]
        k = self.config.kl_num_samples
        if k > 0 and k < B:
            per_seq = self.sequence_log_ratio(log_probs[:k], ref_log_probs[:k], mask[:k])
        else:
            per_seq = self.sequence_log_ratio(log_probs, ref_log_probs, mask)
        return per_seq.mean().clamp(min=0.0)

    def desirable_loss(self, log_ratio: Tensor, z_ref: Tensor) -> Tensor:
        """Loss for desirable (good) responses.

        L = 1 - sigmoid(β · (log_ratio - z_ref))

        Args:
            log_ratio: [N] per-sequence log-ratios for desirable samples.
            z_ref: Scalar KL anchor.

        Returns:
            Scalar mean loss over desirable samples.
        """
        return (1.0 - torch.sigmoid(self.config.beta * (log_ratio - z_ref))).mean()

    def undesirable_loss(self, log_ratio: Tensor, z_ref: Tensor) -> Tensor:
        """Loss for undesirable (bad) responses.

        L = 1 - sigmoid(β · (z_ref - log_ratio))

        Args:
            log_ratio: [M] per-sequence log-ratios for undesirable samples.
            z_ref: Scalar KL anchor.

        Returns:
            Scalar mean loss over undesirable samples.
        """
        return (1.0 - torch.sigmoid(self.config.beta * (z_ref - log_ratio))).mean()

    # ------------------------------------------------------------------
    # Primary training interface
    # ------------------------------------------------------------------

    def total_loss(self, batch: KTOBatch) -> dict[str, Tensor]:
        """Compute the full KTO training loss for a batch.

        Steps:
            1. Compute per-sequence log-ratios [B].
            2. Estimate z_ref (KL anchor) from a subset of batch samples.
            3. Split batch by desirability label.
            4. Compute desirable_loss and undesirable_loss on their
               respective subsets.  Missing subsets contribute 0.
            5. Combine: loss = d_loss + λ_U · u_loss.

        Args:
            batch: KTOBatch with tensors of shape [B, T] and desirable [B].

        Returns:
            Dict with keys:
                "loss"             — total scalar training loss
                "desirable_loss"   — desirable component (scalar)
                "undesirable_loss" — undesirable component (scalar)
                "z_ref"            — KL anchor estimate (scalar)
        """
        log_ratio = self.sequence_log_ratio(
            batch.log_probs, batch.ref_log_probs, batch.attention_mask
        )
        z_ref = self.estimate_kl(
            batch.log_probs, batch.ref_log_probs, batch.attention_mask
        )

        desirable_mask = batch.desirable.bool()
        undesirable_mask = ~desirable_mask

        has_desirable = desirable_mask.any().item()
        has_undesirable = undesirable_mask.any().item()

        zero = log_ratio.new_tensor(0.0)

        if has_desirable:
            d_loss = self.desirable_loss(log_ratio[desirable_mask], z_ref)
        else:
            d_loss = zero

        if has_undesirable:
            u_loss = self.undesirable_loss(log_ratio[undesirable_mask], z_ref)
        else:
            u_loss = zero

        loss = d_loss + self.config.lambda_u * u_loss

        return {
            "loss": loss,
            "desirable_loss": d_loss,
            "undesirable_loss": u_loss,
            "z_ref": z_ref,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def statistics(self, batch: KTOBatch) -> dict[str, float]:
        """Compute informative statistics for a batch (no gradient tracking).

        Returns:
            Dict with keys:
                "mean_log_ratio"         — mean over all sequences
                "z_ref"                  — KL anchor estimate
                "desirable_mean_ratio"   — mean log-ratio for desirable samples
                "undesirable_mean_ratio" — mean log-ratio for undesirable samples
        """
        with torch.no_grad():
            log_ratio = self.sequence_log_ratio(
                batch.log_probs, batch.ref_log_probs, batch.attention_mask
            )
            z_ref = self.estimate_kl(
                batch.log_probs, batch.ref_log_probs, batch.attention_mask
            )

            desirable_mask = batch.desirable.bool()
            undesirable_mask = ~desirable_mask

            d_ratios = log_ratio[desirable_mask] if desirable_mask.any() else log_ratio.new_tensor([0.0])
            u_ratios = log_ratio[undesirable_mask] if undesirable_mask.any() else log_ratio.new_tensor([0.0])

        return {
            "mean_log_ratio": log_ratio.mean().item(),
            "z_ref": z_ref.item(),
            "desirable_mean_ratio": d_ratios.mean().item(),
            "undesirable_mean_ratio": u_ratios.mean().item(),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.alignment import ALIGNMENT_REGISTRY  # noqa: E402

ALIGNMENT_REGISTRY["kto"] = KTOTrainer
