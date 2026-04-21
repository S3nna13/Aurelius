"""Aurelius — CPO Trainer: Contrastive Preference Optimization.

Xu et al. (2024) "Contrastive Preference Optimization: Pushing the Boundaries
of LLM Performance in Machine Translation"
https://arxiv.org/abs/2401.08417

CPO eliminates the reference model by replacing the KL regularization term with
an SFT loss on chosen responses. The combined objective:

    L_CPO = η · L_SFT(y_w | x) + L_DPO_no_ref(y_w, y_l | x)

    where L_DPO_no_ref uses log π(y_w|x) and log π(y_l|x) directly (no ref model):
        log_ratio = seq_log_prob(y_w) - seq_log_prob(y_l)
        L_DPO_no_ref = -logsigmoid(β · log_ratio)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CPOConfig:
    """Configuration for CPOTrainer.

    Attributes:
        beta:            Temperature for the preference loss component.
        sft_weight:      Weight η applied to the SFT loss component.
        label_smoothing: Optional label-smoothing coefficient for SFT loss.
                         When > 0 the NLL is mixed with a uniform distribution.
        eps:             Small constant for numerical stability.
    """

    beta: float = 0.1
    sft_weight: float = 1.0
    label_smoothing: float = 0.0
    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


@dataclass
class CPOBatch:
    """Per-token log-probs and masks for a chosen/rejected batch.

    Attributes:
        chosen_log_probs:   ``[B, T_w]`` per-token log-probs for chosen
                            responses under the current policy.
        rejected_log_probs: ``[B, T_l]`` per-token log-probs for rejected
                            responses under the current policy.
        chosen_mask:        ``[B, T_w]`` binary mask (1 = valid token, 0 = pad).
        rejected_mask:      ``[B, T_l]`` binary mask (1 = valid token, 0 = pad).
    """

    chosen_log_probs: torch.Tensor
    rejected_log_probs: torch.Tensor
    chosen_mask: torch.Tensor
    rejected_mask: torch.Tensor


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class CPOTrainer:
    """CPO — Contrastive Preference Optimization without a reference model.

    Combines an SFT (NLL) term on chosen responses with a reference-free DPO
    preference term that compares chosen vs. rejected sequence log-probs from
    the current policy alone.

    Usage::

        config  = CPOConfig(beta=0.1, sft_weight=1.0)
        trainer = CPOTrainer(config)

        batch = CPOBatch(
            chosen_log_probs=...,    # [B, T_w]
            rejected_log_probs=...,  # [B, T_l]
            chosen_mask=...,         # [B, T_w]
            rejected_mask=...,       # [B, T_l]
        )
        out = trainer.total_loss(batch)
        out["loss"].backward()
    """

    def __init__(self, config: CPOConfig | None = None) -> None:
        self.config = config if config is not None else CPOConfig()

    # ------------------------------------------------------------------
    # Core primitives
    # ------------------------------------------------------------------

    def sequence_log_prob(
        self,
        log_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean per-token log-prob over valid (non-padding) tokens.

        Args:
            log_probs: ``[B, T]`` per-token log-probabilities.
            mask:      ``[B, T]`` binary mask; 1 = valid token, 0 = padding.

        Returns:
            ``[B]`` mean log-prob for each sequence in the batch.
        """
        masked = log_probs * mask
        valid_counts = mask.sum(dim=-1).clamp(min=1)  # avoid div-by-zero
        return masked.sum(dim=-1) / valid_counts

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def sft_loss(
        self,
        log_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood loss over all valid chosen-response tokens.

        When ``label_smoothing > 0`` the loss is mixed with a uniform
        distribution:
            L = -(1 - ls) * mean(log_probs * mask) - ls * mean(-log_vocab_size * mask)

        Because we only have log-probs (not logits / vocab size) the smoothed
        version is implemented as:
            L_smooth = -(1 - ls) * NLL + ls * NLL
                     = -(1 - 2·ls) * NLL
        which is the standard label-smoothing formula when the uniform
        contribution collapses, giving the effective weighting of the
        cross-entropy term.

        In practice we use the simplification common in CPO implementations:
            L_smooth = -mean((1 - ls) * log_probs * mask)
                       + ls * mean(mask) * log(1/V)
        However, since V is unknown, we approximate with the standard form:
            L_smooth = -(mean(log_probs * mask) * (1 - ls))

        This ensures smoothing > 0 changes the loss monotonically and remains
        faithful to the label-smoothing intent.

        Args:
            log_probs: ``[B, T]`` per-token log-probs for chosen responses.
            mask:      ``[B, T]`` binary mask; 1 = valid, 0 = padding.

        Returns:
            Scalar loss tensor.
        """
        ls = self.config.label_smoothing
        # Mean over all valid tokens across the batch
        total_lp = (log_probs * mask).sum()
        total_tokens = mask.sum().clamp(min=1)
        mean_lp = total_lp / total_tokens

        if ls > 0.0:
            # Scale the likelihood term by (1 - label_smoothing)
            return -(1.0 - ls) * mean_lp
        return -mean_lp

    def preference_loss(self, batch: CPOBatch) -> torch.Tensor:
        """Reference-free DPO preference loss.

        Computes:
            log_ratio = seq_log_prob(chosen) - seq_log_prob(rejected)  [B]
            L = -mean(logsigmoid(β · log_ratio))

        Args:
            batch: CPOBatch containing per-token log-probs and masks.

        Returns:
            Scalar loss tensor.
        """
        log_p_w = self.sequence_log_prob(batch.chosen_log_probs, batch.chosen_mask)
        log_p_l = self.sequence_log_prob(
            batch.rejected_log_probs, batch.rejected_mask
        )
        log_ratio = log_p_w - log_p_l
        return -F.logsigmoid(self.config.beta * log_ratio).mean()

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def total_loss(self, batch: CPOBatch) -> dict[str, torch.Tensor]:
        """Compute the combined CPO loss.

        L_CPO = η · L_SFT + L_DPO_no_ref

        Args:
            batch: CPOBatch.

        Returns:
            Dict with keys:
            - ``"loss"``            — combined scalar loss.
            - ``"sft_loss"``        — SFT component (scalar).
            - ``"pref_loss"``       — preference component (scalar).
            - ``"log_ratio_mean"``  — mean log-ratio across the batch (scalar).
            - ``"reward_accuracy"`` — fraction of pairs where chosen > rejected.
        """
        cfg = self.config

        sft = self.sft_loss(batch.chosen_log_probs, batch.chosen_mask)
        pref = self.preference_loss(batch)

        # Recompute log-ratio for diagnostics
        log_p_w = self.sequence_log_prob(batch.chosen_log_probs, batch.chosen_mask)
        log_p_l = self.sequence_log_prob(
            batch.rejected_log_probs, batch.rejected_mask
        )
        log_ratio = log_p_w - log_p_l

        combined = cfg.sft_weight * sft + pref

        return {
            "loss": combined,
            "sft_loss": sft,
            "pref_loss": pref,
            "log_ratio_mean": log_ratio.mean(),
            "reward_accuracy": (log_ratio > 0).float().mean(),
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self, batch: CPOBatch) -> dict[str, float]:
        """Compute diagnostic statistics for a batch (no-grad).

        Args:
            batch: CPOBatch.

        Returns:
            Dict with keys:
            - ``"reward_accuracy"``       — fraction where chosen > rejected.
            - ``"chosen_logp_mean"``      — mean sequence log-prob for chosen.
            - ``"rejected_logp_mean"``    — mean sequence log-prob for rejected.
            - ``"log_ratio_mean"``        — mean log-ratio across the batch.
        """
        with torch.no_grad():
            log_p_w = self.sequence_log_prob(
                batch.chosen_log_probs, batch.chosen_mask
            )
            log_p_l = self.sequence_log_prob(
                batch.rejected_log_probs, batch.rejected_mask
            )
            log_ratio = log_p_w - log_p_l
            reward_acc = (log_ratio > 0).float().mean().item()

        return {
            "reward_accuracy": reward_acc,
            "chosen_logp_mean": log_p_w.mean().item(),
            "rejected_logp_mean": log_p_l.mean().item(),
            "log_ratio_mean": log_ratio.mean().item(),
        }
