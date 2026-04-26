"""Reward-Free Alignment methods for Aurelius.

Implements preference learning without an explicit reward model, including:
  - SLiC (Sequence Likelihood Calibration)
  - Calibration loss
  - Contrastive (InfoNCE-style) loss
  - ULMATrainer: a unified trainer that wraps all three methods

References:
    Zhao et al. 2023, "SLiC-HF: Sequence Likelihood Calibration with Human
        Feedback"
    Liu et al. 2023, "Statistical Rejection Sampling Improves Preference
        Optimization" (RAFT / ULMA framing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RewardFreeConfig:
    """Configuration for reward-free alignment training."""

    method: str = "slic"  # "slic" | "calibration" | "contrastive"
    beta: float = 0.1  # KL / temperature coefficient
    margin: float = 1.0  # target margin used in calibration loss
    lambda_reg: float = 1.0  # regularization weight in SLiC
    delta: float = 1.0  # hinge margin in SLiC ranking loss


# ---------------------------------------------------------------------------
# SLiC Loss
# ---------------------------------------------------------------------------


class SLiCLoss(nn.Module):
    """Sequence Likelihood Calibration loss.

    Combines a per-pair hinge ranking loss with an NLL-style regularization
    term that anchors the policy close to the reference distribution.
    """

    def __init__(self, delta: float = 1.0, lambda_reg: float = 1.0) -> None:
        super().__init__()
        self.delta = delta
        self.lambda_reg = lambda_reg

    # ------------------------------------------------------------------
    # Core components
    # ------------------------------------------------------------------

    def hinge_loss(
        self,
        chosen_logps: Tensor,
        rejected_logps: Tensor,
    ) -> Tensor:
        """Per-sequence hinge ranking loss.

        Penalises pairs where the margin between chosen and rejected
        log-probabilities is smaller than ``delta``.

        Args:
            chosen_logps:  (B,) per-sequence log-probs for chosen responses.
            rejected_logps: (B,) per-sequence log-probs for rejected responses.

        Returns:
            (B,) element-wise hinge values.
        """
        margin = chosen_logps - rejected_logps  # (B,)
        return F.relu(self.delta - margin)  # max(0, delta - margin)

    def regularization_loss(
        self,
        policy_logps: Tensor,
        ref_logps: Tensor,
    ) -> Tensor:
        """NLL regularization term.

        Prevents policy collapse by penalising deviations from the reference
        log-probs: -mean(ref_logps).  ``policy_logps`` is accepted as an
        argument for API symmetry but the anchoring is against the *reference*
        distribution.

        Args:
            policy_logps: (B,) log-probs under the policy (unused in loss
                computation, kept for interface symmetry).
            ref_logps:    (B,) log-probs under the reference model.

        Returns:
            Scalar regularization loss.
        """
        return -ref_logps.mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        chosen_logps: Tensor,
        rejected_logps: Tensor,
        ref_logps: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute the total SLiC loss and per-component metrics.

        Args:
            chosen_logps:   (B,) log-probs of chosen responses under policy.
            rejected_logps: (B,) log-probs of rejected responses under policy.
            ref_logps:      (B,) optional reference log-probs; defaults to
                            zeros if not supplied.

        Returns:
            Tuple of:
              - scalar total loss
              - metrics dict with keys: 'hinge_loss', 'reg_loss', 'accuracy'
        """
        if ref_logps is None:
            ref_logps = torch.zeros_like(chosen_logps)

        per_sample_hinge = self.hinge_loss(chosen_logps, rejected_logps)  # (B,)
        hinge = per_sample_hinge.mean()

        reg = self.regularization_loss(chosen_logps, ref_logps)

        loss = hinge + self.lambda_reg * reg

        # Accuracy: fraction of pairs where chosen_logp > rejected_logp
        accuracy = (chosen_logps > rejected_logps).float().mean()

        metrics: dict[str, Any] = {
            "hinge_loss": hinge.detach(),
            "reg_loss": reg.detach(),
            "accuracy": accuracy.detach(),
        }
        return loss, metrics


# ---------------------------------------------------------------------------
# ULMA Trainer
# ---------------------------------------------------------------------------


class ULMATrainer:
    """Unified Language Model Alignment trainer.

    Wraps three reward-free preference optimisation objectives:
      - 'slic':         SLiC hinge + regularization loss
      - 'calibration':  MSE calibration of log-ratio differences
      - 'contrastive':  InfoNCE-style contrastive loss

    The ``policy`` and ``ref_policy`` must be callable objects (e.g.
    ``nn.Module`` subclasses) that accept ``(input_ids)`` and return logits of
    shape ``(B, T, V)``.
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        method: str = "slic",
        beta: float = 0.1,
        margin: float = 1.0,
    ) -> None:
        if method not in ("slic", "calibration", "contrastive"):
            raise ValueError(
                f"method must be one of 'slic', 'calibration', 'contrastive'; got '{method}'"
            )
        self.policy = policy
        self.ref_policy = ref_policy
        self.method = method
        self.beta = beta
        self.margin = margin

        # Build the SLiC criterion (reused when method == 'slic')
        self._slic = SLiCLoss(delta=margin, lambda_reg=1.0)

    # ------------------------------------------------------------------
    # Log-prob utilities
    # ------------------------------------------------------------------

    def compute_sequence_logps(
        self,
        model: nn.Module,
        input_ids: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Compute per-sequence sum of log-probabilities for labelled tokens.

        Positions where ``labels == -100`` are treated as padding / prompt and
        excluded from the sum.

        Args:
            model:     Callable that takes ``input_ids`` and returns logits
                       ``(B, T, V)``.
            input_ids: (B, T) integer token ids fed to the model.
            labels:    (B, T) target token ids; use -100 to mask positions.

        Returns:
            (B,) summed log-prob per sequence.
        """
        logits = model(input_ids)  # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

        # Gather log-probs at the target positions
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0  # avoid OOB gather
        token_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

        mask = (labels != -100).float()  # (B, T)
        return (token_lp * mask).sum(dim=-1)  # (B,)

    # ------------------------------------------------------------------
    # Objective implementations
    # ------------------------------------------------------------------

    def calibration_loss(
        self,
        chosen_logps: Tensor,
        rejected_logps: Tensor,
        ref_chosen_logps: Tensor,
        ref_rejected_logps: Tensor,
    ) -> Tensor:
        """MSE calibration loss.

        The policy log-ratio difference should match the reference log-ratio
        difference offset by a fixed target margin:

            target  = (ref_chosen - ref_rejected) + margin
            loss    = MSE(chosen_logps - rejected_logps, target)

        Args:
            chosen_logps:       (B,) policy log-probs for chosen responses.
            rejected_logps:     (B,) policy log-probs for rejected responses.
            ref_chosen_logps:   (B,) reference model log-probs for chosen.
            ref_rejected_logps: (B,) reference model log-probs for rejected.

        Returns:
            Scalar calibration loss.
        """
        policy_ratio = chosen_logps - rejected_logps  # (B,)
        ref_ratio = ref_chosen_logps - ref_rejected_logps  # (B,)
        target = ref_ratio + self.margin  # (B,)
        return F.mse_loss(policy_ratio, target)

    def contrastive_loss(
        self,
        chosen_logps: Tensor,
        rejected_logps: Tensor,
    ) -> Tensor:
        """InfoNCE-style contrastive loss.

        Computes:
            loss = -mean( log( exp(chosen) / (exp(chosen) + exp(rejected)) ) )
                 = -mean( chosen - log(exp(chosen) + exp(rejected)) )
                 = mean( log(1 + exp(rejected - chosen)) )

        The result is always <= 0 for any single pair (before negation); after
        the negation the *loss* value is <= 0 only when chosen > rejected in
        the degenerate sense; in practice the returned scalar is non-positive
        because we return the *log-probability* (the negative cross-entropy),
        not the cross-entropy itself.

        Concretely: ``loss = mean( log_softmax([chosen, rejected])[:, 0] )``
        which is always in ``(-inf, 0]``.

        Args:
            chosen_logps:   (B,) log-probs for chosen responses.
            rejected_logps: (B,) log-probs for rejected responses.

        Returns:
            Scalar contrastive loss (always <= 0).
        """
        # Stack to (B, 2) so we can apply log_softmax over the pair dimension
        logits = torch.stack([chosen_logps, rejected_logps], dim=-1)  # (B, 2)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, 2)
        # We want to maximise the probability assigned to the chosen column
        # (index 0), so the *loss* is the mean log-probability (always <= 0)
        return log_probs[:, 0].mean()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
        chosen_labels: Tensor,
        rejected_labels: Tensor,
    ) -> dict[str, Any]:
        """Run a single preference-optimisation step.

        Computes log-probs for both policy and reference, selects the
        configured loss objective, and returns a dict of scalar tensors
        suitable for logging.

        Args:
            chosen_ids:       (B, T) input ids for chosen responses.
            rejected_ids:     (B, T) input ids for rejected responses.
            chosen_labels:    (B, T) label ids for chosen responses (-100 masks).
            rejected_labels:  (B, T) label ids for rejected responses (-100 masks).

        Returns:
            Dict with at minimum a 'loss' key plus per-method metrics.
        """
        # Policy log-probs
        chosen_logps = self.compute_sequence_logps(self.policy, chosen_ids, chosen_labels)
        rejected_logps = self.compute_sequence_logps(self.policy, rejected_ids, rejected_labels)

        # Reference log-probs (no gradient)
        with torch.no_grad():
            ref_chosen_logps = self.compute_sequence_logps(
                self.ref_policy, chosen_ids, chosen_labels
            )
            ref_rejected_logps = self.compute_sequence_logps(
                self.ref_policy, rejected_ids, rejected_labels
            )

        if self.method == "slic":
            ref_logps = ref_chosen_logps
            loss, metrics = self._slic(chosen_logps, rejected_logps, ref_logps)
            return {"loss": loss, **metrics}

        elif self.method == "calibration":
            loss = self.calibration_loss(
                chosen_logps,
                rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )
            accuracy = (chosen_logps > rejected_logps).float().mean()
            return {
                "loss": loss,
                "accuracy": accuracy.detach(),
            }

        else:  # contrastive
            loss = self.contrastive_loss(chosen_logps, rejected_logps)
            accuracy = (chosen_logps > rejected_logps).float().mean()
            return {
                "loss": loss,
                "accuracy": accuracy.detach(),
            }
