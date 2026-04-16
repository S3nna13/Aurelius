"""Aurelius — Nash Mirror Descent (Nash-MD) alignment.

Native PyTorch implementation of Nash Learning from Human Feedback (Nash-MD).

Reference: Munos et al., "Nash Learning from Human Feedback", arXiv:2401.01335.

Variable notation follows the paper:
  π   — policy (current model)
  π_ref — reference policy (frozen)
  π_t — policy at iteration t
  η   — learning rate / step size
  β   — KL regularization coefficient
  P(y > y' | x) — preference probability (Bradley-Terry model)
  nash_w = P(y_w > y_l | x) = σ(r_w - r_l)
  nash_l = P(y_l > y_w | x) = 1 - nash_w
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nash-MD Loss
# ---------------------------------------------------------------------------


@dataclass
class NashMDMetrics:
    """Diagnostics returned alongside the scalar loss."""

    loss: torch.Tensor
    nash_w: torch.Tensor       # P(y_w > y_l | x) per sample — eq. 9 notation
    nash_l: torch.Tensor       # P(y_l > y_w | x) per sample
    weight_w: torch.Tensor     # nash_w - 0.5  (centred weight for y_w)
    weight_l: torch.Tensor     # nash_l - 0.5  (centred weight for y_l)
    loss_policy: torch.Tensor  # pure Nash policy term
    loss_kl: torch.Tensor      # KL regularisation term


class NashMDLoss:
    """Compute the Nash-MD loss (Algorithm 2, eq. 9, arXiv:2401.01335).

    Nash-MD update (mirror descent on the KL-regularised Nash objective):

        π_{t+1}(y|x)  ∝  π_t(y|x) · exp(η · E_{y'~π_t}[P(y > y' | x)])

    Practical preference-dataset form (eq. 9):

        L_NashMD(π) = -(weight_w · log π(y_w|x)
                        + weight_l · log π(y_l|x)).mean()
                     + β · KL(π ‖ π_ref)   [approximated on chosen side]

    where:
        nash_w   = σ(r_w - r_l)          # P(y_w > y_l | x)
        nash_l   = 1 - nash_w             # P(y_l > y_w | x)
        weight_w = nash_w - 0.5           # centred; positive when y_w preferred
        weight_l = nash_l - 0.5           # centred; negative when y_w preferred
    """

    def __call__(
        self,
        log_probs_w: torch.Tensor,
        log_probs_l: torch.Tensor,
        ref_log_probs_w: torch.Tensor,
        ref_log_probs_l: torch.Tensor,
        reward_w: torch.Tensor,
        reward_l: torch.Tensor,
        beta: float = 0.1,
    ) -> torch.Tensor:
        """Compute Nash-MD scalar loss.

        Args:
            log_probs_w:     log π(y_w | x)  shape (B,)
            log_probs_l:     log π(y_l | x)  shape (B,)
            ref_log_probs_w: log π_ref(y_w | x)  shape (B,)
            ref_log_probs_l: log π_ref(y_l | x)  shape (B,)
            reward_w:        scalar reward for y_w, shape (B,)
            reward_l:        scalar reward for y_l, shape (B,)
            beta:            KL regularisation coefficient β ≥ 0

        Returns:
            Scalar loss tensor (differentiable).
        """
        metrics = self.forward_with_metrics(
            log_probs_w=log_probs_w,
            log_probs_l=log_probs_l,
            ref_log_probs_w=ref_log_probs_w,
            ref_log_probs_l=ref_log_probs_l,
            reward_w=reward_w,
            reward_l=reward_l,
            beta=beta,
        )
        return metrics.loss

    def forward_with_metrics(
        self,
        log_probs_w: torch.Tensor,
        log_probs_l: torch.Tensor,
        ref_log_probs_w: torch.Tensor,
        ref_log_probs_l: torch.Tensor,
        reward_w: torch.Tensor,
        reward_l: torch.Tensor,
        beta: float = 0.1,
    ) -> NashMDMetrics:
        """Compute loss and return full diagnostic breakdown."""
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")

        # Ensure all tensors are 1-D (batch,) for consistency
        log_probs_w = log_probs_w.view(-1)
        log_probs_l = log_probs_l.view(-1)
        ref_log_probs_w = ref_log_probs_w.view(-1)
        ref_log_probs_l = ref_log_probs_l.view(-1)  # noqa: F841  (kept for API symmetry)
        reward_w = reward_w.view(-1)
        reward_l = reward_l.view(-1)

        # ── Nash preference weights (paper eq. 9) ──────────────────────────
        # nash_w = P(y_w > y_l | x) = σ(r_w - r_l)
        reward_diff = reward_w - reward_l           # (B,)
        nash_w = torch.sigmoid(reward_diff)         # P(y_w > y_l | x) ∈ (0,1)
        nash_l = 1.0 - nash_w                       # P(y_l > y_w | x)

        # Centred weights (subtract 0.5 so equal-reward pairs contribute 0)
        weight_w = nash_w - 0.5                     # > 0 when y_w better
        weight_l = nash_l - 0.5                     # < 0 when y_w better

        # ── Policy term ─────────────────────────────────────────────────────
        # L_policy = -(weight_w · log π(y_w) + weight_l · log π(y_l)).mean()
        loss_policy = -(weight_w * log_probs_w + weight_l * log_probs_l).mean()

        # ── KL regularisation ───────────────────────────────────────────────
        # KL(π ‖ π_ref) approximated on the chosen side (consistent with DPO literature)
        # KL ≈ (log π(y_w) - log π_ref(y_w)).mean()
        if beta > 0.0:
            loss_kl = beta * (log_probs_w - ref_log_probs_w).mean()
        else:
            loss_kl = torch.zeros((), dtype=log_probs_w.dtype, device=log_probs_w.device)

        loss = loss_policy + loss_kl

        return NashMDMetrics(
            loss=loss,
            nash_w=nash_w,
            nash_l=nash_l,
            weight_w=weight_w,
            weight_l=weight_l,
            loss_policy=loss_policy,
            loss_kl=loss_kl,
        )


# ---------------------------------------------------------------------------
# Nash-MD Trainer
# ---------------------------------------------------------------------------


class NashMDTrainer:
    """Lightweight Nash-MD training wrapper.

    Wraps :class:`NashMDLoss` and accepts a batch dict so it can be dropped
    into any training loop.

    Expected batch keys
    -------------------
    log_probs_w      : log π(y_w | x)        Tensor (B,)
    log_probs_l      : log π(y_l | x)        Tensor (B,)
    ref_log_probs_w  : log π_ref(y_w | x)    Tensor (B,)
    ref_log_probs_l  : log π_ref(y_l | x)    Tensor (B,)
    reward_w         : scalar reward for y_w  Tensor (B,)
    reward_l         : scalar reward for y_l  Tensor (B,)
    """

    _REQUIRED_KEYS = (
        "log_probs_w",
        "log_probs_l",
        "ref_log_probs_w",
        "ref_log_probs_l",
        "reward_w",
        "reward_l",
    )

    def __init__(self, beta: float = 0.1) -> None:
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        self.beta = beta
        self._loss_fn = NashMDLoss()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute Nash-MD loss from a batch dictionary.

        Args:
            batch: dict with keys listed in class docstring.

        Returns:
            Scalar loss tensor.
        """
        missing = [k for k in self._REQUIRED_KEYS if k not in batch]
        if missing:
            raise KeyError(f"NashMDTrainer.compute_loss: missing batch keys {missing}")

        return self._loss_fn(
            log_probs_w=batch["log_probs_w"],
            log_probs_l=batch["log_probs_l"],
            ref_log_probs_w=batch["ref_log_probs_w"],
            ref_log_probs_l=batch["ref_log_probs_l"],
            reward_w=batch["reward_w"],
            reward_l=batch["reward_l"],
            beta=self.beta,
        )
