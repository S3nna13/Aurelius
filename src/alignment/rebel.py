"""Aurelius — REBEL: Reward-Based Optimization for Large Language Models.

Native PyTorch implementation of REBEL (Gao et al., 2024, arXiv:2404.16767).

REBEL frames alignment as regression of relative rewards rather than
classification (as in DPO).  Instead of maximising the log-likelihood of
a preference, we directly regress the log-ratio difference log(π/π_ref) to
match the scaled reward difference.

Reference: "REBEL: Reinforcement Learning via Regressing Relative Rewards",
           Gao et al., 2024, arXiv:2404.16767

Variable names follow the paper notation throughout (Section 3, Eq. 5).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

__all__ = ["REBELLoss", "REBELTrainer"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core loss
# ---------------------------------------------------------------------------


class REBELLoss(nn.Module):
    """REBEL loss (Eq. 5, Gao et al. 2024).

    Given a preference triple (x, y_w, y_l) with scalar rewards
    r_w = R(x, y_w) and r_l = R(x, y_l), define:

        Δ_w = log π(y_w|x) - log π_ref(y_w|x)
        Δ_l = log π(y_l|x) - log π_ref(y_l|x)
        Δ   = Δ_w - Δ_l                          [pairwise log-ratio diff]
        target = (r_w - r_l) / β                 [scaled reward difference]

    REBEL loss (no regularisation):
        L = (Δ - target)²

    Regularised REBEL (λ > 0):
        L = (Δ - target)² + λ * (Δ_w² + Δ_l²)

    Args:
        beta:       Temperature parameter β > 0 that scales the reward signal.
        lambda_reg: Regularisation coefficient λ ≥ 0 that penalises large
                    deviations from the reference policy.
    """

    def __init__(self, beta: float = 0.1, lambda_reg: float = 0.0) -> None:
        super().__init__()
        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        if lambda_reg < 0.0:
            raise ValueError(f"lambda_reg must be non-negative, got {lambda_reg}")
        self.beta = beta
        self.lambda_reg = lambda_reg

    # ------------------------------------------------------------------
    def forward(
        self,
        log_probs_w: torch.Tensor,
        log_probs_l: torch.Tensor,
        ref_log_probs_w: torch.Tensor,
        ref_log_probs_l: torch.Tensor,
        reward_w: torch.Tensor,
        reward_l: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the REBEL loss.

        All tensor arguments must be 1-D (batch,) or scalar.

        Args:
            log_probs_w:     log π(y_w|x)     — policy log-prob for winner.
            log_probs_l:     log π(y_l|x)     — policy log-prob for loser.
            ref_log_probs_w: log π_ref(y_w|x) — reference log-prob for winner.
            ref_log_probs_l: log π_ref(y_l|x) — reference log-prob for loser.
            reward_w:        r_w = R(x, y_w)  — scalar reward for winner.
            reward_l:        r_l = R(x, y_l)  — scalar reward for loser.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a
            dict with keys: 'delta', 'target', 'regression_error',
            'reward_margin'.
        """
        # ---- shape guard -----------------------------------------------
        tensors = (log_probs_w, log_probs_l, ref_log_probs_w, ref_log_probs_l, reward_w, reward_l)
        shapes = {t.shape for t in tensors}
        if len(shapes) > 1:
            raise ValueError(
                f"All inputs must share the same shape; got shapes: "
                f"{[tuple(t.shape) for t in tensors]}"
            )

        # ---- REBEL core (Eq. 5) ----------------------------------------
        # Δ_w, Δ_l  (per-sample log-ratio deviations)
        Delta_w: torch.Tensor = log_probs_w - ref_log_probs_w  # Δ_w
        Delta_l: torch.Tensor = log_probs_l - ref_log_probs_l  # Δ_l

        # Δ = Δ_w - Δ_l   (pairwise log-ratio difference)
        Delta: torch.Tensor = Delta_w - Delta_l

        # target = (r_w - r_l) / β
        reward_margin: torch.Tensor = reward_w - reward_l
        target: torch.Tensor = reward_margin / self.beta

        # regression error per sample
        regression_error: torch.Tensor = Delta - target

        # MSE regression loss (mean over batch)
        loss: torch.Tensor = (regression_error**2).mean()

        # ---- optional KL regularisation --------------------------------
        if self.lambda_reg != 0.0:
            reg = self.lambda_reg * (Delta_w**2 + Delta_l**2).mean()
            loss = loss + reg

        metrics: dict[str, torch.Tensor] = {
            "delta": Delta.detach(),
            "target": target.detach(),
            "regression_error": regression_error.detach(),
            "reward_margin": reward_margin.detach(),
        }
        return loss, metrics

    # convenience alias so loss_fn(…) works like nn.Module.__call__
    __call__ = nn.Module.__call__


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class REBELTrainer:
    """Minimal REBEL trainer that wraps :class:`REBELLoss`.

    Expected batch keys:
        - 'log_probs_w'     : log π(y_w|x)
        - 'log_probs_l'     : log π(y_l|x)
        - 'ref_log_probs_w' : log π_ref(y_w|x)
        - 'ref_log_probs_l' : log π_ref(y_l|x)
        - 'reward_w'        : R(x, y_w)
        - 'reward_l'        : R(x, y_l)
    """

    REQUIRED_KEYS = (
        "log_probs_w",
        "log_probs_l",
        "ref_log_probs_w",
        "ref_log_probs_l",
        "reward_w",
        "reward_l",
    )

    def __init__(self, beta: float = 0.1, lambda_reg: float = 0.0) -> None:
        self.loss_fn = REBELLoss(beta=beta, lambda_reg=lambda_reg)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute REBEL loss from a batch dict.

        Args:
            batch: dict containing the required keys (see class docstring).

        Returns:
            Scalar loss tensor.
        """
        missing = [k for k in self.REQUIRED_KEYS if k not in batch]
        if missing:
            raise KeyError(f"Batch is missing required keys: {missing}")

        loss, _ = self.loss_fn(
            log_probs_w=batch["log_probs_w"],
            log_probs_l=batch["log_probs_l"],
            ref_log_probs_w=batch["ref_log_probs_w"],
            ref_log_probs_l=batch["ref_log_probs_l"],
            reward_w=batch["reward_w"],
            reward_l=batch["reward_l"],
        )
        return loss
