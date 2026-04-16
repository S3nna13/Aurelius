"""Aurelius — Self-Play Preference Optimization (SPPO).

Native PyTorch implementation of SPPO from:
  "Self-Play Preference Optimization for Language Model Alignment"
  Wu et al., 2024 — arXiv:2405.00675

SPPO frames alignment as a two-player constant-sum game where the policy
plays against itself (self-play). The Nash equilibrium of this game is the
optimal policy.

Algorithm 1 (SPPO-1, first-order approximation, eq. 11):
  Given preference dataset {(x, y_w, y_l)} where y_w ≻ y_l under π*,

  Win-probability approximation:
    p_hat(y_w, y_l | x) ≈ sigmoid(
        (log π(y_w|x) − log π_ref(y_w|x)
       − log π(y_l|x) + log π_ref(y_l|x)) / T
    )

  MLE loss (negative log-likelihood of preference):
    L_SPPO = −E[log p_hat(y_w, y_l | x)]
           = E[log(1 + exp(−diff / T))]
           = −E[logsigmoid(diff / T)]

  where diff = (log π(y_w|x) − log π_ref(y_w|x))
                − (log π(y_l|x) − log π_ref(y_l|x))

  T is the temperature (default 0.1) that controls preference sharpness.

Difference from DPO (arXiv:2305.18290):
  - Self-play game framing → win-probability interpretation
  - Temperature T (SPPO) vs. β (DPO): controls preference signal strength
  - Supports iterative self-play update (Algorithm 1, step 4)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SPPOLoss
# ---------------------------------------------------------------------------

class SPPOLoss(nn.Module):
    """SPPO loss (eq. 11, Wu et al. 2024).

    Parameters
    ----------
    T : float
        Temperature controlling the sharpness of the preference signal.
        Default 0.1 matches the paper's reported best hyper-parameter.
    """

    def __init__(self, T: float = 0.1) -> None:
        super().__init__()
        if T <= 0:
            raise ValueError(f"Temperature T must be > 0, got {T}")
        self.T = T

    def forward(
        self,
        log_probs_w: torch.Tensor,
        log_probs_l: torch.Tensor,
        ref_log_probs_w: torch.Tensor,
        ref_log_probs_l: torch.Tensor,
        T: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute SPPO loss.

        Parameters
        ----------
        log_probs_w : Tensor, shape (B,)
            Sequence log-probabilities log π(y_w | x) under the current policy.
        log_probs_l : Tensor, shape (B,)
            Sequence log-probabilities log π(y_l | x) under the current policy.
        ref_log_probs_w : Tensor, shape (B,)
            Sequence log-probabilities log π_ref(y_w | x) under the reference.
        ref_log_probs_l : Tensor, shape (B,)
            Sequence log-probabilities log π_ref(y_l | x) under the reference.
        T : float, optional
            Override instance temperature for this forward pass.

        Returns
        -------
        loss : scalar Tensor
            SPPO loss −E[logsigmoid(diff / T)].
        metrics : dict
            'win_prob'      — mean p_hat(y_w, y_l | x), in [0, 1]
            'reward_margin' — mean diff (= mean implicit reward gap)
        """
        temperature = T if T is not None else self.T

        # Implicit reward for chosen and rejected (eq. 11 numerator terms)
        r_w = log_probs_w - ref_log_probs_w   # log π(y_w) − log π_ref(y_w)
        r_l = log_probs_l - ref_log_probs_l   # log π(y_l) − log π_ref(y_l)

        # Reward margin (diff in paper notation)
        diff = r_w - r_l                       # shape (B,)

        # Win-probability approximation (eq. 11)
        # p_hat = σ(diff / T)
        win_prob = torch.sigmoid(diff / temperature)  # shape (B,)

        # MLE loss: −E[log p_hat] = −E[logsigmoid(diff / T)]
        loss = -F.logsigmoid(diff / temperature).mean()

        metrics: Dict[str, torch.Tensor] = {
            "win_prob": win_prob.mean().detach(),
            "reward_margin": diff.mean().detach(),
        }
        return loss, metrics

    # Convenience alias matching the call signature in the docstring
    def __call__(
        self,
        log_probs_w: torch.Tensor,
        log_probs_l: torch.Tensor,
        ref_log_probs_w: torch.Tensor,
        ref_log_probs_l: torch.Tensor,
        T: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return super().__call__(
            log_probs_w, log_probs_l, ref_log_probs_w, ref_log_probs_l, T
        )


# ---------------------------------------------------------------------------
# SPPOTrainer
# ---------------------------------------------------------------------------

class SPPOTrainer:
    """Minimal SPPO trainer wrapper for integration with the Aurelius pipeline.

    Wraps ``SPPOLoss`` and expects batches with keys:
        'log_probs_w'     — log π(y_w | x) under policy
        'log_probs_l'     — log π(y_l | x) under policy
        'ref_log_probs_w' — log π_ref(y_w | x)
        'ref_log_probs_l' — log π_ref(y_l | x)

    Optionally the batch may supply 'T' (per-batch temperature override).

    Parameters
    ----------
    T : float
        Default temperature for the SPPO loss.
    model : nn.Module, optional
        Policy model. Not used in loss computation itself (callers supply
        pre-computed log-probs) but kept for future iterative self-play updates
        (Algorithm 1, step 4).
    ref_model : nn.Module, optional
        Frozen reference model. Same note as ``model``.
    """

    def __init__(
        self,
        T: float = 0.1,
        model: Optional[nn.Module] = None,
        ref_model: Optional[nn.Module] = None,
    ) -> None:
        self.T = T
        self.model = model
        self.ref_model = ref_model
        self.criterion = SPPOLoss(T=T)

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        T: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute SPPO loss from a pre-processed batch.

        Parameters
        ----------
        batch : dict
            Must contain 'log_probs_w', 'log_probs_l',
            'ref_log_probs_w', 'ref_log_probs_l'.
        T : float, optional
            Per-call temperature override.

        Returns
        -------
        loss : scalar Tensor
        """
        required = {"log_probs_w", "log_probs_l", "ref_log_probs_w", "ref_log_probs_l"}
        missing = required - batch.keys()
        if missing:
            raise KeyError(f"Batch missing required keys: {missing}")

        temperature = T if T is not None else self.T
        loss, _ = self.criterion(
            batch["log_probs_w"],
            batch["log_probs_l"],
            batch["ref_log_probs_w"],
            batch["ref_log_probs_l"],
            T=temperature,
        )
        return loss
