"""Aurelius — DITTO: Direct Iterative Tailored Text Optimization.

Native PyTorch implementation of DITTO, framed as online DPO with an
iteratively updated reference policy:

  "DITTO: Alignment as Direct Preference Optimization with Online Policy
  Sampling" — DITTO is DPO where the reference policy is not frozen but
  chases the current policy via an exponential moving average (EMA) or
  hard copy, preventing over-optimization to a stale reference.

Algorithm (per iteration k):
  1. Generate responses online from current policy π_k.
  2. Score responses with a reward model → create (y_w, y_l) pairs.
  3. Update reference: π_ref_{k+1} ← EMA(π_ref_k, π_k) or hard copy.
  4. Train policy with DPO loss using updated π_ref_{k+1}.

DITTO Loss (identical form to DPO, reference just changes between iters):
  diff = β * ((lp_w − lp_ref_w) − (lp_l − lp_ref_l))
  loss = −mean(logsigmoid(diff))

Reference update (EMA / soft update):
  π_ref_params ← α * π_ref_params + (1 − α) * π_params

Key distinction from DPO: the reference is not frozen — it tracks the
current policy, preventing excessive KL divergence from a stale anchor.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DITTOLoss
# ---------------------------------------------------------------------------

class DITTOLoss(nn.Module):
    """DITTO alignment loss.

    Identical in form to the standard DPO loss, but the reference policy
    is iteratively updated externally (by ``DITTOReferenceUpdater``) rather
    than frozen throughout training.

    Parameters
    ----------
    beta : float
        KL-regularisation coefficient (same role as β in DPO). Default 0.1.
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.beta = beta

    def forward(
        self,
        log_probs_w: torch.Tensor,
        log_probs_l: torch.Tensor,
        ref_log_probs_w: torch.Tensor,
        ref_log_probs_l: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the DITTO loss.

        Parameters
        ----------
        log_probs_w : Tensor, shape (B,)
            Sequence log-probs log π(y_w | x) under the current policy.
        log_probs_l : Tensor, shape (B,)
            Sequence log-probs log π(y_l | x) under the current policy.
        ref_log_probs_w : Tensor, shape (B,)
            Sequence log-probs log π_ref(y_w | x) under the (updated) reference.
        ref_log_probs_l : Tensor, shape (B,)
            Sequence log-probs log π_ref(y_l | x) under the (updated) reference.

        Returns
        -------
        loss : scalar Tensor
            DITTO loss: −E[logsigmoid(β * ((lp_w − lp_ref_w) − (lp_l − lp_ref_l)))].
        metrics : dict
            'reward_margin'  — mean implicit reward gap (float).
            'reward_accuracy' — fraction of pairs where y_w is ranked higher (float).
        """
        # Implicit reward differences (policy log-ratio advantage)
        r_w = log_probs_w - ref_log_probs_w   # shape (B,)
        r_l = log_probs_l - ref_log_probs_l   # shape (B,)

        # Reward margin
        diff = self.beta * (r_w - r_l)        # shape (B,)

        # DITTO / DPO loss
        loss = -F.logsigmoid(diff).mean()

        reward_margin = diff.mean().detach().item()
        reward_accuracy = (diff > 0).float().mean().detach().item()

        metrics: Dict[str, float] = {
            "reward_margin": reward_margin,
            "reward_accuracy": reward_accuracy,
        }
        return loss, metrics

    # Convenience alias so callers can use instance as a callable directly.
    def __call__(
        self,
        log_probs_w: torch.Tensor,
        log_probs_l: torch.Tensor,
        ref_log_probs_w: torch.Tensor,
        ref_log_probs_l: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return super().__call__(log_probs_w, log_probs_l, ref_log_probs_w, ref_log_probs_l)


# ---------------------------------------------------------------------------
# DITTOReferenceUpdater
# ---------------------------------------------------------------------------

class DITTOReferenceUpdater:
    """Manages iterative reference-policy updates for DITTO.

    Two update modes:
    - **Soft / EMA update** (default): blends current reference with policy.
        π_ref_params ← α * π_ref_params + (1 − α) * π_params
      With α = 1.0 the reference never changes; α = 0.0 copies the policy
      completely (equivalent to a hard update every step).

    - **Hard update**: copies all policy parameters verbatim into ref_model.

    The reference model's parameters are expected to have ``requires_grad=False``
    (frozen for gradient computation). Both update methods preserve this.

    Parameters
    ----------
    alpha : float
        EMA decay coefficient in [0, 1]. Default 0.99 (slow chase).
    """

    def __init__(self, alpha: float = 0.99) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    @torch.no_grad()
    def update(self, policy_model: nn.Module, ref_model: nn.Module) -> None:
        """Soft EMA update: ref ← α * ref + (1 − α) * policy.

        Parameters
        ----------
        policy_model : nn.Module
            Current policy being trained.
        ref_model : nn.Module
            Reference model whose parameters are updated in-place.
        """
        for ref_param, pol_param in zip(
            ref_model.parameters(), policy_model.parameters()
        ):
            ref_param.data.mul_(self.alpha).add_(pol_param.data, alpha=1.0 - self.alpha)

    @torch.no_grad()
    def hard_update(self, policy_model: nn.Module, ref_model: nn.Module) -> None:
        """Hard copy: ref ← policy (exact copy of all parameters).

        Parameters
        ----------
        policy_model : nn.Module
            Current policy being trained.
        ref_model : nn.Module
            Reference model whose parameters are overwritten in-place.
        """
        for ref_param, pol_param in zip(
            ref_model.parameters(), policy_model.parameters()
        ):
            ref_param.data.copy_(pol_param.data)


# ---------------------------------------------------------------------------
# DITTOTrainer
# ---------------------------------------------------------------------------

class DITTOTrainer:
    """Minimal DITTO trainer that wires together loss and reference updates.

    Expects pre-computed log-probabilities in each batch (callers supply
    forward passes from policy / reference models externally, keeping this
    class model-architecture-agnostic).

    Batch dict keys expected by ``compute_loss``:
        'log_probs_w'     — log π(y_w | x) under current policy, shape (B,)
        'log_probs_l'     — log π(y_l | x) under current policy, shape (B,)
        'ref_log_probs_w' — log π_ref(y_w | x), shape (B,)
        'ref_log_probs_l' — log π_ref(y_l | x), shape (B,)

    Parameters
    ----------
    beta : float
        KL-regularisation coefficient for the DITTO loss.
    alpha : float
        EMA decay coefficient for the reference updater.
    """

    def __init__(
        self,
        beta: float = 0.1,
        alpha: float = 0.99,
    ) -> None:
        self.criterion = DITTOLoss(beta=beta)
        self.updater = DITTOReferenceUpdater(alpha=alpha)

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute DITTO loss from a pre-processed batch.

        Parameters
        ----------
        batch : dict
            Must contain 'log_probs_w', 'log_probs_l',
            'ref_log_probs_w', 'ref_log_probs_l'.

        Returns
        -------
        loss : scalar Tensor
        """
        required = {"log_probs_w", "log_probs_l", "ref_log_probs_w", "ref_log_probs_l"}
        missing = required - batch.keys()
        if missing:
            raise KeyError(f"Batch missing required keys: {missing}")

        loss, _ = self.criterion(
            batch["log_probs_w"],
            batch["log_probs_l"],
            batch["ref_log_probs_w"],
            batch["ref_log_probs_l"],
        )
        return loss

    def update_reference(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        step: int,
        hard_update_interval: Optional[int] = None,
    ) -> None:
        """Update the reference model.

        If ``hard_update_interval`` is provided and ``step`` is a multiple
        of it, performs a hard copy; otherwise performs the EMA soft update.

        Parameters
        ----------
        policy_model : nn.Module
            The current policy model.
        ref_model : nn.Module
            The reference model to update in-place.
        step : int
            Current training step (used with hard_update_interval).
        hard_update_interval : int, optional
            If set, hard-copy the policy into ref every this many steps.
            When None (default), always use EMA soft update.
        """
        if hard_update_interval is not None and step % hard_update_interval == 0:
            self.updater.hard_update(policy_model, ref_model)
        else:
            self.updater.update(policy_model, ref_model)
