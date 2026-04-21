"""Token-Level Credit Assignment from Response-Level Reward.

When only a response-level scalar reward is available (no per-step PRM),
credit must be distributed across the T tokens of a sequence.  This module
implements four strategies:

1. **Uniform** — every valid token receives ``reward / T_valid``.

2. **Gamma-discounted** — earlier tokens receive less credit because they are
   further from the reward signal:
   ``credit_t = γ^(T_valid - t - 1) * reward``

3. **GAE (Generalised Advantage Estimation)** — uses a value function V(s_t)
   to compute per-token TD errors and then folds them with the λ-return:
   ``δ_t = r_t + γ V_{t+1} - V_t``
   ``A_t = Σ_{k≥0} (γλ)^k δ_{t+k}``
   where r_t = 0 for all t < T_valid-1 and r_{T_valid-1} = reward.

4. **End-decay** — credit concentrates near the last token:
   ``credit_t = reward * end_decay^(T_valid - 1 - t)``

After computing raw credits, the assigner can optionally z-score normalise
the non-masked values so that the mean is ≈ 0 and std ≈ 1, which stabilises
RL gradient magnitudes.

Pure native PyTorch only; no external dependencies beyond stdlib + torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# TokenCreditConfig
# ---------------------------------------------------------------------------

@dataclass
class TokenCreditConfig:
    """Configuration for :class:`TokenCreditAssigner`.

    Attributes:
        method:    One of ``"uniform"``, ``"discounted"``, ``"gae"``,
                   ``"end_decay"``.
        gamma:     Discount factor γ ∈ (0, 1].
        lam:       GAE lambda λ ∈ [0, 1].
        end_decay: Decay rate for the ``end_decay`` method.
        normalize: If True, z-score normalise the final credits over all
                   non-masked positions (zero-mean, unit-std).
        eps:       Small constant to avoid division by zero.
    """

    method: str = "gae"
    gamma: float = 0.99
    lam: float = 0.95
    end_decay: float = 0.9
    normalize: bool = True
    eps: float = 1e-8


# ---------------------------------------------------------------------------
# TokenCreditAssigner
# ---------------------------------------------------------------------------

class TokenCreditAssigner:
    """Distribute a response-level reward over individual tokens.

    Args:
        config: A :class:`TokenCreditConfig` instance.
    """

    def __init__(self, config: TokenCreditConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public assignment methods
    # ------------------------------------------------------------------

    def uniform(self, rewards: Tensor, mask: Tensor) -> Tensor:
        """Assign equal credit to every valid token.

        Args:
            rewards: ``[B]`` scalar reward per sequence.
            mask:    ``[B, T]`` float/bool mask (1 = valid, 0 = padding).

        Returns:
            ``[B, T]`` credits; masked positions are 0.
        """
        mask = mask.float()
        T_valid = mask.sum(dim=1, keepdim=True).clamp(min=self.config.eps)  # [B, 1]
        # Broadcast reward over time dimension
        credits = (rewards.unsqueeze(1) / T_valid) * mask              # [B, T]
        return credits

    def discounted(self, rewards: Tensor, mask: Tensor) -> Tensor:
        """Gamma-discounted credit: earlier tokens receive less credit.

        ``credit_t = γ^(T_valid - t - 1) * reward``

        Args:
            rewards: ``[B]`` scalar reward per sequence.
            mask:    ``[B, T]`` float/bool mask.

        Returns:
            ``[B, T]`` credits; masked positions are 0.
        """
        mask = mask.float()
        B, T = mask.shape
        gamma = self.config.gamma

        # t indices: 0..T-1
        t_idx = torch.arange(T, device=mask.device, dtype=torch.float32)  # [T]
        T_valid = mask.sum(dim=1, keepdim=True)  # [B, 1]

        # exponent = T_valid - t - 1, clamped to ≥ 0 so padded positions don't
        # produce negative exponents that would give values > 1.
        exponent = (T_valid - t_idx.unsqueeze(0) - 1).clamp(min=0.0)  # [B, T]
        discount = gamma ** exponent                                    # [B, T]

        credits = rewards.unsqueeze(1) * discount * mask               # [B, T]
        return credits

    def end_decay(self, rewards: Tensor, mask: Tensor) -> Tensor:
        """Decay credit from the last token backward.

        ``credit_t = reward * end_decay^(T_valid - 1 - t)``

        The last valid token receives the full ``reward``; earlier tokens
        receive exponentially less.

        Args:
            rewards: ``[B]`` scalar reward per sequence.
            mask:    ``[B, T]`` float/bool mask.

        Returns:
            ``[B, T]`` credits; masked positions are 0.
        """
        mask = mask.float()
        B, T = mask.shape
        rate = self.config.end_decay

        t_idx = torch.arange(T, device=mask.device, dtype=torch.float32)  # [T]
        T_valid = mask.sum(dim=1, keepdim=True)  # [B, 1]

        exponent = (T_valid - 1 - t_idx.unsqueeze(0)).clamp(min=0.0)  # [B, T]
        decay_factor = rate ** exponent                                 # [B, T]

        credits = rewards.unsqueeze(1) * decay_factor * mask           # [B, T]
        return credits

    def gae(
        self,
        rewards: Tensor,
        values: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Generalised Advantage Estimation (GAE) per token.

        The immediate reward ``r_t = 0`` for all ``t < T_valid - 1`` and
        ``r_{T_valid-1} = reward``.  A backward scan accumulates:

        ``δ_t   = r_t + γ V_{t+1} - V_t``
        ``A_t   = δ_t + (γλ) A_{t+1}``

        Args:
            rewards: ``[B]`` scalar reward per sequence.
            values:  ``[B, T]`` value estimates V(s_t).
            mask:    ``[B, T]`` float/bool mask.

        Returns:
            ``[B, T]`` advantage estimates; masked positions are 0.
        """
        mask = mask.float()
        B, T = mask.shape
        gamma = self.config.gamma
        lam = self.config.lam
        gl = gamma * lam

        # Build per-token immediate rewards: 0 everywhere except last valid token.
        # Last valid index per row = sum(mask, dim=1) - 1
        T_valid_int = mask.sum(dim=1).long()  # [B]
        r_t = torch.zeros(B, T, device=rewards.device, dtype=rewards.dtype)
        for b in range(B):
            last = T_valid_int[b].item() - 1
            if last >= 0:
                r_t[b, int(last)] = rewards[b]

        # V_{T} = 0 (terminal state has no future value)
        # We append a zero column so that V_{t+1} for the last position = 0.
        values_ext = torch.cat(
            [values, torch.zeros(B, 1, device=values.device, dtype=values.dtype)],
            dim=1,
        )  # [B, T+1]

        # Backward scan
        advantages = torch.zeros(B, T, device=rewards.device, dtype=rewards.dtype)
        A_next = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

        for t in reversed(range(T)):
            valid_t = mask[:, t]  # [B]
            V_t = values[:, t]
            V_tp1 = values_ext[:, t + 1]
            delta = r_t[:, t] + gamma * V_tp1 - V_t         # [B]
            A_t = delta + gl * A_next                         # [B]
            # Zero out for masked positions; also reset A_next for masked rows
            A_t = A_t * valid_t
            advantages[:, t] = A_t
            # When a position is masked (padding), future A_next should be 0
            # because we've left the valid region.
            A_next = A_t * valid_t

        return advantages  # [B, T]

    # ------------------------------------------------------------------
    # Main dispatch + normalisation
    # ------------------------------------------------------------------

    def assign(
        self,
        rewards: Tensor,
        mask: Tensor,
        values: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute per-token credits and optionally normalise.

        Args:
            rewards: ``[B]`` scalar reward per sequence.
            mask:    ``[B, T]`` float/bool mask (1 = valid, 0 = padding).
            values:  ``[B, T]`` value estimates; required when
                     ``config.method == "gae"``.

        Returns:
            ``[B, T]`` per-token credits.

        Raises:
            ValueError: If ``method`` is unknown or ``values`` is missing for
                        GAE.
        """
        method = self.config.method

        if method == "uniform":
            credits = self.uniform(rewards, mask)
        elif method == "discounted":
            credits = self.discounted(rewards, mask)
        elif method == "end_decay":
            credits = self.end_decay(rewards, mask)
        elif method == "gae":
            if values is None:
                raise ValueError(
                    "TokenCreditAssigner.assign: 'values' tensor is required "
                    "for method='gae'."
                )
            credits = self.gae(rewards, values, mask)
        else:
            raise ValueError(
                f"TokenCreditAssigner: unknown method '{method}'. "
                "Choose from 'uniform', 'discounted', 'gae', 'end_decay'."
            )

        if self.config.normalize:
            credits = self._normalize(credits, mask)

        return credits

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self, credits: Tensor, mask: Tensor) -> dict:
        """Compute summary statistics over non-masked credit positions.

        Args:
            credits: ``[B, T]`` per-token credits.
            mask:    ``[B, T]`` float/bool mask.

        Returns:
            Dict with keys ``mean``, ``std``, ``min``, ``max``.
        """
        mask_f = mask.float()
        valid = credits[mask_f.bool()]
        if valid.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(valid.mean().item()),
            "std":  float(valid.std().item()) if valid.numel() > 1 else 0.0,
            "min":  float(valid.min().item()),
            "max":  float(valid.max().item()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, credits: Tensor, mask: Tensor) -> Tensor:
        """Z-score normalise credits over all non-masked positions."""
        mask_f = mask.float()
        valid = credits[mask_f.bool()]
        if valid.numel() == 0:
            return credits
        mean = valid.mean()
        std = valid.std() if valid.numel() > 1 else torch.zeros(1, device=credits.device, dtype=credits.dtype).squeeze()
        credits = (credits - mean) / (std + self.config.eps)
        # Re-zero masked positions that may have been shifted by the mean.
        credits = credits * mask_f
        return credits


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.training import TRAINING_REGISTRY  # noqa: E402

TRAINING_REGISTRY["token_credit"] = TokenCreditAssigner

__all__ = ["TokenCreditConfig", "TokenCreditAssigner"]
