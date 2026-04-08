"""Reward model ensemble for more reliable RLHF scoring.

Combines multiple RewardModel instances and supports mean, min, and geometric
mean (product) aggregation. Also provides uncertainty estimation via std across
ensemble members.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.alignment.reward_model import RewardModel


@dataclass
class EnsembleConfig:
    """Configuration for RewardEnsemble."""
    n_members: int = 3
    aggregation: str = "mean"          # "mean" | "min" | "product"
    uncertainty_threshold: float = 0.5  # flag high-uncertainty samples


class RewardEnsemble(nn.Module):
    """An ensemble of RewardModel instances for robust RLHF reward scoring.

    Members are stored in an ``nn.ModuleList`` so that ``parameters()``,
    ``train()`` / ``eval()``, and ``state_dict()`` all propagate correctly.

    Args:
        reward_models: List of :class:`~src.alignment.reward_model.RewardModel`
            instances (or any ``nn.Module`` whose ``forward`` accepts
            ``input_ids`` and returns a ``(B,)`` tensor).
    """

    def __init__(self, reward_models: list[RewardModel]) -> None:
        super().__init__()
        if not reward_models:
            raise ValueError("reward_models must be a non-empty list")
        self.members = nn.ModuleList(reward_models)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_scores(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run all members and stack their scores.

        Returns:
            Tensor of shape ``(n_members, B)``.
        """
        member_scores = []
        for member in self.members:
            s = member.forward(input_ids)  # (B,)
            member_scores.append(s)
        return torch.stack(member_scores, dim=0)  # (n_members, B)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(self, scores: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """Aggregate per-member scores along the member dimension (dim 0).

        Args:
            scores: ``(n_members, B)`` tensor of member scores.
            mode:   One of ``"mean"``, ``"min"``, or ``"product"``.

        Returns:
            ``(B,)`` aggregated scores.
        """
        if mode == "mean":
            return scores.mean(dim=0)
        elif mode == "min":
            return scores.min(dim=0).values
        elif mode == "product":
            # Geometric mean via log-space to avoid underflow.
            # Reward scores can be negative, so clamp to a small positive value
            # before taking the logarithm.
            #   exp( mean( log( clamp(scores, min=eps) ) ) )
            eps = 1e-8
            return torch.exp(torch.log(scores.clamp(min=eps)).mean(dim=0))
        else:
            raise ValueError(
                f"Unknown aggregation mode '{mode}'. "
                "Choose from 'mean', 'min', 'product'."
            )

    def score(
        self,
        input_ids: torch.Tensor,
        mode: str = "mean",
    ) -> torch.Tensor:
        """Score a batch of sequences.

        Args:
            input_ids: ``(B, seq_len)`` token ids.
            mode:       Aggregation mode — ``"mean"`` (default), ``"min"``,
                        or ``"product"``.

        Returns:
            ``(B,)`` scalar reward per item.
        """
        stacked = self._collect_scores(input_ids)   # (n_members, B)
        return self.aggregate(stacked, mode=mode)   # (B,)

    def score_with_uncertainty(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a batch and return mean + std across ensemble members.

        For a single-member ensemble the std is exactly 0.

        Args:
            input_ids: ``(B, seq_len)`` token ids.

        Returns:
            Tuple ``(mean_scores, std_scores)``, each ``(B,)``.
        """
        stacked = self._collect_scores(input_ids)   # (n_members, B)
        mean = stacked.mean(dim=0)                  # (B,)
        # torch.std with correction=0 to match population std; for N=1 this is 0
        if stacked.shape[0] == 1:
            std = torch.zeros_like(mean)
        else:
            std = stacked.std(dim=0, correction=0)  # (B,)
        return mean, std


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def filter_by_uncertainty(
    ensemble: RewardEnsemble,
    input_ids: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter batch items by uncertainty.

    Samples whose ensemble std is below *threshold* are considered "safe".

    Args:
        ensemble:   A :class:`RewardEnsemble` instance.
        input_ids:  ``(B, seq_len)`` token ids.
        threshold:  Uncertainty (std) threshold.

    Returns:
        ``(safe_mask, scores)``
        - ``safe_mask``: bool ``(B,)`` tensor — ``True`` where std < threshold.
        - ``scores``:    ``(B,)`` mean reward scores.
    """
    mean, std = ensemble.score_with_uncertainty(input_ids)
    safe_mask = std < threshold
    return safe_mask, mean
