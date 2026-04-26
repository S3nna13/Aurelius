"""Hierarchical Reward Model with interpretable sub-criteria.

Decomposes reward into multiple learned sub-objectives, each corresponding
to a different quality dimension (helpfulness, safety, quality, etc.).
Aggregates via learned weights or explicit weighting schemes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RewardCriterion(nn.Module):
    """Single interpretable reward criterion.

    Maps a hidden state (B, d_model) to a scalar score (B,).
    """

    def __init__(self, name: str, d_model: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.name = name
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        """hidden: (B, d_model) -> score: (B,)"""
        return self.scorer(hidden).squeeze(-1)


class CriterionWeights(nn.Module):
    """Aggregation weights over criteria, optionally learnable.

    Weights are stored as raw (pre-softmax) values so they can be learned
    via gradient descent when ``learnable=True``.
    """

    def __init__(self, n_criteria: int, learnable: bool = False) -> None:
        super().__init__()
        if learnable:
            self.raw_weights = nn.Parameter(torch.ones(n_criteria))
        else:
            self.register_buffer("raw_weights", torch.ones(n_criteria))

    def get_weights(self) -> Tensor:
        """Returns softmax-normalised weights: (n_criteria,) summing to 1."""
        return F.softmax(self.raw_weights, dim=0)


class HierarchicalRewardModel(nn.Module):
    """Hierarchical reward model decomposing reward into sub-criteria.

    Args:
        d_model: Dimension of input hidden states.
        criteria: List of RewardCriterion modules.
        weight_scheme: CriterionWeights module for aggregation.
    """

    def __init__(
        self,
        d_model: int,
        criteria: list[RewardCriterion],
        weight_scheme: CriterionWeights,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.criteria = nn.ModuleList(criteria)
        self.weight_scheme = weight_scheme

    def forward(self, hidden: Tensor) -> dict[str, Tensor]:
        """Compute hierarchical reward.

        Args:
            hidden: (B, d_model)

        Returns:
            dict with keys:
                'total_reward'     : (B,)
                'criterion_scores' : dict mapping criterion name -> (B,)
                'weights'          : (n_criteria,)
        """
        weights = self.weight_scheme.get_weights()  # (n_criteria,)

        criterion_scores: dict[str, Tensor] = {}
        scores_list: list[Tensor] = []
        for criterion in self.criteria:
            score = criterion(hidden)  # (B,)
            criterion_scores[criterion.name] = score
            scores_list.append(score)

        # total_reward = sum_i(w_i * score_i), shape (B,)
        stacked = torch.stack(scores_list, dim=1)  # (B, n_criteria)
        total_reward = (stacked * weights.unsqueeze(0)).sum(dim=1)  # (B,)

        return {
            "total_reward": total_reward,
            "criterion_scores": criterion_scores,
            "weights": weights,
        }

    def criterion_names(self) -> list[str]:
        """Returns list of criterion names in order."""
        return [c.name for c in self.criteria]

    def breakdown(self, hidden: Tensor) -> dict[str, float]:
        """Returns mean score per criterion and total as a float dict.

        Args:
            hidden: (B, d_model)

        Returns:
            dict mapping criterion name -> mean score (float), plus 'total'.
        """
        out = self.forward(hidden)
        result: dict[str, float] = {}
        for name, score in out["criterion_scores"].items():
            result[name] = score.mean().item()
        result["total"] = out["total_reward"].mean().item()
        return result


class HierarchicalRewardLoss:
    """Loss functions for training a HierarchicalRewardModel."""

    def __init__(
        self,
        reward_model: HierarchicalRewardModel,
        margin: float = 0.0,
    ) -> None:
        self.reward_model = reward_model
        self.margin = margin

    def preference_loss(self, hidden_w: Tensor, hidden_l: Tensor) -> dict[str, float]:
        """Bradley-Terry preference loss.

        Args:
            hidden_w: (B, d_model) hidden states for preferred (winner) responses.
            hidden_l: (B, d_model) hidden states for rejected (loser) responses.

        Returns:
            dict with 'total_loss' and 'margin' keys (float values).
        """
        r_w = self.reward_model(hidden_w)["total_reward"]  # (B,)
        r_l = self.reward_model(hidden_l)["total_reward"]  # (B,)

        loss = -F.logsigmoid(r_w - r_l - self.margin).mean()
        margin_val = (r_w - r_l).mean().item()

        return {
            "total_loss": loss.item(),
            "margin": margin_val,
        }

    def diversity_regularizer(self, hidden: Tensor) -> Tensor:
        """Penalise correlated criterion scores to encourage diverse criteria.

        Stacks per-criterion scores, computes their correlation matrix, and
        returns the mean absolute value of the off-diagonal entries.

        Args:
            hidden: (B, d_model)

        Returns:
            Scalar tensor >= 0.
        """
        B = hidden.shape[0]
        scores = torch.stack(
            [c(hidden) for c in self.reward_model.criteria], dim=1
        )  # (B, n_criteria)

        # Centre each criterion's scores
        scores_c = scores - scores.mean(dim=0, keepdim=True)  # (B, n_criteria)

        # Correlation matrix (n_criteria, n_criteria)
        corr = (scores_c.T @ scores_c) / (B - 1)

        # Off-diagonal mask
        n = corr.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=corr.device)

        off_diag = corr[mask]
        reg = off_diag.abs().mean()

        return reg.clamp(min=0.0)


class MultiObjectiveRewardOptimizer:
    """Utility for scalarising and comparing multi-objective reward vectors."""

    def __init__(self, n_criteria: int) -> None:
        self.n_criteria = n_criteria
        self.weights = torch.ones(n_criteria) / n_criteria

    def set_weights(self, weights: Tensor) -> None:
        """Normalise and store weights.

        Args:
            weights: (n_criteria,) positive weights.
        """
        total = weights.sum()
        self.weights = weights / total if total > 0 else weights.clone()

    def scalarize(self, criterion_scores: dict[str, Tensor], criteria_order: list[str]) -> Tensor:
        """Weighted-sum scalarisation of per-criterion score tensors.

        Args:
            criterion_scores: dict mapping criterion name -> (B,).
            criteria_order:   list of criterion names defining weight order.

        Returns:
            (B,) weighted sum.
        """
        stacked = torch.stack(
            [criterion_scores[name] for name in criteria_order], dim=1
        )  # (B, n_criteria)
        w = self.weights.to(stacked.device, stacked.dtype)
        return (stacked * w.unsqueeze(0)).sum(dim=1)  # (B,)

    def dominated(self, scores_a: Tensor, scores_b: Tensor) -> bool:
        """Check whether b Pareto-dominates a.

        b dominates a iff b >= a on every criterion AND b > a on at least one.

        Args:
            scores_a: (n_criteria,)
            scores_b: (n_criteria,)

        Returns:
            True if b dominates a.
        """
        return bool((scores_b >= scores_a).all().item() and (scores_b > scores_a).any().item())
