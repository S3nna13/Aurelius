"""Reward Model Rank Correlation Analysis.

Evaluates how well a reward model's ranking of responses correlates with
human preference rankings, using Spearman and Kendall-tau correlation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RankCorrelationMetrics:
    """Container for rank correlation results between reward and human rankings."""

    spearman_rho: float
    kendall_tau: float
    n_pairs: int
    concordant: int
    discordant: int


class SpearmanCorrelation:
    """Computes Spearman rank correlation in pure PyTorch."""

    def __call__(self, x: Tensor, y: Tensor) -> float:
        """Compute Spearman rho between two 1-D float tensors.

        Args:
            x: 1-D float tensor.
            y: 1-D float tensor of the same length.

        Returns:
            rho in [-1, 1].
        """
        if x.shape != y.shape or x.dim() != 1:
            raise ValueError("x and y must be matching 1-D tensors")

        # Rank via double argsort (ordinal ranks, 0-based → convert to float)
        rank_x = x.argsort().argsort().float()
        rank_y = y.argsort().argsort().float()

        # Pearson correlation on ranks
        rx = rank_x - rank_x.mean()
        ry = rank_y - rank_y.mean()
        denom = rx.norm() * ry.norm()
        if denom.item() == 0.0:
            return 0.0
        rho = (rx @ ry) / denom
        return float(rho.clamp(-1.0, 1.0).item())


class KendallTau:
    """Kendall's tau-b correlation in pure PyTorch (O(N²))."""

    def __call__(self, x: Tensor, y: Tensor) -> float:
        """Compute Kendall tau-b between two 1-D float tensors.

        Concordant pair: (x_i - x_j) * (y_i - y_j) > 0
        Discordant pair: (x_i - x_j) * (y_i - y_j) < 0
        Tied pair: either difference equals 0

        Args:
            x: 1-D float tensor.
            y: 1-D float tensor of the same length.

        Returns:
            tau in [-1, 1].
        """
        if x.shape != y.shape or x.dim() != 1:
            raise ValueError("x and y must be matching 1-D tensors")

        n = x.shape[0]
        concordant = 0
        discordant = 0
        tied_x = 0
        tied_y = 0

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i].item() - x[j].item()
                dy = y[i].item() - y[j].item()
                product = dx * dy
                if product > 0:
                    concordant += 1
                elif product < 0:
                    discordant += 1
                else:
                    if dx == 0:
                        tied_x += 1
                    if dy == 0:
                        tied_y += 1

        # tau-b denominator accounts for ties
        n_pairs = concordant + discordant + tied_x + tied_y
        denom = ((concordant + discordant + tied_x) * (concordant + discordant + tied_y)) ** 0.5
        if denom == 0.0:
            return 0.0
        tau = (concordant - discordant) / denom
        # clamp to [-1, 1] for floating-point safety
        tau = max(-1.0, min(1.0, tau))
        return tau


class RewardRankEvaluator:
    """Evaluates a reward model's ranking quality against human preferences."""

    def __init__(self, spearman: SpearmanCorrelation, kendall: KendallTau) -> None:
        self.spearman = spearman
        self.kendall = kendall

    def evaluate(self, reward_scores: Tensor, human_rankings: Tensor) -> RankCorrelationMetrics:
        """Compute full rank correlation metrics.

        Args:
            reward_scores: 1-D tensor of reward model scores, shape (N,).
            human_rankings: 1-D tensor of human preference rankings, shape (N,).

        Returns:
            RankCorrelationMetrics with Spearman rho, Kendall tau, and pair counts.
        """
        if reward_scores.shape != human_rankings.shape or reward_scores.dim() != 1:
            raise ValueError("reward_scores and human_rankings must be matching 1-D tensors")

        rho = self.spearman(reward_scores, human_rankings)
        tau = self.kendall(reward_scores, human_rankings)

        # Count concordant/discordant pairs for the metrics dataclass
        n = reward_scores.shape[0]
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = reward_scores[i].item() - reward_scores[j].item()
                dy = human_rankings[i].item() - human_rankings[j].item()
                product = dx * dy
                if product > 0:
                    concordant += 1
                elif product < 0:
                    discordant += 1

        n_pairs = n * (n - 1) // 2
        return RankCorrelationMetrics(
            spearman_rho=rho,
            kendall_tau=tau,
            n_pairs=n_pairs,
            concordant=concordant,
            discordant=discordant,
        )

    def evaluate_pairwise(
        self,
        reward_scores: Tensor,
        preferred_indices: Tensor,
        rejected_indices: Tensor,
    ) -> float:
        """Compute pairwise accuracy: fraction where reward ranks preferred > rejected.

        Args:
            reward_scores: 1-D tensor of reward scores for all candidates, shape (M,).
            preferred_indices: 1-D integer tensor of length N with preferred sample indices.
            rejected_indices: 1-D integer tensor of length N with rejected sample indices.

        Returns:
            Accuracy in [0, 1].
        """
        if preferred_indices.shape != rejected_indices.shape or preferred_indices.dim() != 1:
            raise ValueError("preferred_indices and rejected_indices must be matching 1-D tensors")

        n = preferred_indices.shape[0]
        if n == 0:
            return 0.0

        preferred_scores = reward_scores[preferred_indices]
        rejected_scores = reward_scores[rejected_indices]
        correct = (preferred_scores > rejected_scores).sum().item()
        return float(correct) / float(n)
