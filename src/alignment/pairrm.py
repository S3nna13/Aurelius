"""Pairwise ranking-model utilities for response reranking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairwisePreference:
    chosen_score: float
    rejected_score: float

    @property
    def margin(self) -> float:
        return self.chosen_score - self.rejected_score


def pairrm_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    """Logistic pairwise ranking loss."""
    if chosen_scores.shape != rejected_scores.shape:
        raise ValueError("chosen_scores and rejected_scores must match")
    return F.softplus(-(chosen_scores - rejected_scores)).mean()


def pairrm_accuracy(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    """Fraction of pairs ranked correctly."""
    if chosen_scores.shape != rejected_scores.shape:
        raise ValueError("chosen_scores and rejected_scores must match")
    return (chosen_scores > rejected_scores).to(dtype=torch.float32).mean()


def pairrm_margin_stats(preferences: list[PairwisePreference]) -> dict[str, float]:
    """Compute simple summary stats over preference margins."""
    if not preferences:
        return {"mean_margin": 0.0, "min_margin": 0.0, "max_margin": 0.0}
    margins = torch.tensor([preference.margin for preference in preferences], dtype=torch.float32)
    return {
        "mean_margin": margins.mean().item(),
        "min_margin": margins.min().item(),
        "max_margin": margins.max().item(),
    }


def rerank_candidates(candidate_scores: torch.Tensor) -> torch.Tensor:
    """Return candidate indices sorted by descending score."""
    if candidate_scores.dim() != 1:
        raise ValueError("candidate_scores must be 1D")
    return torch.argsort(candidate_scores, descending=True)
