"""Metrics for comparing reward-model ensembles."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def ensemble_mean(scores: torch.Tensor) -> torch.Tensor:
    """Mean reward across ensemble members."""
    if scores.dim() != 2:
        raise ValueError("scores must be 2D [models, examples]")
    return scores.mean(dim=0)


def ensemble_std(scores: torch.Tensor) -> torch.Tensor:
    """Standard deviation across ensemble members."""
    if scores.dim() != 2:
        raise ValueError("scores must be 2D [models, examples]")
    return scores.std(dim=0, unbiased=False)


def ensemble_disagreement(scores: torch.Tensor) -> torch.Tensor:
    """Mean per-example ensemble spread."""
    return ensemble_std(scores).mean()


@dataclass(frozen=True)
class RewardEnsembleReport:
    mean: torch.Tensor
    std: torch.Tensor
    disagreement: torch.Tensor


def reward_ensemble_report(scores: torch.Tensor) -> RewardEnsembleReport:
    """Bundle ensemble mean/std/disagreement metrics."""
    return RewardEnsembleReport(
        mean=ensemble_mean(scores),
        std=ensemble_std(scores),
        disagreement=ensemble_disagreement(scores),
    )
