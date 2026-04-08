"""Calibration metrics for evidence relevance scores."""

from __future__ import annotations

import torch


def evidence_brier_score(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Brier score for evidence relevance predictions."""
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must match")
    return ((scores - labels.float()) ** 2).mean()


def evidence_ece(scores: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    """ECE for evidence relevance predictions."""
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must match")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=scores.device)
    ece = torch.tensor(0.0, device=scores.device)
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        if mask.any():
            conf = scores[mask].mean()
            acc = labels[mask].float().mean()
            ece = ece + mask.float().mean() * (conf - acc).abs()
    return ece


def evidence_calibration_report(scores: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> dict[str, torch.Tensor]:
    """Bundle evidence calibration metrics."""
    return {
        "brier": evidence_brier_score(scores, labels),
        "ece": evidence_ece(scores, labels, n_bins=n_bins),
        "mean_score": scores.mean(),
        "mean_label": labels.float().mean(),
    }
