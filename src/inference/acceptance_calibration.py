"""Calibration helpers for speculative acceptance probabilities."""

from __future__ import annotations

import torch


def acceptance_brier_score(predicted: torch.Tensor, accepted: torch.Tensor) -> torch.Tensor:
    """Brier score for acceptance-probability predictions."""
    if predicted.shape != accepted.shape:
        raise ValueError("predicted and accepted must match")
    return ((predicted - accepted.float()) ** 2).mean()


def acceptance_ece(predicted: torch.Tensor, accepted: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    """Expected calibration error for acceptance probabilities."""
    if predicted.shape != accepted.shape:
        raise ValueError("predicted and accepted must match")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=predicted.device)
    ece = torch.tensor(0.0, device=predicted.device)
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (predicted >= lower) & (predicted <= upper)
        else:
            mask = (predicted >= lower) & (predicted < upper)
        if mask.any():
            conf = predicted[mask].mean()
            acc = accepted[mask].float().mean()
            ece = ece + mask.float().mean() * (conf - acc).abs()
    return ece


def calibrated_acceptance_report(predicted: torch.Tensor, accepted: torch.Tensor, n_bins: int = 10) -> dict[str, torch.Tensor]:
    """Bundle common acceptance calibration metrics."""
    return {
        "brier": acceptance_brier_score(predicted, accepted),
        "ece": acceptance_ece(predicted, accepted, n_bins=n_bins),
        "mean_predicted": predicted.mean(),
        "mean_accepted": accepted.float().mean(),
    }
