"""Confidence calibration metrics for probabilistic predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def expected_calibration_error(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    n_bins: int = 10,
) -> torch.Tensor:
    """Compute ECE over binned confidences."""
    if confidences.shape != correctness.shape:
        raise ValueError("confidences and correctness must match")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=confidences.device)
    ece = torch.tensor(0.0, device=confidences.device)
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if mask.any():
            acc = correctness[mask].float().mean()
            conf = confidences[mask].mean()
            ece = ece + mask.float().mean() * (acc - conf).abs()
    return ece


def brier_score(confidences: torch.Tensor, correctness: torch.Tensor) -> torch.Tensor:
    """Compute the Brier score for binary correctness labels."""
    if confidences.shape != correctness.shape:
        raise ValueError("confidences and correctness must match")
    return ((confidences - correctness.float()) ** 2).mean()


@dataclass(frozen=True)
class CalibrationReport:
    ece: torch.Tensor
    brier: torch.Tensor
    avg_confidence: torch.Tensor
    avg_accuracy: torch.Tensor


def calibration_report(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    n_bins: int = 10,
) -> CalibrationReport:
    """Bundle common calibration metrics in one report."""
    return CalibrationReport(
        ece=expected_calibration_error(confidences, correctness, n_bins=n_bins),
        brier=brier_score(confidences, correctness),
        avg_confidence=confidences.mean(),
        avg_accuracy=correctness.float().mean(),
    )
