"""Margin-based metrics over judge scores."""

from __future__ import annotations

import torch


def score_margin(left_scores: torch.Tensor, right_scores: torch.Tensor) -> torch.Tensor:
    """Elementwise score margin between left and right candidates."""
    if left_scores.shape != right_scores.shape:
        raise ValueError("left_scores and right_scores must match")
    return left_scores - right_scores


def mean_margin(left_scores: torch.Tensor, right_scores: torch.Tensor) -> torch.Tensor:
    """Average judge margin across examples."""
    return score_margin(left_scores, right_scores).mean()


def decisive_fraction(left_scores: torch.Tensor, right_scores: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Fraction of examples with absolute margin above threshold."""
    margins = score_margin(left_scores, right_scores)
    return (margins.abs() >= threshold).float().mean()

