"""Advanced reward model calibration with temperature scaling, Platt scaling,
isotonic regression, and Expected Calibration Error (ECE) metrics.

This v2 module introduces:
- CalibrationConfig dataclass for unified configuration
- expected_calibration_error standalone function
- TemperatureScaler with .fit() returning metrics dict
- PlattScaler with .fit() returning metrics dict
- IsotonicCalibrator pure-PyTorch PAVA implementation
- calibrate_reward_scores convenience function
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CalibrationConfig:
    method: str = "temperature"   # "temperature", "platt", "isotonic"
    n_bins: int = 10              # bins for ECE computation
    lr: float = 0.01             # learning rate for parameter optimization
    max_iter: int = 200          # max optimization iterations


def expected_calibration_error(
    scores: Tensor,
    labels: Tensor,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Bins predictions into n_bins equal-width buckets [0, 1/n_bins), ..., [k/n_bins, 1].
    ECE = sum_b |B_b|/N * |acc(b) - conf(b)|
    where acc(b) = fraction of positive labels in bin b,
          conf(b) = mean predicted score in bin b.

    Args:
        scores: (N,) predicted scores/probabilities in [0, 1]
        labels: (N,) binary ground-truth labels (0 or 1)
        n_bins: number of equal-width bins

    Returns:
        float in [0, 1]
    """
    scores = scores.detach().float()
    labels = labels.detach().float()
    n = len(scores)

    if n == 0:
        return 0.0

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for b in range(n_bins):
        lo = bin_boundaries[b].item()
        hi = bin_boundaries[b + 1].item()

        # Last bin is inclusive on the right edge
        if b == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)

        if mask.sum() == 0:
            continue

        bin_scores = scores[mask]
        bin_labels = labels[mask]
        bin_size = mask.sum().item()

        conf = bin_scores.mean().item()
        acc = bin_labels.mean().item()

        ece += (bin_size / n) * abs(acc - conf)

    return float(ece)


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling: score_cal = score / T."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, scores: Tensor) -> Tensor:
        """Apply temperature scaling. Returns calibrated scores."""
        return scores / self.temperature.clamp(min=1e-8)

    def fit(
        self,
        scores: Tensor,
        labels: Tensor,
        config: CalibrationConfig,
    ) -> dict[str, float]:
        """Optimize temperature via NLL loss on validation set.

        Args:
            scores: (N,) uncalibrated scores (logits)
            labels: (N,) binary labels (0 or 1)
            config: CalibrationConfig with lr, max_iter, n_bins

        Returns:
            dict with keys: 'temperature', 'ece_before', 'ece_after'
        """
        # ECE before: treat sigmoid of raw scores as probabilities
        probs_before = torch.sigmoid(scores.detach().float())
        ece_before = expected_calibration_error(probs_before, labels, config.n_bins)

        # Reset temperature
        with torch.no_grad():
            self.temperature.fill_(1.0)

        scores_d = scores.detach().float()
        labels_d = labels.detach().float()

        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=config.lr, max_iter=config.max_iter
        )

        def closure():
            optimizer.zero_grad()
            calibrated = scores_d / self.temperature.clamp(min=1e-8)
            loss = F.binary_cross_entropy_with_logits(calibrated, labels_d)
            loss.backward()
            return loss

        optimizer.step(closure)

        # ECE after
        with torch.no_grad():
            probs_after = torch.sigmoid(self.forward(scores_d))
        ece_after = expected_calibration_error(probs_after, labels, config.n_bins)

        return {
            "temperature": float(self.temperature.item()),
            "ece_before": float(ece_before),
            "ece_after": float(ece_after),
        }


class PlattScaler(nn.Module):
    """Platt scaling: score_cal = sigmoid(a * score + b)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, scores: Tensor) -> Tensor:
        """Apply Platt scaling. Returns calibrated probabilities in (0, 1)."""
        return torch.sigmoid(self.a * scores + self.b)

    def fit(
        self,
        scores: Tensor,
        labels: Tensor,
        config: CalibrationConfig,
    ) -> dict[str, float]:
        """Optimize a, b via binary cross-entropy.

        Args:
            scores: (N,) uncalibrated scores (logits)
            labels: (N,) binary labels (0 or 1)
            config: CalibrationConfig with lr, max_iter, n_bins

        Returns:
            dict with keys: 'a', 'b', 'ece_before', 'ece_after'
        """
        # ECE before
        probs_before = torch.sigmoid(scores.detach().float())
        ece_before = expected_calibration_error(probs_before, labels, config.n_bins)

        # Reset parameters
        with torch.no_grad():
            self.a.fill_(1.0)
            self.b.fill_(0.0)

        scores_d = scores.detach().float()
        labels_d = labels.detach().float()

        optimizer = torch.optim.LBFGS(
            [self.a, self.b], lr=config.lr, max_iter=config.max_iter
        )

        def closure():
            optimizer.zero_grad()
            logits = self.a * scores_d + self.b
            loss = F.binary_cross_entropy_with_logits(logits, labels_d)
            loss.backward()
            return loss

        optimizer.step(closure)

        # ECE after
        with torch.no_grad():
            probs_after = self.forward(scores_d)
        ece_after = expected_calibration_error(probs_after, labels, config.n_bins)

        return {
            "a": float(self.a.item()),
            "b": float(self.b.item()),
            "ece_before": float(ece_before),
            "ece_after": float(ece_after),
        }


class IsotonicCalibrator:
    """Pure PyTorch isotonic regression calibration via pool adjacent violators."""

    def __init__(self) -> None:
        self._x_breakpoints: Optional[Tensor] = None
        self._y_breakpoints: Optional[Tensor] = None

    def fit(self, scores: Tensor, labels: Tensor) -> None:
        """Fit isotonic regression on scores -> labels mapping.

        Uses Pool Adjacent Violators Algorithm (PAVA) to ensure monotone
        non-decreasing mapping from scores to calibrated probabilities.

        Args:
            scores: (N,) raw scores
            labels: (N,) binary labels (0 or 1)
        """
        scores_f = scores.detach().float()
        labels_f = labels.detach().float()

        # Sort by ascending score
        order = torch.argsort(scores_f)
        x_sorted = scores_f[order]
        y_sorted = labels_f[order]

        # PAVA: each block is [sum_y, count, [x values]]
        blocks: list[list] = []

        for i in range(len(x_sorted)):
            xi = x_sorted[i].item()
            yi = y_sorted[i].item()
            blocks.append([yi, 1, [xi]])

            # Merge while monotonicity is violated
            while len(blocks) > 1:
                mean_prev = blocks[-2][0] / blocks[-2][1]
                mean_curr = blocks[-1][0] / blocks[-1][1]
                if mean_prev > mean_curr:
                    prev = blocks.pop(-2)
                    curr = blocks[-1]
                    curr[0] += prev[0]
                    curr[1] += prev[1]
                    curr[2] = prev[2] + curr[2]
                else:
                    break

        x_pts = []
        y_pts = []
        for block in blocks:
            sum_y, count, xs = block
            x_pts.append(sum(xs) / len(xs))
            y_pts.append(sum_y / count)

        self._x_breakpoints = torch.tensor(x_pts, dtype=torch.float32)
        self._y_breakpoints = torch.tensor(y_pts, dtype=torch.float32)

    def predict(self, scores: Tensor) -> Tensor:
        """Apply calibration using fitted isotonic mapping (nearest lookup).

        Uses nearest-neighbor lookup into the fitted breakpoints.

        Args:
            scores: (N,) raw scores

        Returns:
            (N,) calibrated probabilities clamped to [0, 1]
        """
        if self._x_breakpoints is None or self._y_breakpoints is None:
            raise RuntimeError("Must call fit() before predict().")

        x_bp = self._x_breakpoints
        y_bp = self._y_breakpoints
        scores_f = scores.detach().float()

        result = torch.empty_like(scores_f)
        for i, s in enumerate(scores_f):
            s_val = s.item()
            # Find nearest breakpoint
            dists = torch.abs(x_bp - s_val)
            nearest_idx = torch.argmin(dists).item()
            result[i] = y_bp[nearest_idx]

        return result.clamp(0.0, 1.0)


def calibrate_reward_scores(
    scores: Tensor,
    labels: Tensor,
    config: CalibrationConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Calibrate reward scores using specified method.

    Args:
        scores: (N,) raw reward scores (logits for temperature/platt,
                probabilities for isotonic)
        labels: (N,) binary ground-truth labels (0 or 1)
        config: CalibrationConfig specifying method and hyperparameters

    Returns:
        (calibrated_scores, metrics_dict)
        metrics_dict contains: 'ece_before', 'ece_after', 'method'
    """
    method = config.method

    if method == "temperature":
        scaler = TemperatureScaler()
        fit_metrics = scaler.fit(scores, labels, config)
        with torch.no_grad():
            calibrated = torch.sigmoid(scaler(scores.float()))
        metrics: dict[str, float] = {
            "ece_before": fit_metrics["ece_before"],
            "ece_after": fit_metrics["ece_after"],
            "method": method,  # type: ignore[assignment]
        }

    elif method == "platt":
        scaler_p = PlattScaler()
        fit_metrics_p = scaler_p.fit(scores, labels, config)
        with torch.no_grad():
            calibrated = scaler_p(scores.float())
        metrics = {
            "ece_before": fit_metrics_p["ece_before"],
            "ece_after": fit_metrics_p["ece_after"],
            "method": method,  # type: ignore[assignment]
        }

    elif method == "isotonic":
        # For isotonic, treat sigmoid of scores as probabilities for ECE before
        probs_before = torch.sigmoid(scores.detach().float())
        ece_before = expected_calibration_error(probs_before, labels, config.n_bins)

        cal = IsotonicCalibrator()
        cal.fit(scores, labels)
        calibrated = cal.predict(scores)

        ece_after = expected_calibration_error(calibrated, labels, config.n_bins)
        metrics = {
            "ece_before": float(ece_before),
            "ece_after": float(ece_after),
            "method": method,  # type: ignore[assignment]
        }

    else:
        raise ValueError(
            f"Unknown calibration method: {method!r}. "
            "Choose 'temperature', 'platt', or 'isotonic'."
        )

    # Ensure method is stored as str in the dict
    metrics["method"] = method  # type: ignore[assignment]
    return calibrated, metrics
