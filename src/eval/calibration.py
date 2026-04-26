"""Model calibration: ECE, MCE, Brier score, temperature scaling, reliability diagrams."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CalibrationConfig:
    """Configuration for calibration routines."""

    n_bins: int = 10
    min_samples_per_bin: int = 5
    temperature_init: float = 1.0


# ---------------------------------------------------------------------------
# Per-sample helpers
# ---------------------------------------------------------------------------


def compute_confidence(logits: Tensor) -> Tensor:
    """Return max softmax probability per sample.

    Args:
        logits: (N, vocab) raw logits.

    Returns:
        (N,) confidence values in [0, 1].
    """
    probs = F.softmax(logits.float(), dim=-1)
    return probs.max(dim=-1).values


def compute_accuracy(logits: Tensor, labels: Tensor) -> Tensor:
    """Return per-sample accuracy (1 if argmax == label, else 0).

    Args:
        logits: (N, vocab) raw logits.
        labels: (N,) integer class labels.

    Returns:
        (N,) float tensor of 0s and 1s.
    """
    preds = logits.argmax(dim=-1)
    return (preds == labels).float()


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def _bin_data(
    confidences: Tensor,
    accuracies: Tensor,
    n_bins: int,
) -> list[tuple[Tensor, Tensor]]:
    """Return list of (bin_confs, bin_accs) for each bin, empty tensors for empty bins."""
    confidences = confidences.float()
    accuracies = accuracies.float()
    bins: list[tuple[Tensor, Tensor]] = []
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=confidences.device)
    for i in range(n_bins):
        lo, hi = boundaries[i], boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        bins.append((confidences[mask], accuracies[mask]))
    return bins


def compute_ece(
    confidences: Tensor,
    accuracies: Tensor,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error: weighted average |acc - conf| per bin.

    Args:
        confidences: (N,) predicted confidence values.
        accuracies:  (N,) per-sample accuracy (0 or 1).
        n_bins:      Number of equal-width bins in [0, 1].

    Returns:
        ECE as a float in [0, 1].
    """
    n = confidences.shape[0]
    if n == 0:
        return 0.0
    bins = _bin_data(confidences, accuracies, n_bins)
    ece = 0.0
    for bin_confs, bin_accs in bins:
        size = bin_confs.shape[0]
        if size == 0:
            continue
        ece += (size / n) * abs(bin_accs.mean().item() - bin_confs.mean().item())
    return float(ece)


def compute_mce(
    confidences: Tensor,
    accuracies: Tensor,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error: max |acc - conf| over non-empty bins.

    Args:
        confidences: (N,) predicted confidence values.
        accuracies:  (N,) per-sample accuracy (0 or 1).
        n_bins:      Number of equal-width bins.

    Returns:
        MCE as a float in [0, 1].
    """
    bins = _bin_data(confidences, accuracies, n_bins)
    mce = 0.0
    for bin_confs, bin_accs in bins:
        if bin_confs.shape[0] == 0:
            continue
        gap = abs(bin_accs.mean().item() - bin_confs.mean().item())
        if gap > mce:
            mce = gap
    return float(mce)


def reliability_diagram_data(
    confidences: Tensor,
    accuracies: Tensor,
    n_bins: int = 10,
) -> dict[str, Tensor]:
    """Compute data for a reliability diagram.

    Args:
        confidences: (N,) predicted confidence values.
        accuracies:  (N,) per-sample accuracy (0 or 1).
        n_bins:      Number of equal-width bins.

    Returns:
        Dict with keys:
            ``bin_centers``     (n_bins,) midpoint of each bin.
            ``bin_accuracies``  (n_bins,) mean accuracy per bin (0 for empty).
            ``bin_confidences`` (n_bins,) mean confidence per bin (center for empty).
            ``bin_counts``      (n_bins,) number of samples per bin.
    """
    device = confidences.device
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
    centers = (boundaries[:-1] + boundaries[1:]) / 2.0

    bin_accs = torch.zeros(n_bins, device=device)
    bin_confs = centers.clone()
    bin_counts = torch.zeros(n_bins, device=device)

    bins = _bin_data(confidences, accuracies, n_bins)
    for i, (bc, ba) in enumerate(bins):
        size = bc.shape[0]
        bin_counts[i] = float(size)
        if size > 0:
            bin_accs[i] = ba.mean()
            bin_confs[i] = bc.mean()

    return {
        "bin_centers": centers,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
    }


def compute_brier_score(probs: Tensor, labels: Tensor) -> float:
    """Mean squared error between predicted probabilities and one-hot labels.

    Supports both binary (N,) and multiclass (N, C) inputs.

    Args:
        probs:  (N,) or (N, C) predicted probabilities.
        labels: (N,) integer class labels (multiclass) or binary (0/1).

    Returns:
        Brier score as a float in [0, 1].
    """
    probs = probs.float()
    if probs.dim() == 1:
        # Binary case: treat labels as float targets
        targets = labels.float()
        return float(((probs - targets) ** 2).mean().item())
    else:
        # Multiclass: one-hot encode labels
        n, c = probs.shape
        one_hot = torch.zeros(n, c, device=probs.device, dtype=torch.float32)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        return float(((probs - one_hot) ** 2).mean().item())


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


class TemperatureScaling(nn.Module):
    """Single learnable temperature parameter for post-hoc calibration."""

    def __init__(self, temperature_init: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor([temperature_init]).log())

    @property
    def temperature(self) -> Tensor:
        """Temperature T > 0 via softplus to ensure positivity."""
        return self.log_temperature.exp()

    def forward(self, logits: Tensor) -> Tensor:
        """Divide logits by temperature.

        Args:
            logits: (..., C) raw logits.

        Returns:
            (..., C) scaled logits.
        """
        return logits / self.temperature

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        n_steps: int = 100,
        lr: float = 0.01,
    ) -> float:
        """Optimise temperature via NLL loss (cross-entropy).

        Args:
            logits:  (N, C) uncalibrated logits.
            labels:  (N,) integer class labels.
            n_steps: Number of gradient steps.
            lr:      Learning rate.

        Returns:
            Final temperature value as a float.
        """
        logits = logits.detach().float()
        labels = labels.detach().long()

        optimizer = torch.optim.Adam([self.log_temperature], lr=lr)

        for _ in range(n_steps):
            optimizer.zero_grad()
            scaled = logits / self.temperature
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            optimizer.step()

        return float(self.temperature.item())


# ---------------------------------------------------------------------------
# End-to-end calibrator
# ---------------------------------------------------------------------------


class ModelCalibrator:
    """End-to-end calibration evaluation for a language model."""

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config

    def evaluate(self, logits: Tensor, labels: Tensor) -> dict[str, float]:
        """Compute calibration metrics for given logits and labels.

        Args:
            logits: (N, vocab) raw model logits.
            labels: (N,) integer class labels.

        Returns:
            Dict with keys: ``ece``, ``mce``, ``brier_score``,
            ``mean_confidence``, ``accuracy``.
        """
        confidences = compute_confidence(logits)
        accuracies = compute_accuracy(logits, labels)

        probs = F.softmax(logits.float(), dim=-1)
        brier = compute_brier_score(probs, labels)

        return {
            "ece": compute_ece(confidences, accuracies, self.config.n_bins),
            "mce": compute_mce(confidences, accuracies, self.config.n_bins),
            "brier_score": brier,
            "mean_confidence": float(confidences.mean().item()),
            "accuracy": float(accuracies.mean().item()),
        }
