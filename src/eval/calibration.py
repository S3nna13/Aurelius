"""Uncertainty calibration: temperature scaling, Platt scaling, ECE, reliability diagrams."""

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
    temperature_init: float = 1.5
    lr: float = 0.01
    n_iter: int = 100


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ece(
    probs: Tensor,
    labels: Tensor,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Bin predictions by confidence, then compute the weighted average of
    |accuracy - confidence| across non-empty bins.

    Args:
        probs: (N,) predicted probabilities for the correct class.
        labels: (N,) binary correctness labels (1 = correct, 0 = wrong).
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE as a float in [0, 1].
    """
    probs = probs.float()
    labels = labels.float()
    n = probs.shape[0]
    if n == 0:
        return 0.0

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        bin_size = mask.sum().item()
        if bin_size == 0:
            continue
        bin_acc = labels[mask].mean().item()
        bin_conf = probs[mask].mean().item()
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_mce(probs: Tensor, labels: Tensor, n_bins: int = 10) -> float:
    """Maximum Calibration Error.

    Maximum |accuracy - confidence| across non-empty bins.

    Args:
        probs: (N,) predicted probabilities for the correct class.
        labels: (N,) binary correctness labels.
        n_bins: Number of equal-width bins.

    Returns:
        MCE as a float in [0, 1].
    """
    probs = probs.float()
    labels = labels.float()
    n = probs.shape[0]
    if n == 0:
        return 0.0

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    mce = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if mask.sum().item() == 0:
            continue
        bin_acc = labels[mask].mean().item()
        bin_conf = probs[mask].mean().item()
        mce = max(mce, abs(bin_acc - bin_conf))

    return float(mce)


def compute_brier_score(probs: Tensor, labels: Tensor) -> float:
    """Mean squared error between predicted probabilities and binary labels.

    Args:
        probs: (N,) predicted probabilities.
        labels: (N,) binary labels.

    Returns:
        Brier score in [0, 1].
    """
    probs = probs.float()
    labels = labels.float()
    return float(((probs - labels) ** 2).mean().item())


def reliability_diagram_data(
    probs: Tensor,
    labels: Tensor,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Compute data needed to draw a reliability diagram.

    Args:
        probs: (N,) predicted probabilities for the correct class.
        labels: (N,) binary correctness labels.
        n_bins: Number of equal-width bins.

    Returns:
        Dictionary with keys:
            ``bin_centers``: midpoint of each bin.
            ``bin_accs``: mean accuracy per bin (0 for empty bins).
            ``bin_confs``: mean confidence per bin (bin_center for empty bins).
            ``bin_sizes``: number of samples per bin.
    """
    probs = probs.float()
    labels = labels.float()

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    bin_centers = ((bin_boundaries[:-1] + bin_boundaries[1:]) / 2).tolist()

    bin_accs: list[float] = []
    bin_confs: list[float] = []
    bin_sizes: list[float] = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        bin_size = int(mask.sum().item())
        bin_sizes.append(float(bin_size))
        if bin_size == 0:
            bin_accs.append(0.0)
            bin_confs.append(bin_centers[i])
        else:
            bin_accs.append(float(labels[mask].mean().item()))
            bin_confs.append(float(probs[mask].mean().item()))

    return {
        "bin_centers": bin_centers,
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_sizes": bin_sizes,
    }


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """Single temperature parameter for post-hoc calibration of logits."""

    def __init__(self, temperature: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature]))

    def forward(self, logits: Tensor) -> Tensor:
        """Divide logits by temperature, return softmax probabilities.

        Args:
            logits: (..., C) raw logits.

        Returns:
            (..., C) probability distribution.
        """
        return F.softmax(logits / self.temperature, dim=-1)

    def get_temperature(self) -> float:
        """Return the current scalar temperature value."""
        return float(self.temperature.item())


class PlattScaler(nn.Module):
    """Affine calibration: a * logit + b, then sigmoid for binary confidence."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, logits: Tensor) -> Tensor:
        """Apply a * logits + b, then sigmoid.

        Args:
            logits: (N,) scalar confidence scores.

        Returns:
            (N,) calibrated probabilities in (0, 1).
        """
        return torch.sigmoid(self.a * logits + self.b)


# ---------------------------------------------------------------------------
# Fitting routines
# ---------------------------------------------------------------------------

def fit_temperature_scaling(
    logits: Tensor,
    labels: Tensor,
    cfg: CalibrationConfig,
) -> TemperatureScaler:
    """Fit temperature by minimising NLL on validation logits using LBFGS.

    Args:
        logits: (N, C) uncalibrated logits.
        labels: (N,) integer class labels.
        cfg: Calibration configuration.

    Returns:
        Fitted :class:`TemperatureScaler`.
    """
    scaler = TemperatureScaler(temperature=cfg.temperature_init)
    optimizer = torch.optim.LBFGS(
        [scaler.temperature], lr=cfg.lr, max_iter=cfg.n_iter
    )

    logits = logits.detach()
    labels = labels.detach()

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / scaler.temperature
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


def fit_platt_scaling(
    scores: Tensor,
    labels: Tensor,
    cfg: CalibrationConfig,
) -> PlattScaler:
    """Fit Platt scaling by minimising BCE loss using Adam.

    Args:
        scores: (N,) scalar confidence scores.
        labels: (N,) binary labels (0 or 1).
        cfg: Calibration configuration.

    Returns:
        Fitted :class:`PlattScaler`.
    """
    scaler = PlattScaler()
    optimizer = torch.optim.Adam(scaler.parameters(), lr=cfg.lr)

    scores = scores.detach().float()
    labels = labels.detach().float()

    for _ in range(cfg.n_iter):
        optimizer.zero_grad()
        probs = scaler(scores)
        loss = F.binary_cross_entropy(probs, labels)
        loss.backward()
        optimizer.step()

    return scaler


# ---------------------------------------------------------------------------
# End-to-end calibration pipeline
# ---------------------------------------------------------------------------

class ModelCalibrator:
    """End-to-end calibration pipeline for a language model."""

    def __init__(self, model: nn.Module, cfg: CalibrationConfig) -> None:
        self.model = model
        self.cfg = cfg

    @torch.no_grad()
    def collect_logits(
        self, input_ids: Tensor, target_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run model, collect per-position logits for next-token prediction.

        Args:
            input_ids: (B, T) input token ids.
            target_ids: (B, T) target token ids (same shape as input_ids).

        Returns:
            Tuple of:
                logits: (N, vocab_size) where N = B * (T-1).
                labels: (N,) integer next-token labels.
        """
        _, logits, _ = self.model(input_ids)
        # logits[:, :-1, :] predicts position 1..T-1; target is token at 1..T-1
        pred_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
        lbl = target_ids[:, 1:].contiguous()            # (B, T-1)
        B, Tm1, V = pred_logits.shape
        return pred_logits.view(B * Tm1, V), lbl.view(B * Tm1)

    def calibrate(
        self, input_ids: Tensor, target_ids: Tensor
    ) -> TemperatureScaler:
        """Collect logits and fit temperature scaling.

        Args:
            input_ids: (B, T) input token ids.
            target_ids: (B, T) target token ids.

        Returns:
            Fitted :class:`TemperatureScaler`.
        """
        logits, labels = self.collect_logits(input_ids, target_ids)
        return fit_temperature_scaling(logits, labels, self.cfg)

    def evaluate_calibration(
        self, input_ids: Tensor, target_ids: Tensor
    ) -> dict[str, float]:
        """Evaluate calibration before and after temperature scaling.

        Returns a dictionary with keys:
            ``ece_before``, ``ece_after``, ``brier_before``, ``brier_after``,
            ``temperature``.
        """
        logits, labels = self.collect_logits(input_ids, target_ids)

        # --- Before calibration ---
        probs_all_before = F.softmax(logits, dim=-1)       # (N, V)
        # Probability assigned to the correct class
        correct_probs_before = probs_all_before[
            torch.arange(labels.shape[0]), labels
        ]  # (N,)
        # Binary correctness: 1 if argmax == label
        correct_before = (probs_all_before.argmax(dim=-1) == labels).float()

        ece_before = compute_ece(correct_probs_before, correct_before, self.cfg.n_bins)
        brier_before = compute_brier_score(correct_probs_before, correct_before)

        # --- Fit temperature ---
        scaler = fit_temperature_scaling(logits, labels, self.cfg)

        # --- After calibration ---
        probs_all_after = scaler(logits)                   # (N, V)
        correct_probs_after = probs_all_after[
            torch.arange(labels.shape[0]), labels
        ]
        correct_after = (probs_all_after.argmax(dim=-1) == labels).float()

        ece_after = compute_ece(correct_probs_after, correct_after, self.cfg.n_bins)
        brier_after = compute_brier_score(correct_probs_after, correct_after)

        return {
            "ece_before": ece_before,
            "ece_after": ece_after,
            "brier_before": brier_before,
            "brier_after": brier_after,
            "temperature": scaler.get_temperature(),
        }
