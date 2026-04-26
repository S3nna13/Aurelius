"""Platt scaling, isotonic regression (PAVA), and temperature scaling for post-hoc calibration.

References:
    - Platt (1999): "Probabilistic outputs for support vector machines and comparisons
      to regularized likelihood methods."
    - Guo et al. (2017): "On Calibration of Modern Neural Networks." arXiv:1706.04599.
    - Classic Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Temperature Scaling  (Guo et al. 2017)
# ---------------------------------------------------------------------------


class TemperatureScaler:
    """Scales model logits by a single learned temperature T > 0.

    calibrated_prob = softmax(logits / T)

    T is found by minimising negative log-likelihood on a held-out calibration
    set using simple SGD.
    """

    def __init__(self) -> None:
        self._temperature: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        """Optimal temperature found during :meth:`fit`."""
        return self._temperature

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        n_iters: int = 200,
    ) -> TemperatureScaler:
        """Find optimal temperature T by minimising NLL on calibration data.

        Args:
            logits: (N, C) raw (un-scaled) logits from the model.
            labels: (N,) integer class labels in [0, C).
            lr:     Learning rate for SGD.
            n_iters: Number of optimisation steps.

        Returns:
            self (for chaining).
        """
        logits = logits.detach().float()
        labels = labels.detach().long()

        if logits.ndim != 2:
            raise ValueError(f"logits must be 2-D (N, C), got shape {logits.shape}")
        if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError("labels must be 1-D with the same length as logits")

        # Optimise log(T) to keep T > 0 throughout training.
        log_t = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.SGD([log_t], lr=lr)

        for _ in range(n_iters):
            optimizer.zero_grad()
            t = log_t.exp()
            scaled = logits / t
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            optimizer.step()

        self._temperature = float(log_t.exp().item())
        return self

    def calibrate(self, logits: Tensor) -> Tensor:
        """Apply temperature scaling and return calibrated probabilities.

        Args:
            logits: (N, C) raw logits.

        Returns:
            (N, C) probability tensor (rows sum to 1).
        """
        if logits.ndim != 2:
            raise ValueError(f"logits must be 2-D (N, C), got shape {logits.shape}")
        scaled = logits.float() / self._temperature
        return F.softmax(scaled, dim=-1)


# ---------------------------------------------------------------------------
# Platt Scaling  (Platt 1999)
# ---------------------------------------------------------------------------


class PlattScaler:
    """Logistic calibration for binary classification scores.

    P(y=1 | f) = σ(A * f + B)

    A and B are learned by minimising binary cross-entropy over (scores, labels).
    """

    def __init__(self) -> None:
        self._A: float = 1.0
        self._B: float = 0.0

    def fit(
        self,
        scores: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        n_iters: int = 200,
    ) -> PlattScaler:
        """Fit logistic calibration parameters A and B.

        Args:
            scores: (N,) real-valued model scores (e.g. log-odds or SVM outputs).
            labels: (N,) binary labels in {0, 1}.
            lr:     Learning rate for SGD.
            n_iters: Number of optimisation steps.

        Returns:
            self (for chaining).
        """
        scores = scores.detach().float()
        labels = labels.detach().float()

        if scores.ndim != 1:
            raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
        if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
            raise ValueError("labels must be 1-D with the same length as scores")

        A = torch.zeros(1, requires_grad=True)
        B = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.SGD([A, B], lr=lr)

        for _ in range(n_iters):
            optimizer.zero_grad()
            logits = A * scores + B
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()

        self._A = float(A.item())
        self._B = float(B.item())
        return self

    def calibrate(self, scores: Tensor) -> Tensor:
        """Map raw scores to calibrated probabilities via sigmoid.

        Args:
            scores: (N,) model scores.

        Returns:
            (N,) probability tensor with values in (0, 1).
        """
        if scores.ndim != 1:
            raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
        logits = self._A * scores.float() + self._B
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Isotonic Regression / PAVA  (pure PyTorch)
# ---------------------------------------------------------------------------


def _pava(values: Tensor) -> Tensor:
    """Pool Adjacent Violators Algorithm — pure PyTorch implementation.

    Returns a non-decreasing sequence (y_1, …, y_n) that minimises
    Σ (x_i − y_i)² subject to y_1 ≤ y_2 ≤ … ≤ y_n.

    Args:
        values: 1-D float tensor of length N.

    Returns:
        1-D float tensor of length N, non-decreasing.
    """
    n = values.shape[0]
    # Each block: (mean, count)
    means: list[float] = []
    counts: list[int] = []

    for i in range(n):
        v = float(values[i].item())
        means.append(v)
        counts.append(1)

        # Merge with predecessor while the isotonic constraint is violated.
        while len(means) > 1 and means[-2] > means[-1]:
            m1, c1 = means[-2], counts[-2]
            m2, c2 = means[-1], counts[-1]
            merged_mean = (m1 * c1 + m2 * c2) / (c1 + c2)
            means.pop()
            counts.pop()
            means[-1] = merged_mean
            counts[-1] = c1 + c2

    # Expand blocks back to per-sample values.
    result = []
    for mean, count in zip(means, counts):
        result.extend([mean] * count)

    return torch.tensor(result, dtype=torch.float32)


class IsotonicCalibrator:
    """Post-hoc calibration via isotonic regression (PAVA).

    Fits a non-decreasing step function from sorted model scores to
    calibrated probabilities.  Unseen scores are handled by linear
    interpolation (clamped at the boundary values).
    """

    def __init__(self) -> None:
        self._x_fit: Tensor | None = None
        self._y_fit: Tensor | None = None

    def fit(self, scores: Tensor, labels: Tensor) -> IsotonicCalibrator:
        """Fit the isotonic mapping on calibration data.

        Args:
            scores: (N,) model scores.
            labels: (N,) binary labels in {0, 1}.

        Returns:
            self (for chaining).
        """
        scores = scores.detach().float()
        labels = labels.detach().float()

        if scores.ndim != 1:
            raise ValueError(f"scores must be 1-D, got shape {scores.shape}")
        if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
            raise ValueError("labels must be 1-D with the same length as scores")

        # Sort by score.
        order = torch.argsort(scores)
        sorted_scores = scores[order]
        sorted_labels = labels[order]

        # PAVA on the sorted labels.
        isotonic_values = _pava(sorted_labels)

        self._x_fit = sorted_scores
        self._y_fit = isotonic_values.clamp(0.0, 1.0)
        return self

    def calibrate(self, scores: Tensor) -> Tensor:
        """Map scores to calibrated probabilities using the fitted isotonic mapping.

        Uses linear interpolation; values outside the training range are clamped
        to the boundary calibrated probabilities.

        Args:
            scores: (N,) model scores.

        Returns:
            (N,) probability tensor with values in [0, 1].
        """
        if self._x_fit is None or self._y_fit is None:
            raise RuntimeError("IsotonicCalibrator.fit() must be called before calibrate().")
        if scores.ndim != 1:
            raise ValueError(f"scores must be 1-D, got shape {scores.shape}")

        scores_f = scores.float().contiguous()
        x = self._x_fit
        y = self._y_fit

        # torch.searchsorted requires a sorted 1-D tensor as the first argument.
        # right=False gives the leftmost insertion point.
        idx = torch.searchsorted(x, scores_f)  # in [0, len(x)]
        idx_hi = idx.clamp(1, len(x) - 1)
        idx_lo = (idx_hi - 1).clamp(0, len(x) - 2)

        x_lo = x[idx_lo]
        x_hi = x[idx_hi]
        y_lo = y[idx_lo]
        y_hi = y[idx_hi]

        # Linear interpolation weight; avoid divide-by-zero.
        dx = x_hi - x_lo
        safe_dx = dx.clone()
        safe_dx[safe_dx == 0.0] = 1.0
        t = ((scores_f - x_lo) / safe_dx).clamp(0.0, 1.0)

        # Where x_lo == x_hi, just take y_lo.
        interp = torch.where(dx == 0.0, y_lo, y_lo + t * (y_hi - y_lo))
        return interp.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Calibration Metrics
# ---------------------------------------------------------------------------


class CalibrationMetrics:
    """Static calibration metrics — no fitting required.

    All methods are pure PyTorch; no scipy/sklearn dependency.
    """

    @staticmethod
    def ece(probs: Tensor, labels: Tensor, n_bins: int = 15) -> float:
        """Expected Calibration Error (ECE).

        Partitions [0, 1] into *n_bins* equal-width bins and computes:

            ECE = Σ_m (|B_m| / n) × |acc_m − conf_m|

        Args:
            probs:  (N,) confidence values (max-class probabilities) in [0, 1].
            labels: (N,) binary correctness indicators (1 = correct, 0 = wrong),
                    *or* the predicted-class probabilities for binary tasks.
            n_bins: Number of equal-width bins.

        Returns:
            ECE as a Python float in [0, 1].
        """
        probs = probs.detach().float()
        labels = labels.detach().float()

        if probs.ndim != 1:
            raise ValueError(f"probs must be 1-D, got shape {probs.shape}")
        if labels.shape != probs.shape:
            raise ValueError("probs and labels must have the same shape")

        n = probs.shape[0]
        if n == 0:
            return 0.0

        ece_val = 0.0
        bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)

        for m in range(n_bins):
            lo = float(bin_edges[m].item())
            hi = float(bin_edges[m + 1].item())

            # Include the right edge only for the last bin.
            if m < n_bins - 1:
                mask = (probs >= lo) & (probs < hi)
            else:
                mask = (probs >= lo) & (probs <= hi)

            if mask.sum() == 0:
                continue

            conf_m = probs[mask].mean().item()
            acc_m = labels[mask].mean().item()
            ece_val += (mask.sum().item() / n) * abs(acc_m - conf_m)

        return float(ece_val)

    @staticmethod
    def reliability_diagram_data(
        probs: Tensor,
        labels: Tensor,
        n_bins: int = 15,
    ) -> dict[str, list]:
        """Compute data needed to draw a reliability diagram.

        Args:
            probs:  (N,) confidence values in [0, 1].
            labels: (N,) binary correctness indicators.
            n_bins: Number of equal-width bins.

        Returns:
            dict with keys:
                ``bin_centers``  — list of bin mid-point confidences
                ``accuracies``   — list of mean accuracies per bin (or None)
                ``confidences``  — list of mean confidences per bin (or None)
                ``counts``       — list of sample counts per bin
        """
        probs = probs.detach().float()
        labels = labels.detach().float()

        bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
        bin_centers: list = []
        accuracies: list = []
        confidences: list = []
        counts: list = []

        for m in range(n_bins):
            lo = float(bin_edges[m].item())
            hi = float(bin_edges[m + 1].item())
            center = (lo + hi) / 2.0

            if m < n_bins - 1:
                mask = (probs >= lo) & (probs < hi)
            else:
                mask = (probs >= lo) & (probs <= hi)

            count = int(mask.sum().item())
            bin_centers.append(center)
            counts.append(count)

            if count == 0:
                accuracies.append(None)
                confidences.append(None)
            else:
                accuracies.append(float(labels[mask].mean().item()))
                confidences.append(float(probs[mask].mean().item()))

        return {
            "bin_centers": bin_centers,
            "accuracies": accuracies,
            "confidences": confidences,
            "counts": counts,
        }
