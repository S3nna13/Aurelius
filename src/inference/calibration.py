"""Post-hoc calibration methods for language model confidence scores.

LLMs are often poorly calibrated — their confidence doesn't match actual accuracy.
This module provides:
- Expected Calibration Error (ECE) evaluation
- Temperature scaling (single-parameter post-hoc calibration)
- Top-K threshold calibration for selective prediction
- Reliability diagram data
- Sequence-level calibration via token log-prob aggregation
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CalibrationResult:
    ece: float  # Expected Calibration Error (lower is better)
    mce: float  # Maximum Calibration Error
    brier_score: float  # Brier score (lower is better)
    n_samples: int
    bin_confidences: list[float]  # mean confidence per bin
    bin_accuracies: list[float]  # mean accuracy per bin
    bin_counts: list[int]  # samples per bin


def expected_calibration_error(
    confidences: torch.Tensor,  # (N,) predicted probabilities for selected class
    correctness: torch.Tensor,  # (N,) 0/1 whether prediction was correct
    n_bins: int = 15,
) -> CalibrationResult:
    """Compute ECE by binning predictions by confidence.

    ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|

    Also computes MCE = max_b |acc(B_b) - conf(B_b)|
    Brier score = mean((confidence - correctness)^2)

    Bins are equal-width over [0, 1]: [0, 1/n_bins), [1/n_bins, 2/n_bins), ...
    """
    confidences = confidences.float()
    correctness = correctness.float()
    n = confidences.shape[0]

    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    ece = 0.0
    mce = 0.0

    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        # Include upper boundary in last bin
        if b < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        count = mask.sum().item()
        if count > 0:
            bin_conf = confidences[mask].mean().item()
            bin_acc = correctness[mask].mean().item()
            gap = abs(bin_acc - bin_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
        else:
            bin_conf = (lo + hi) / 2.0
            bin_acc = 0.0

        bin_confidences.append(bin_conf)
        bin_accuracies.append(bin_acc)
        bin_counts.append(int(count))

    brier_score = ((confidences - correctness) ** 2).mean().item()

    return CalibrationResult(
        ece=ece,
        mce=mce,
        brier_score=brier_score,
        n_samples=n,
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
    )


class TemperatureCalibrator(nn.Module):
    """Single-parameter temperature scaling for LLM logits.

    Trained by minimizing NLL on a held-out validation set.
    After calibration: calibrated_logits = logits / T

    Log temperature parameterization ensures T > 0 always.

    Args:
        init_temperature: initial temperature value (default 1.5)
    """

    def __init__(self, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.log_T = nn.Parameter(torch.log(torch.tensor(init_temperature)))

    @property
    def temperature(self) -> float:
        return self.log_T.exp().item()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits: return logits / temperature"""
        return logits / self.log_T.exp()

    def fit(
        self,
        logits: torch.Tensor,  # (N, V) or (N, T, V) validation logits
        labels: torch.Tensor,  # (N,) or (N, T) true labels
        n_steps: int = 200,
        lr: float = 0.01,
    ) -> list[float]:
        """Optimize temperature via LBFGS on NLL.

        Returns list of per-step NLL losses.
        """
        # Flatten sequence dimension if needed: (N, T, V) -> (N*T, V)
        if logits.dim() == 3:
            n, t, v = logits.shape
            logits_flat = logits.reshape(n * t, v)
            labels_flat = labels.reshape(n * t)
        else:
            logits_flat = logits
            labels_flat = labels

        logits_flat = logits_flat.detach()
        labels_flat = labels_flat.detach()

        optimizer = torch.optim.LBFGS([self.log_T], lr=lr, max_iter=20)
        losses: list[float] = []

        for _ in range(n_steps):

            def closure():
                optimizer.zero_grad()
                scaled = logits_flat / self.log_T.exp()
                loss = F.cross_entropy(scaled, labels_flat)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            losses.append(loss.item() if hasattr(loss, "item") else float(loss))

        return losses


class TopKCalibrator:
    """Calibrates confidence by adjusting the threshold for high-confidence predictions.

    Find threshold tau such that predictions with max_prob > tau are
    correct `target_precision` fraction of the time.

    Args:
        target_precision: desired precision on selective predictions (default 0.9)
    """

    def __init__(self, target_precision: float = 0.9) -> None:
        self.target_precision = target_precision
        self.threshold: float = 0.5

    def fit(
        self,
        confidences: torch.Tensor,
        correctness: torch.Tensor,
    ) -> float:
        """Binary search for threshold tau.

        Returns fitted tau.
        """
        confidences = confidences.float()
        correctness = correctness.float()

        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            mask = confidences > mid
            if mask.sum() == 0:
                # No predictions above threshold — lower it
                hi = mid
                continue
            precision = correctness[mask].mean().item()
            if precision < self.target_precision:
                lo = mid
            else:
                hi = mid

        self.threshold = (lo + hi) / 2.0
        return self.threshold

    def predict(self, confidence: float) -> bool:
        """Return True if should predict (above threshold)."""
        return confidence > self.threshold


def reliability_diagram_data(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    n_bins: int = 10,
) -> dict:
    """Data for plotting reliability diagram.

    Returns:
        {
            'bin_centers': list[float],   # e.g., [0.05, 0.15, ..., 0.95]
            'bin_accuracies': list[float],
            'bin_confidences': list[float],
            'bin_counts': list[int],
            'ece': float,
        }
    """
    result = expected_calibration_error(confidences, correctness, n_bins=n_bins)

    bin_centers = [(b + 0.5) / n_bins for b in range(n_bins)]

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": result.bin_accuracies,
        "bin_confidences": result.bin_confidences,
        "bin_counts": result.bin_counts,
        "ece": result.ece,
    }


class SequenceCalibrator:
    """Sequence-level calibration: calibrate the probability of an entire generation.

    Methods:
    - "length_penalty": adjust by exp(-mean_token_prob) * length_penalty_factor
    - "product": sequence_prob = product of token probabilities (sum of log probs)
    - "mean_log": sequence_prob = exp(mean of log token probs)

    Args:
        method: aggregation method (default "mean_log")
        temperature: applied per-token before aggregation (default 1.0)
    """

    def __init__(self, method: str = "mean_log", temperature: float = 1.0) -> None:
        self.method = method
        self.temperature = temperature

    def score(self, token_log_probs: list[float]) -> float:
        """Aggregate per-token log probs into a sequence score.

        Returns float (higher = more probable).
        """
        if not token_log_probs:
            return float("-inf")

        # Apply temperature scaling in log domain: log_prob / T
        scaled = [lp / self.temperature for lp in token_log_probs]

        if self.method == "product":
            # Log-domain product = sum of log probs
            return sum(scaled)
        elif self.method == "mean_log":
            # Mean of log probs (length-normalized)
            return sum(scaled) / len(scaled)
        elif self.method == "length_penalty":
            sum(scaled) / len(scaled)
            # Penalize: higher mean_log_prob (less negative) is better
            # length_penalty: adjust by sequence length
            length_factor = len(token_log_probs) ** 0.6
            return sum(scaled) / length_factor
        else:
            raise ValueError(
                f"Unknown method: {self.method!r}. Choose from 'product', 'mean_log', 'length_penalty'."  # noqa: E501
            )

    def compare(
        self,
        candidates: list[list[float]],  # list of token_log_probs lists
    ) -> int:
        """Return index of highest-scoring candidate."""
        scores = [self.score(c) for c in candidates]
        return int(max(range(len(scores)), key=lambda i: scores[i]))
