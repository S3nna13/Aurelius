"""Token-level confidence calibration for AureliusTransformer.

This module provides tools to measure and improve how well a model's
confidence scores reflect actual accuracy at the token level.

Components:
- CalibrationConfig: configuration dataclass
- temperature_scale: divide logits by a temperature scalar
- compute_token_confidence: max softmax probability per token position
- compute_ece: Expected Calibration Error scalar
- compute_reliability_diagram: bin-level data for reliability diagrams
- TemperatureScaler: grid-search temperature fitting + transform
- CalibrationEvaluator: evaluate a model's calibration on a data iterator
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    n_bins: int = 10
    temperature: float = 1.0
    use_ece: bool = True
    smoothing: float = 0.0


# ---------------------------------------------------------------------------
# Core tensor utilities
# ---------------------------------------------------------------------------

def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by temperature.

    Args:
        logits: arbitrary shape tensor of logit values
        temperature: positive scalar (1.0 = no change)

    Returns:
        Tensor of same shape as logits.
    """
    return logits / temperature


def compute_token_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token confidence as the max softmax probability.

    Args:
        logits: (B, T, V) logits tensor

    Returns:
        (B, T) tensor of max softmax probability in [0, 1]
    """
    probs = F.softmax(logits, dim=-1)          # (B, T, V)
    confidence, _ = probs.max(dim=-1)          # (B, T)
    return confidence


# ---------------------------------------------------------------------------
# ECE and reliability diagram
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: torch.Tensor,
    correct: torch.Tensor,
    n_bins: int,
) -> float:
    """Expected Calibration Error.

    Partitions predictions into n_bins equal-width confidence bins,
    computes |accuracy - mean_confidence| per bin, and returns the
    weighted average (weighted by bin fraction of total samples).

    Args:
        confidences: (N,) predicted max probabilities
        correct: (N,) float tensor of 0/1 correctness indicators
        n_bins: number of equal-width bins in [0, 1]

    Returns:
        ECE as a Python float.
    """
    confidences = confidences.float()
    correct = correct.float()
    n = confidences.shape[0]

    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        count = mask.sum().item()
        if count > 0:
            bin_conf = confidences[mask].mean().item()
            bin_acc = correct[mask].mean().item()
            ece += (count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_reliability_diagram(
    confidences: torch.Tensor,
    correct: torch.Tensor,
    n_bins: int,
) -> dict:
    """Compute data for a reliability diagram.

    Args:
        confidences: (N,) predicted max probabilities
        correct: (N,) float tensor of 0/1 correctness indicators
        n_bins: number of equal-width bins in [0, 1]

    Returns:
        dict with keys:
          "bin_confidences": list of length n_bins (mean confidence per bin)
          "bin_accuracies":  list of length n_bins (mean accuracy per bin)
          "bin_counts":      list of length n_bins (sample count per bin)
    """
    confidences = confidences.float()
    correct = correct.float()

    bin_confidences: list[float] = []
    bin_accuracies: list[float] = []
    bin_counts: list[int] = []

    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        count = int(mask.sum().item())
        if count > 0:
            bin_conf = confidences[mask].mean().item()
            bin_acc = correct[mask].mean().item()
        else:
            bin_conf = (lo + hi) / 2.0
            bin_acc = 0.0

        bin_confidences.append(bin_conf)
        bin_accuracies.append(bin_acc)
        bin_counts.append(count)

    return {
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


# ---------------------------------------------------------------------------
# TemperatureScaler
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """Grid-search temperature scaler.

    fit() selects the temperature from a candidate grid that minimises
    negative log-likelihood on a list of (logits, labels) pairs.
    transform() applies the fitted temperature to new logits.

    Args:
        config: CalibrationConfig (uses config.temperature as initial value)
    """

    _GRID = [0.5, 0.75, 1.0, 1.5, 2.0]

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config
        self._temperature: float = config.temperature

    @property
    def temperature(self) -> float:
        return self._temperature

    def fit(
        self,
        logits_list: list[torch.Tensor],
        labels_list: list[torch.Tensor],
    ) -> None:
        """Find the optimal temperature via grid search over self._GRID.

        For each candidate temperature T, compute the mean NLL across all
        (logits, labels) pairs and keep T with the lowest NLL.

        Args:
            logits_list: list of (N, V) or (B, T, V) logit tensors
            labels_list: list of (N,) or (B, T) label tensors (LongTensor)
        """
        best_T = self._temperature
        best_nll = float("inf")

        for T in self._GRID:
            total_nll = 0.0
            total_n = 0
            for logits, labels in zip(logits_list, labels_list):
                logits_f = logits.detach().float()
                labels_l = labels.detach()

                # Flatten potential (B, T, V) -> (B*T, V)
                if logits_f.dim() == 3:
                    B, Tlen, V = logits_f.shape
                    logits_f = logits_f.reshape(B * Tlen, V)
                    labels_l = labels_l.reshape(B * Tlen)

                scaled = logits_f / T
                nll = F.cross_entropy(scaled, labels_l, reduction="sum").item()
                total_nll += nll
                total_n += labels_l.shape[0]

            mean_nll = total_nll / max(total_n, 1)
            if mean_nll < best_nll:
                best_nll = mean_nll
                best_T = T

        self._temperature = best_T

    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the fitted temperature to logits.

        Args:
            logits: arbitrary-shape logit tensor

        Returns:
            Tensor of same shape scaled by fitted temperature.
        """
        return temperature_scale(logits, self._temperature)


# ---------------------------------------------------------------------------
# CalibrationEvaluator
# ---------------------------------------------------------------------------

class CalibrationEvaluator:
    """Evaluate a model's token-level calibration on a data iterator.

    The model must follow the Aurelius API:
        loss, logits, pkv = model(input_ids)

    Args:
        config: CalibrationConfig controlling ECE bin count etc.
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()

    def evaluate(
        self,
        model: Any,
        data_iter: Iterator[torch.Tensor],
    ) -> dict:
        """Run model on all batches and compute calibration metrics.

        Each item from data_iter should be a (B, T) LongTensor of token IDs.
        We use input_ids[:, :-1] as context and input_ids[:, 1:] as targets
        so that every position has a known next-token label.

        Args:
            model: callable with signature (input_ids) -> (loss, logits, pkv)
            data_iter: iterable of (B, T) input_id tensors

        Returns:
            dict with keys:
              "ece":             float ECE across all tokens
              "mean_confidence": float mean of max softmax probability
              "accuracy":        float fraction of tokens predicted correctly
        """
        all_confidences: list[torch.Tensor] = []
        all_correct: list[torch.Tensor] = []

        model.eval()
        with torch.no_grad():
            for batch in data_iter:
                # batch: (B, T)
                if batch.shape[1] < 2:
                    continue

                input_ids = batch[:, :-1]   # (B, T-1)
                targets = batch[:, 1:]       # (B, T-1)

                _loss, logits, _pkv = model(input_ids)
                # logits: (B, T-1, V)

                confidence = compute_token_confidence(logits)  # (B, T-1)
                preds = logits.argmax(dim=-1)                  # (B, T-1)
                correct = (preds == targets).float()           # (B, T-1)

                all_confidences.append(confidence.reshape(-1))
                all_correct.append(correct.reshape(-1))

        if not all_confidences:
            return {"ece": 0.0, "mean_confidence": 0.0, "accuracy": 0.0}

        confidences_cat = torch.cat(all_confidences)
        correct_cat = torch.cat(all_correct)

        ece = compute_ece(confidences_cat, correct_cat, self.config.n_bins)
        mean_confidence = confidences_cat.mean().item()
        accuracy = correct_cat.mean().item()

        return {
            "ece": ece,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
        }
