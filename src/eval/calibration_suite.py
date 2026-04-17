"""Calibration Evaluation Suite — ECE, reliability diagrams, Brier score."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


class CalibrationBins:
    """Equal-width confidence binning for calibration evaluation."""

    def __init__(self, n_bins: int = 10) -> None:
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        self.n_bins = n_bins

    def bin(self, confidences: Tensor, correct: Tensor) -> Dict[str, Tensor]:
        """Bin samples by confidence and compute per-bin statistics.

        Args:
            confidences: (N,) float tensor in [0, 1].
            correct: (N,) bool/int tensor — 1 if prediction is correct.

        Returns:
            Dict with keys:
                'bin_confidences': (n_bins,) mean confidence per bin.
                'bin_accuracies':  (n_bins,) mean accuracy per bin.
                'bin_counts':      (n_bins,) number of samples per bin.
        """
        confidences = confidences.float()
        correct = correct.float()
        n = confidences.shape[0]

        bin_confidences = torch.zeros(self.n_bins, dtype=torch.float)
        bin_accuracies = torch.zeros(self.n_bins, dtype=torch.float)
        bin_counts = torch.zeros(self.n_bins, dtype=torch.long)

        # Edges: [0, 1/n_bins, 2/n_bins, ..., 1]
        edges = torch.linspace(0.0, 1.0, self.n_bins + 1)

        for b in range(self.n_bins):
            lo = edges[b].item()
            hi = edges[b + 1].item()
            # Include right edge in last bin
            if b < self.n_bins - 1:
                mask = (confidences >= lo) & (confidences < hi)
            else:
                mask = (confidences >= lo) & (confidences <= hi)

            count = mask.sum().item()
            bin_counts[b] = int(count)
            if count > 0:
                bin_confidences[b] = confidences[mask].mean()
                bin_accuracies[b] = correct[mask].mean()

        return {
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
        }

    def ece(self, confidences: Tensor, correct: Tensor) -> float:
        """Expected Calibration Error.

        ECE = sum_b |acc_b - conf_b| * (n_b / N)
        """
        n = confidences.shape[0]
        if n == 0:
            return 0.0
        data = self.bin(confidences, correct)
        counts = data["bin_counts"].float()
        weights = counts / float(n)
        gap = (data["bin_accuracies"] - data["bin_confidences"]).abs()
        return float((gap * weights).sum().item())


class ReliabilityDiagram:
    """Calibration curve data including ECE, MCE, and overconfidence stats."""

    def __init__(self, bins: CalibrationBins) -> None:
        self.bins = bins

    def compute(self, confidences: Tensor, correct: Tensor) -> Dict:
        """Compute reliability diagram data.

        Args:
            confidences: (N,) float tensor in [0, 1].
            correct: (N,) bool/int tensor.

        Returns:
            Dict with bin data plus:
                'ece': Expected Calibration Error.
                'mce': Maximum Calibration Error (max per-bin gap).
                'overconfident_fraction': fraction of bins where conf > acc
                    (weighted by count).
        """
        data = self.bins.bin(confidences, correct)
        counts = data["bin_counts"].float()
        total = counts.sum().item()

        gap = (data["bin_accuracies"] - data["bin_confidences"]).abs()

        # ECE
        if total > 0:
            weights = counts / total
            ece = float((gap * weights).sum().item())
        else:
            ece = 0.0

        # MCE — maximum gap across non-empty bins
        nonempty = data["bin_counts"] > 0
        if nonempty.any():
            mce = float(gap[nonempty].max().item())
        else:
            mce = 0.0

        # Overconfident fraction: fraction of samples in bins where conf > acc
        overconf_mask = data["bin_confidences"] > data["bin_accuracies"]
        if total > 0:
            overconf_fraction = float((counts[overconf_mask].sum() / total).item())
        else:
            overconf_fraction = 0.0

        return {
            **data,
            "ece": ece,
            "mce": mce,
            "overconfident_fraction": overconf_fraction,
        }


class BrierScore:
    """Proper scoring rule for probabilistic predictions."""

    def __call__(self, probs: Tensor, labels: Tensor) -> float:
        """Compute Brier score.

        Args:
            probs: (N,) for binary or (N, C) for multiclass.
            labels: (N,) int class indices.

        Returns:
            float Brier score.
                Binary   (N,):   mean((p - y)^2)
                Multiclass (N,C): mean(sum_c (p_c - y_c)^2)
        """
        probs = probs.float()
        labels = labels.long()

        if probs.dim() == 1:
            # Binary case
            y = labels.float()
            return float(((probs - y) ** 2).mean().item())
        else:
            # Multiclass case
            n, c = probs.shape
            # One-hot encode labels
            y_onehot = torch.zeros(n, c, dtype=torch.float, device=probs.device)
            y_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
            per_sample = ((probs - y_onehot) ** 2).sum(dim=1)
            return float(per_sample.mean().item())


class CalibrationEvaluator:
    """Runs all calibration metrics on a set of logits and labels."""

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self._bins = CalibrationBins(n_bins=n_bins)
        self._diagram = ReliabilityDiagram(bins=self._bins)
        self._brier = BrierScore()

    def evaluate(self, logits: Tensor, labels: Tensor) -> Dict[str, float]:
        """Compute all calibration metrics.

        Args:
            logits: (N, C) raw logits.
            labels: (N,) int class indices.

        Returns:
            Dict with keys: 'ece', 'mce', 'brier', 'overconfident_fraction',
            'n_samples'.
        """
        logits = logits.float()
        labels = labels.long()

        probs = F.softmax(logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        correct = (predictions == labels).float()

        diag = self._diagram.compute(confidences, correct)
        brier = self._brier(probs, labels)

        return {
            "ece": diag["ece"],
            "mce": diag["mce"],
            "brier": brier,
            "overconfident_fraction": diag["overconfident_fraction"],
            "n_samples": float(labels.shape[0]),
        }
