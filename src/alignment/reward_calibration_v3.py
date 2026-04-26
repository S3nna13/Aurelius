"""Reward Model Calibration v3: Temperature Scaling and Reliability Diagrams.

Calibrates reward model outputs so that reward differences correspond to
meaningful probability differences for preference prediction.

Distinct from reward_calibration.py (Platt/isotonic) and v2 (lstsq linear).
This module focuses on temperature scaling and ECE-based evaluation.

References:
    Guo et al. 2017 (Temperature Scaling) — https://arxiv.org/abs/1706.04599
    Niculescu-Mizil & Caruana 2005 (Calibration methods)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class TemperatureScaler(nn.Module):
    """Post-hoc calibration via a single learned temperature parameter.

    calibrated_score = reward / abs(temperature)

    Temperature is learned by minimizing NLL (BCEWithLogitsLoss) on a
    validation set of (reward, binary_label) pairs.
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature]))

    def forward(self, rewards: Tensor) -> Tensor:
        """Divide rewards by absolute temperature.

        Args:
            rewards: (N,) raw reward scores.

        Returns:
            (N,) temperature-scaled rewards.
        """
        return rewards / self.temperature.abs()

    def fit(
        self,
        rewards: Tensor,
        labels: Tensor,
        n_epochs: int = 100,
        lr: float = 0.01,
    ) -> float:
        """Fit temperature by minimizing BCEWithLogitsLoss.

        Args:
            rewards: (N,) raw reward scores.
            labels:  (N,) binary labels (1=positive, 0=negative).
            n_epochs: number of Adam optimisation steps.
            lr: learning rate.

        Returns:
            Final temperature value as a positive float.
        """
        rewards = rewards.detach().float()
        labels = labels.detach().float()

        # Reset temperature to 1.0 before fitting
        with torch.no_grad():
            self.temperature.fill_(1.0)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(n_epochs):
            optimizer.zero_grad()
            scaled = rewards / self.temperature.abs()
            loss = loss_fn(scaled, labels)
            loss.backward()
            optimizer.step()

        return self.temperature_value()

    def temperature_value(self) -> float:
        """Return the current temperature as a positive float."""
        return abs(self.temperature.item())


class HistogramBinCalibrator:
    """Non-parametric calibration via equal-width histogram binning.

    Each bin stores the mean label of training samples that fell into it.
    At inference time each score is mapped to its bin's stored value.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self.bin_edges: Tensor | None = None
        self.bin_values: Tensor | None = None

    def fit(self, scores: Tensor, labels: Tensor) -> None:
        """Fit bin boundaries and per-bin calibration values.

        Args:
            scores: (N,) raw scores.
            labels: (N,) binary labels in {0, 1}.
        """
        scores = scores.detach().float()
        labels = labels.detach().float()

        min_val = scores.min().item()
        max_val = scores.max().item()

        # Avoid degenerate case where all scores are identical
        if min_val == max_val:
            max_val = min_val + 1.0

        self.bin_edges = torch.linspace(min_val, max_val, self.n_bins + 1)

        bin_values = torch.zeros(self.n_bins)
        for b in range(self.n_bins):
            lo = self.bin_edges[b].item()
            hi = self.bin_edges[b + 1].item()
            if b < self.n_bins - 1:
                mask = (scores >= lo) & (scores < hi)
            else:
                # Last bin is inclusive on right edge
                mask = (scores >= lo) & (scores <= hi)

            if mask.sum() > 0:
                bin_values[b] = labels[mask].mean()
            else:
                bin_values[b] = 0.0

        self.bin_values = bin_values

    def calibrate(self, scores: Tensor) -> Tensor:
        """Map each score to its bin's calibration value.

        Args:
            scores: (N,) scores to calibrate.

        Returns:
            (N,) calibrated values.
        """
        if self.bin_edges is None or self.bin_values is None:
            raise RuntimeError("HistogramBinCalibrator must be fit before calibrating.")

        scores = scores.detach().float()
        out = torch.zeros(scores.shape[0])

        min_edge = self.bin_edges[0].item()
        max_edge = self.bin_edges[-1].item()

        for i, s in enumerate(scores):
            val = s.item()
            # Clip to training range
            val_clipped = max(min_edge, min(max_edge, val))

            # Find bin index
            bin_idx = int((val_clipped - min_edge) / (max_edge - min_edge) * self.n_bins)
            # Clamp to valid range
            bin_idx = min(bin_idx, self.n_bins - 1)
            out[i] = self.bin_values[bin_idx]

        return out


class ReliabilityDiagram:
    """Computes ECE and per-bin reliability statistics.

    Used to assess calibration quality and produce data for reliability plots.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def compute(self, probs: Tensor, labels: Tensor) -> dict[str, Any]:
        """Compute ECE and per-bin statistics.

        Args:
            probs:  (N,) predicted probabilities in [0, 1].
            labels: (N,) binary true labels in {0, 1}.

        Returns:
            Dict with keys:
                'ece'             : float, Expected Calibration Error.
                'bin_confidences' : List[float], mean predicted prob per bin.
                'bin_accuracies'  : List[float], fraction positive per bin.
                'bin_counts'      : List[int], number of samples per bin.
        """
        probs = probs.detach().float()
        labels = labels.detach().float()
        n = probs.shape[0]

        bin_confidences: list[float] = []
        bin_accuracies: list[float] = []
        bin_counts: list[int] = []

        bin_edges = torch.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0

        for b in range(self.n_bins):
            lo = bin_edges[b].item()
            hi = bin_edges[b + 1].item()
            if b < self.n_bins - 1:
                mask = (probs >= lo) & (probs < hi)
            else:
                mask = (probs >= lo) & (probs <= hi)

            count = int(mask.sum().item())
            bin_counts.append(count)

            if count > 0:
                mean_conf = probs[mask].mean().item()
                mean_acc = labels[mask].mean().item()
            else:
                mean_conf = 0.0
                mean_acc = 0.0

            bin_confidences.append(mean_conf)
            bin_accuracies.append(mean_acc)

            if count > 0:
                ece += (count / n) * abs(mean_conf - mean_acc)

        return {
            "ece": float(ece),
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
        }


class PreferenceCalibrator:
    """Calibrates reward differences for preference prediction.

    Wraps a TemperatureScaler to convert raw winner/loser rewards into a
    probability that the winner's reward truly exceeds the loser's.
    """

    def __init__(self, temperature_scaler: TemperatureScaler) -> None:
        self.temperature_scaler = temperature_scaler

    def preference_prob(self, reward_w: Tensor, reward_l: Tensor) -> Tensor:
        """Compute calibrated preference probability P(w > l).

        Args:
            reward_w: (N,) reward scores for the preferred (winner) response.
            reward_l: (N,) reward scores for the rejected (loser) response.

        Returns:
            (N,) probabilities in [0, 1] that w is preferred over l.
        """
        calibrated_w = self.temperature_scaler(reward_w)
        calibrated_l = self.temperature_scaler(reward_l)
        return torch.sigmoid(calibrated_w - calibrated_l)

    def preference_calibration_error(self, reward_w: Tensor, reward_l: Tensor) -> float:
        """ECE of the preference predictor treating every pair as label=1.

        The winner is always the "positive" class, so ground-truth labels are
        all 1.  A well-calibrated predictor should return probabilities close
        to 1 for clear wins and close to 0.5 for ambiguous pairs.

        Args:
            reward_w: (N,) reward scores for winner.
            reward_l: (N,) reward scores for loser.

        Returns:
            ECE as a float in [0, 1].
        """
        probs = self.preference_prob(reward_w, reward_l).detach()
        labels = torch.ones(probs.shape[0])
        diagram = ReliabilityDiagram(n_bins=10)
        result = diagram.compute(probs, labels)
        return float(result["ece"])
