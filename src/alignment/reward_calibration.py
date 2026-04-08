"""Reward model calibration: map raw scores to calibrated probabilities.

Raw reward model scores are often miscalibrated — they don't correspond to true
"probability of being good." Calibration maps raw scores to calibrated
probabilities using held-out data, improving downstream RLHF training.

Three methods are provided:
- TemperatureScaling: single learned temperature parameter (1 DoF)
- PlattScaling: affine transform before sigmoid (2 DoF)
- IsotonicCalibration: non-parametric monotone regression (PAVA)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaling(nn.Module):
    """Post-hoc calibration via a single learned temperature parameter.

    calibrated_score = score / temperature

    Optimal temperature is found by minimizing NLL on a validation set.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling.

        Args:
            scores: (N,) raw logits

        Returns:
            (N,) calibrated logits
        """
        return scores / self.temperature.clamp(min=1e-8)

    def fit(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        n_steps: int = 100,
    ) -> None:
        """Fit temperature on validation data.

        Minimizes binary cross-entropy: BCE(sigmoid(scores/T), labels).

        Args:
            scores: (N,) raw reward logits
            labels: (N,) binary labels (1=good, 0=bad)
            lr: learning rate for optimizer
            n_steps: number of optimization steps
        """
        # Reset temperature to 1.0 before fitting
        with torch.no_grad():
            self.temperature.fill_(1.0)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=n_steps)
        scores_detached = scores.detach()
        labels_detached = labels.float().detach()

        def closure():
            optimizer.zero_grad()
            calibrated = scores_detached / self.temperature.clamp(min=1e-8)
            loss = F.binary_cross_entropy_with_logits(calibrated, labels_detached)
            loss.backward()
            return loss

        optimizer.step(closure)


class PlattScaling(nn.Module):
    """Platt scaling: learned affine transformation before sigmoid.

    calibrated_prob = sigmoid(a * score + b)

    More expressive than temperature scaling (2 params vs 1).
    """

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply Platt scaling.

        Args:
            scores: (N,) raw logits

        Returns:
            (N,) calibrated probabilities in (0, 1)
        """
        return torch.sigmoid(self.a * scores + self.b)

    def fit(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        n_steps: int = 100,
    ) -> None:
        """Minimize BCE loss to find optimal a, b.

        Args:
            scores: (N,) raw reward logits
            labels: (N,) binary labels (1=good, 0=bad)
            lr: learning rate for optimizer
            n_steps: number of optimization steps
        """
        # Reset parameters before fitting
        with torch.no_grad():
            self.a.fill_(1.0)
            self.b.fill_(0.0)

        optimizer = torch.optim.LBFGS([self.a, self.b], lr=lr, max_iter=n_steps)
        scores_detached = scores.detach()
        labels_detached = labels.float().detach()

        def closure():
            optimizer.zero_grad()
            logits = self.a * scores_detached + self.b
            loss = F.binary_cross_entropy_with_logits(logits, labels_detached)
            loss.backward()
            return loss

        optimizer.step(closure)


class IsotonicCalibration:
    """Non-parametric isotonic regression calibration.

    Fits a monotone non-decreasing function from raw scores to probabilities
    using the Pool Adjacent Violators Algorithm (PAVA).
    """

    def __init__(self) -> None:
        self._x_breakpoints: torch.Tensor | None = None
        self._y_breakpoints: torch.Tensor | None = None

    def fit(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        """Fit isotonic regression using PAVA.

        Args:
            scores: (N,) raw scores
            labels: (N,) binary labels
        """
        scores_np = scores.detach().float()
        labels_np = labels.detach().float()

        # Sort by score
        order = torch.argsort(scores_np)
        x_sorted = scores_np[order]
        y_sorted = labels_np[order]

        # PAVA: pool adjacent violators
        # Each block tracks (sum_y, count, x_representative)
        blocks: list[list] = []  # each entry: [sum_y, count, list_of_x]

        for i in range(len(x_sorted)):
            xi = x_sorted[i].item()
            yi = y_sorted[i].item()
            # Start a new block for this point
            blocks.append([yi, 1, [xi]])

            # Merge while the previous block has a higher mean (violates monotonicity)
            while len(blocks) > 1 and (blocks[-2][0] / blocks[-2][1]) > (blocks[-1][0] / blocks[-1][1]):
                prev = blocks.pop(-2)
                curr = blocks[-1]
                curr[0] += prev[0]
                curr[1] += prev[1]
                curr[2] = prev[2] + curr[2]

        # Extract breakpoints: use mean x per block, mean y per block
        x_pts = []
        y_pts = []
        for block in blocks:
            sum_y, count, xs = block
            x_pts.append(sum(xs) / len(xs))
            y_pts.append(sum_y / count)

        self._x_breakpoints = torch.tensor(x_pts, dtype=torch.float32)
        self._y_breakpoints = torch.tensor(y_pts, dtype=torch.float32)

    def predict(self, scores: torch.Tensor) -> torch.Tensor:
        """Interpolate calibrated probabilities for new scores.

        Uses linear interpolation between fitted breakpoints.

        Args:
            scores: (N,) raw scores

        Returns:
            (N,) probabilities in [0, 1]
        """
        if self._x_breakpoints is None or self._y_breakpoints is None:
            raise RuntimeError("Must call fit() before predict().")

        x_bp = self._x_breakpoints
        y_bp = self._y_breakpoints
        scores_f = scores.detach().float()

        result = torch.empty_like(scores_f)
        for i, s in enumerate(scores_f):
            s_val = s.item()
            if s_val <= x_bp[0].item():
                result[i] = y_bp[0]
            elif s_val >= x_bp[-1].item():
                result[i] = y_bp[-1]
            else:
                # Find bracket
                idx = torch.searchsorted(x_bp, torch.tensor(s_val)).item()
                idx = max(1, min(idx, len(x_bp) - 1))
                x0, x1 = x_bp[idx - 1].item(), x_bp[idx].item()
                y0, y1 = y_bp[idx - 1].item(), y_bp[idx].item()
                if x1 == x0:
                    result[i] = (y0 + y1) / 2.0
                else:
                    t = (s_val - x0) / (x1 - x0)
                    result[i] = y0 + t * (y1 - y0)

        return result.clamp(0.0, 1.0)


class CalibrationEvaluator:
    """Evaluate calibration quality."""

    def expected_calibration_error(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 10,
    ) -> float:
        """ECE: weighted average of |accuracy - confidence| across bins.

        1. Bin predictions by predicted probability [0,0.1), [0.1,0.2), ...
        2. For each bin: |mean(labels) - mean(probs)|
        3. Weight by bin size / total size
        4. Sum weighted errors

        Args:
            probs: (N,) predicted probabilities in [0, 1]
            labels: (N,) binary labels

        Returns:
            ECE as a float in [0, 1]
        """
        probs = probs.detach().float()
        labels = labels.detach().float()
        n = len(probs)

        bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0

        for b in range(n_bins):
            lo = bin_boundaries[b].item()
            hi = bin_boundaries[b + 1].item()

            if b == n_bins - 1:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)

            if mask.sum() == 0:
                continue

            bin_probs = probs[mask]
            bin_labels = labels[mask]
            bin_size = mask.sum().item()

            confidence = bin_probs.mean().item()
            accuracy = bin_labels.mean().item()

            ece += (bin_size / n) * abs(accuracy - confidence)

        return float(ece)

    def reliability_diagram_data(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 10,
    ) -> dict:
        """Return data for plotting reliability diagram.

        Args:
            probs: (N,) predicted probabilities in [0, 1]
            labels: (N,) binary labels
            n_bins: number of bins

        Returns:
            dict with keys: bin_centers, bin_accuracy, bin_confidence, bin_sizes
        """
        probs = probs.detach().float()
        labels = labels.detach().float()

        bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = []
        bin_accuracy = []
        bin_confidence = []
        bin_sizes = []

        for b in range(n_bins):
            lo = bin_boundaries[b].item()
            hi = bin_boundaries[b + 1].item()
            center = (lo + hi) / 2.0

            if b == n_bins - 1:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)

            bin_centers.append(center)
            if mask.sum() == 0:
                bin_accuracy.append(0.0)
                bin_confidence.append(center)
                bin_sizes.append(0)
            else:
                bin_accuracy.append(labels[mask].mean().item())
                bin_confidence.append(probs[mask].mean().item())
                bin_sizes.append(mask.sum().item())

        return {
            "bin_centers": bin_centers,
            "bin_accuracy": bin_accuracy,
            "bin_confidence": bin_confidence,
            "bin_sizes": bin_sizes,
        }

    def brier_score(self, probs: torch.Tensor, labels: torch.Tensor) -> float:
        """Brier score: mean((prob - label)^2). Lower is better.

        Args:
            probs: (N,) predicted probabilities in [0, 1]
            labels: (N,) binary labels

        Returns:
            Brier score as a float in [0, 1]
        """
        probs = probs.detach().float()
        labels = labels.detach().float()
        return float(((probs - labels) ** 2).mean().item())


def calibrate_reward_model(
    reward_model,
    val_prompts: list[torch.Tensor],
    val_rewards: torch.Tensor,
    val_labels: torch.Tensor,
    method: str = "temperature",
) -> tuple:
    """Convenience function: calibrate a reward model using validation data.

    Args:
        reward_model: callable(prompt) -> scalar score
        val_prompts: list of prompt tensors (not used if val_rewards already computed)
        val_rewards: (N,) raw reward scores already computed
        val_labels: (N,) binary preference labels
        method: 'temperature', 'platt', or 'isotonic'

    Returns:
        (calibrator, ece_before, ece_after)
    """
    evaluator = CalibrationEvaluator()

    # Compute ECE before calibration using sigmoid of raw scores
    raw_probs = torch.sigmoid(val_rewards.float())
    ece_before = evaluator.expected_calibration_error(raw_probs, val_labels)

    # Create and fit calibrator
    if method == "temperature":
        calibrator = TemperatureScaling()
        calibrator.fit(val_rewards, val_labels)
        calibrated_logits = calibrator(val_rewards)
        calibrated_probs = torch.sigmoid(calibrated_logits)
    elif method == "platt":
        calibrator = PlattScaling()
        calibrator.fit(val_rewards, val_labels)
        calibrated_probs = calibrator(val_rewards)
    elif method == "isotonic":
        calibrator = IsotonicCalibration()
        calibrator.fit(val_rewards, val_labels)
        calibrated_probs = calibrator.predict(val_rewards)
    else:
        raise ValueError(f"Unknown calibration method: {method!r}. Choose 'temperature', 'platt', or 'isotonic'.")

    ece_after = evaluator.expected_calibration_error(calibrated_probs, val_labels)

    return calibrator, ece_before, ece_after
