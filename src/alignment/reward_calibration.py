"""Reward model calibration: temperature scaling, Platt scaling, reliability diagrams."""
from __future__ import annotations

import math
from enum import Enum
from typing import Optional


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class CalibrationMethod(str, Enum):
    TEMPERATURE = "temperature"
    PLATT = "platt"
    ISOTONIC = "isotonic"


class RewardCalibrator:
    """Calibrates raw reward scores to probabilities in [0, 1].

    Supports temperature scaling, Platt scaling, and isotonic regression.
    """

    def __init__(self, method: CalibrationMethod = CalibrationMethod.TEMPERATURE) -> None:
        self.method = method
        # Temperature scaling parameters
        self._temperature: float = 1.0
        # Platt scaling parameters
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0
        # Isotonic stored pairs
        self._isotonic_pairs: Optional[list[tuple[float, int]]] = None
        self._fitted = False

    def fit(self, scores: list[float], labels: list[int]) -> None:
        """Fit calibration parameters.

        Args:
            scores: raw reward scores
            labels: binary preference labels (0 or 1)
        """
        if self.method == CalibrationMethod.TEMPERATURE:
            best_t = 1.0
            best_nll = float("inf")
            n = len(scores)
            for i in range(50):
                t = 0.1 + i * (5.0 - 0.1) / 49.0
                nll = 0.0
                for s, y in zip(scores, labels):
                    p = sigmoid(s / t)
                    # Clamp to avoid log(0)
                    p = max(1e-9, min(1 - 1e-9, p))
                    nll -= y * math.log(p) + (1 - y) * math.log(1 - p)
                nll /= n
                if nll < best_nll:
                    best_nll = nll
                    best_t = t
            self._temperature = best_t

        elif self.method == CalibrationMethod.PLATT:
            a = 1.0
            b = 0.0
            lr = 0.01
            n = len(scores)
            for _ in range(100):
                da = 0.0
                db = 0.0
                for s, y in zip(scores, labels):
                    p = sigmoid(a * s + b)
                    err = p - y
                    da += err * s
                    db += err
                da /= n
                db /= n
                a -= lr * da
                b -= lr * db
            self._platt_a = a
            self._platt_b = b

        elif self.method == CalibrationMethod.ISOTONIC:
            # Store sorted (score, label) pairs
            pairs = sorted(zip(scores, labels), key=lambda x: x[0])
            self._isotonic_pairs = pairs

        self._fitted = True

    def calibrate(self, score: float) -> float:
        """Apply calibration to a single raw score.

        Returns:
            Probability in [0, 1].
        """
        if self.method == CalibrationMethod.TEMPERATURE:
            return sigmoid(score / self._temperature)

        elif self.method == CalibrationMethod.PLATT:
            return sigmoid(self._platt_a * score + self._platt_b)

        elif self.method == CalibrationMethod.ISOTONIC:
            if self._isotonic_pairs is None:
                return 0.5
            # Nearest-neighbor lookup
            stored_scores = [p[0] for p in self._isotonic_pairs]
            stored_labels = [p[1] for p in self._isotonic_pairs]

            best_idx = 0
            best_dist = abs(score - stored_scores[0])
            for i, s in enumerate(stored_scores):
                d = abs(score - s)
                if d < best_dist:
                    best_dist = d
                    best_idx = i

            # Average label for all points at the nearest distance
            total = 0.0
            count = 0
            for i, s in enumerate(stored_scores):
                if abs(abs(score - s) - best_dist) < 1e-9:
                    total += stored_labels[i]
                    count += 1
            return total / count if count > 0 else 0.5

        return 0.5

    def expected_calibration_error(
        self,
        scores: list[float],
        labels: list[int],
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        ECE = sum_b (|B_b| / N) * |acc_b - conf_b|

        Args:
            scores: raw scores to calibrate
            labels: ground-truth binary labels
            n_bins: number of equal-width confidence bins

        Returns:
            ECE in [0, 1]
        """
        confidences = [self.calibrate(s) for s in scores]
        n = len(scores)
        bin_width = 1.0 / n_bins
        ece = 0.0

        for b in range(n_bins):
            lo = b * bin_width
            hi = (b + 1) * bin_width
            # Last bin is inclusive on right
            bin_confs = []
            bin_labels = []
            for c, y in zip(confidences, labels):
                if b == n_bins - 1:
                    in_bin = lo <= c <= hi
                else:
                    in_bin = lo <= c < hi
                if in_bin:
                    bin_confs.append(c)
                    bin_labels.append(y)

            if len(bin_confs) == 0:
                continue

            acc = sum(bin_labels) / len(bin_labels)
            conf = sum(bin_confs) / len(bin_confs)
            ece += (len(bin_confs) / n) * abs(acc - conf)

        return ece


# Registry
REWARD_CALIBRATOR_REGISTRY: dict[str, RewardCalibrator] = {
    "temperature": RewardCalibrator(CalibrationMethod.TEMPERATURE),
    "platt": RewardCalibrator(CalibrationMethod.PLATT),
}
