"""
src/training/loss_variance_monitor.py

Streaming loss-variance monitor for training diagnostics.

Maintains a rolling window of loss values, computes running
mean/variance/CV using Welford's algorithm, detects divergence
(loss spikes), and flags anomalies (variance exceeds threshold,
loss increases trend).
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple


@dataclass
class LossStats:
    """Summary statistics of the loss-value rolling window.

    Attributes:
        mean: Arithmetic mean of losses in the window.
        variance: Sample variance (Welford, n-1 denominator).
        std: Standard deviation (sqrt of variance).
        coefficient_of_variation: std / mean (0.0 if mean == 0).
        min: Minimum loss in the window.
        max: Maximum loss in the window.
        n: Number of loss values currently in the window.
    """

    mean: float
    variance: float
    std: float
    coefficient_of_variation: float
    min: float
    max: float
    n: int


class LossVarianceMonitor:
    """Streaming loss-variance monitor.

    Maintains a rolling window of (step, loss) pairs. Uses Welford's
    online algorithm to recompute mean/variance from the window on
    every stats query in a numerically stable manner.

    Args:
        window_size: Maximum number of recent losses retained.
        spike_threshold: Number of standard deviations above the
            running mean to qualify as a spike.
        divergence_threshold: Ratio by which the recent rolling
            mean must exceed the older rolling mean to be considered
            diverging.
    """

    def __init__(
        self,
        window_size: int = 100,
        spike_threshold: float = 3.0,
        divergence_threshold: float = 10.0,
    ) -> None:
        self.window_size = int(window_size)
        self.spike_threshold = float(spike_threshold)
        self.divergence_threshold = float(divergence_threshold)

        self._window: Deque[Tuple[int, float]] = deque(maxlen=self.window_size)
        self._anomalies: List[Dict] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(self, loss: float, step: int) -> None:
        """Record a new loss value at a given step.

        NaN and +/-Inf values are skipped with a warning and logged
        as anomalies, but do not enter the rolling window.
        """
        loss_f = float(loss)

        if math.isnan(loss_f):
            warnings.warn(
                f"LossVarianceMonitor: NaN loss at step {step}; ignored.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._anomalies.append({"step": int(step), "type": "nan", "value": loss_f})
            return

        if math.isinf(loss_f):
            warnings.warn(
                f"LossVarianceMonitor: Inf loss at step {step}; ignored.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._anomalies.append({"step": int(step), "type": "inf", "value": loss_f})
            return

        # Check for spike using *current* stats before appending.
        if self.is_spike(loss_f):
            self._anomalies.append(
                {"step": int(step), "type": "spike", "value": loss_f}
            )

        self._window.append((int(step), loss_f))

        # Post-append divergence check (if enough data).
        if self.is_diverging():
            # Avoid flooding: only log once per new record if last anomaly
            # at this step isn't already divergence.
            if not (
                self._anomalies
                and self._anomalies[-1].get("step") == int(step)
                and self._anomalies[-1].get("type") == "divergence"
            ):
                self._anomalies.append(
                    {"step": int(step), "type": "divergence", "value": loss_f}
                )

    # ------------------------------------------------------------------
    # Statistics (Welford's algorithm applied over the window)
    # ------------------------------------------------------------------
    def _welford(self) -> Tuple[float, float, int]:
        """Run Welford's online algorithm over the current window.

        Returns:
            (mean, M2, n) where variance = M2 / (n-1) for n >= 2.
        """
        n = 0
        mean = 0.0
        m2 = 0.0
        for _, x in self._window:
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            m2 += delta * delta2
        return mean, m2, n

    def stats(self) -> LossStats:
        """Return current window statistics."""
        if not self._window:
            return LossStats(
                mean=0.0,
                variance=0.0,
                std=0.0,
                coefficient_of_variation=0.0,
                min=0.0,
                max=0.0,
                n=0,
            )

        mean, m2, n = self._welford()
        variance = m2 / (n - 1) if n >= 2 else 0.0
        std = math.sqrt(variance)
        cv = (std / mean) if mean != 0.0 else 0.0

        values = [x for _, x in self._window]
        return LossStats(
            mean=float(mean),
            variance=float(variance),
            std=float(std),
            coefficient_of_variation=float(cv),
            min=float(min(values)),
            max=float(max(values)),
            n=int(n),
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def is_spike(self, loss: float) -> bool:
        """Return True if loss exceeds mean + spike_threshold * std.

        Returns False while the window has fewer than 2 samples
        (no stable std available).
        """
        loss_f = float(loss)
        if math.isnan(loss_f) or math.isinf(loss_f):
            return False
        if len(self._window) < 2:
            return False
        s = self.stats()
        if s.std == 0.0:
            return False
        return loss_f > s.mean + self.spike_threshold * s.std

    def is_diverging(self, recent_window: int = 20) -> bool:
        """Return True if the recent rolling mean grows much faster than the older one.

        Compares mean of last ``recent_window`` samples against the mean of the
        ``recent_window`` samples immediately preceding them. Divergence is
        declared when:

            recent_mean > older_mean + divergence_threshold * |older_mean|
            (with a small epsilon fallback when older_mean ~ 0)
        """
        rw = int(recent_window)
        if rw <= 0:
            return False
        if len(self._window) < 2 * rw:
            return False

        values = [x for _, x in self._window]
        older = values[-2 * rw : -rw]
        recent = values[-rw:]
        older_mean = sum(older) / len(older)
        recent_mean = sum(recent) / len(recent)

        baseline = abs(older_mean) if older_mean != 0.0 else 1e-12
        return recent_mean > older_mean + self.divergence_threshold * baseline

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the window and anomaly history."""
        self._window.clear()
        self._anomalies.clear()

    def anomalies(self) -> List[Dict]:
        """Return a copy of the list of detected anomaly records."""
        return list(self._anomalies)
