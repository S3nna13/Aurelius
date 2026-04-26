"""Adaptive gradient clipping for stable training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveGradientClipper:
    """Clip gradients based on running statistics of gradient norm.

    Adapts the clip threshold based on the observed gradient norm history,
    growing the threshold when norms are consistently high.
    """

    initial_threshold: float = 1.0
    growth_factor: float = 1.05
    decay_factor: float = 0.95
    window_size: int = 100
    _norm_history: list[float] | None = None

    def __post_init__(self) -> None:
        self._norm_history = []

    def clip(self, total_norm: float) -> float:
        if self._norm_history is None:
            self._norm_history = []
        self._norm_history.append(total_norm)
        if len(self._norm_history) > self.window_size:
            self._norm_history.pop(0)

        avg_norm = sum(self._norm_history) / len(self._norm_history)
        threshold = self.initial_threshold
        ratio = avg_norm / max(threshold, 1e-8)
        if ratio > 1.5:
            threshold *= self.growth_factor
        elif ratio < 0.5:
            threshold *= self.decay_factor
        return threshold


ADAPTIVE_CLIPPER = AdaptiveGradientClipper()
