"""Gradient noise injection for improved generalization."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class GradientNoiseInjector:
    """Inject annealed Gaussian noise into gradients during training."""

    initial_noise: float = 0.01
    decay_rate: float = 0.95
    _step: int = 0

    def inject(self, params: list[list[float]], lr: float = 1.0) -> list[list[float]]:
        self._step += 1
        noise_scale = self.initial_noise * (self.decay_rate**self._step) * lr
        noisy = []
        for row in params:
            noisy.append([v + random.gauss(0, noise_scale) for v in row])
        return noisy

    def reset(self) -> None:
        self._step = 0


GRADIENT_NOISE = GradientNoiseInjector()
