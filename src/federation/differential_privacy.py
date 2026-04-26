"""Differential privacy: Gaussian/Laplace noise, sensitivity clipping, privacy budget."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum


class DPMechanism(StrEnum):
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    RANDOMIZED_RESPONSE = "randomized_response"


@dataclass
class PrivacyBudget:
    """Tracks epsilon/delta privacy budget and consumption."""

    epsilon: float
    delta: float = 1e-5
    consumed_epsilon: float = 0.0


class DifferentialPrivacy:
    """Implements differential privacy mechanisms."""

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        seed: int = 42,
    ) -> None:
        self.epsilon = epsilon
        self.delta = delta
        self._rng = random.Random(seed)
        self._budget = PrivacyBudget(epsilon=epsilon, delta=delta)

    def clip_gradient(self, gradient: list[float], max_norm: float = 1.0) -> list[float]:
        """Scale gradient down if its L2 norm exceeds max_norm."""
        norm = math.sqrt(sum(x**2 for x in gradient))
        if norm == 0.0:
            return list(gradient)
        scale = min(1.0, max_norm / norm)
        return [x * scale for x in gradient]

    def add_gaussian_noise(
        self,
        values: list[float],
        sensitivity: float = 1.0,
        sigma: float | None = None,
    ) -> list[float]:
        """Add Gaussian noise calibrated to (epsilon, delta)-DP."""
        if sigma is None:
            # sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
            sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon

        return [v + self._rng.gauss(0, sigma) for v in values]

    def add_laplace_noise(
        self,
        values: list[float],
        sensitivity: float = 1.0,
    ) -> list[float]:
        """Add Laplace noise calibrated to epsilon-DP."""
        scale = sensitivity / self.epsilon
        result = []
        for v in values:
            u = self._rng.uniform(0.001, 1.0)
            sign = 1 if self._rng.random() < 0.5 else -1
            noise = sign * (-scale * math.log(u))
            result.append(v + noise)
        return result

    def privacy_budget(self) -> PrivacyBudget:
        """Return current privacy budget."""
        return self._budget

    def consume(self, epsilon_used: float) -> bool:
        """Deduct from budget; return True if still within epsilon, False if exceeded."""
        self._budget.consumed_epsilon += epsilon_used
        return self._budget.consumed_epsilon <= self._budget.epsilon


DP_MECHANISM = DifferentialPrivacy()
