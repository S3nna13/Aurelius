from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PrivacyBudget:
    epsilon: float
    delta: float
    mechanism: str  # "gaussian" | "laplace" | "rdp_gaussian"


class PrivacyAccountant:
    """Track cumulative privacy budget across training steps."""

    def __init__(self, delta: float = 1e-5, mechanism: str = "gaussian") -> None:
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        if mechanism not in ("gaussian", "laplace", "rdp_gaussian"):
            raise ValueError("mechanism must be 'gaussian', 'laplace', or 'rdp_gaussian'")
        self.delta = delta
        self.mechanism = mechanism
        self._total_epsilon: float = 0.0
        self._total_steps: int = 0

    def accumulate(
        self,
        noise_multiplier: float,
        sample_rate: float,
        steps: int = 1,
    ) -> PrivacyBudget:
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if not (0 < sample_rate <= 1):
            raise ValueError("sample_rate must be in (0, 1]")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        # Per-step epsilon for the Gaussian mechanism at sensitivity 1
        per_step_eps = math.sqrt(2.0 * math.log(1.25 / self.delta)) / noise_multiplier
        # Simplified Rényi DP composition: accumulate with sqrt(steps * sample_rate)
        self._total_epsilon += per_step_eps * math.sqrt(steps * sample_rate)
        self._total_steps += steps
        return self.get_budget()

    def get_budget(self) -> PrivacyBudget:
        return PrivacyBudget(
            epsilon=self._total_epsilon,
            delta=self.delta,
            mechanism=self.mechanism,
        )

    def reset(self) -> None:
        self._total_epsilon = 0.0
        self._total_steps = 0

    def steps_for_epsilon(
        self,
        target_epsilon: float,
        noise_multiplier: float,
        sample_rate: float,
    ) -> int:
        if target_epsilon <= 0:
            raise ValueError("target_epsilon must be positive")
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if not (0 < sample_rate <= 1):
            raise ValueError("sample_rate must be in (0, 1]")
        per_step_eps = math.sqrt(2.0 * math.log(1.25 / self.delta)) / noise_multiplier
        # total_eps(n) = per_step_eps * sqrt(n * sample_rate) <= target_epsilon
        # n <= (target_epsilon / per_step_eps)^2 / sample_rate
        max_steps = int((target_epsilon / per_step_eps) ** 2 / sample_rate)
        return max(0, max_steps)

    def report(self) -> dict:
        return {
            "epsilon": self._total_epsilon,
            "delta": self.delta,
            "mechanism": self.mechanism,
            "total_steps": self._total_steps,
        }
