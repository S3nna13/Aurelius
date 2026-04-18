"""Differential privacy budget accountant using Rényi DP composition."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict

import torch  # noqa: F401 — imported for codebase consistency


@dataclass
class Mechanism:
    """Describes a single DP mechanism application."""

    name: str
    sigma: float
    sensitivity: float = 1.0


class PrivacyAccountant:
    """Tracks (epsilon, delta) privacy cost across multiple mechanism applications.

    Uses Rényi DP composition to accumulate privacy loss and then converts to
    approximate (epsilon, delta)-DP.
    """

    def __init__(self, delta: float = 1e-5) -> None:
        self.delta = delta
        self._steps: List[Mechanism] = []

    # ------------------------------------------------------------------
    # Core accounting methods
    # ------------------------------------------------------------------

    def add_step(self, mechanism: Mechanism) -> None:
        """Record one application of a DP mechanism."""
        self._steps.append(mechanism)

    def _rdp_gaussian(self, alpha: float, sigma: float, sensitivity: float) -> float:
        """Rényi divergence of order alpha for the Gaussian mechanism."""
        return alpha * sensitivity ** 2 / (2.0 * sigma ** 2)

    def compute_rdp(self, alpha: float) -> float:
        """Total RDP budget at order alpha over all recorded steps."""
        return sum(
            self._rdp_gaussian(alpha, m.sigma, m.sensitivity) for m in self._steps
        )

    def rdp_to_dp(self, rdp_epsilon: float, alpha: float) -> float:
        """Convert RDP (alpha, rdp_epsilon) to (epsilon, delta)-DP epsilon.

        Requires alpha > 1.
        """
        if alpha <= 1:
            raise ValueError("alpha must be greater than 1 for RDP-to-DP conversion")
        return rdp_epsilon + math.log(1.0 / self.delta) / (alpha - 1.0)

    def get_epsilon(self, alpha: float = 8.0) -> float:
        """Compute total epsilon for current steps at the given RDP order."""
        rdp_eps = self.compute_rdp(alpha)
        return self.rdp_to_dp(rdp_eps, alpha)

    def reset(self) -> None:
        """Clear all recorded mechanism applications."""
        self._steps = []

    # ------------------------------------------------------------------
    # Properties and utilities
    # ------------------------------------------------------------------

    @property
    def n_steps(self) -> int:
        """Number of recorded mechanism applications."""
        return len(self._steps)

    def privacy_spent(self, alphas: List[float]) -> Dict[float, float]:
        """Compute get_epsilon for each alpha, returning alpha -> epsilon mapping."""
        return {alpha: self.get_epsilon(alpha) for alpha in alphas}
