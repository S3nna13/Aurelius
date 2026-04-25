"""Differential privacy aggregation (DP-SGD style).

Aggregates per-client gradients with L2 clipping and Gaussian noise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class DPConfig:
    """Configuration for DP aggregation."""

    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    target_epsilon: float = 1.0


@dataclass(frozen=True)
class DPAggregationResult:
    """Result of a DP aggregation step."""

    aggregated: list[float]
    noise_added: list[float]
    epsilon_used: float
    num_clients: int


class DPAggregator:
    """Clips per-client gradients, averages them, and adds Gaussian noise."""

    def __init__(self, config: DPConfig | None = None) -> None:
        self.config = config or DPConfig()

    def clip_gradient(self, grad: list[float], max_norm: float) -> list[float]:
        norm = math.sqrt(sum(g * g for g in grad))
        if norm > max_norm and norm > 0.0:
            scale = max_norm / norm
        else:
            scale = 1.0
        return [g * scale for g in grad]

    def add_gaussian_noise(
        self, grad: list[float], std: float, seed: int | None = None
    ) -> list[float]:
        if seed is not None:
            random.seed(seed)
        return [g + random.gauss(0.0, std) for g in grad]

    def aggregate(self, client_grads: list[list[float]]) -> DPAggregationResult:
        if not client_grads:
            return DPAggregationResult(
                aggregated=[],
                noise_added=[],
                epsilon_used=0.0,
                num_clients=0,
            )

        n = len(client_grads)
        max_norm = self.config.max_grad_norm
        clipped = [self.clip_gradient(g, max_norm) for g in client_grads]

        dim = len(clipped[0])
        averaged = [
            sum(c[i] for c in clipped) / n for i in range(dim)
        ]

        std = self.config.noise_multiplier * max_norm / n
        noise = [random.gauss(0.0, std) for _ in range(dim)]
        noisy = [a + z for a, z in zip(averaged, noise)]

        noise_mult = self.config.noise_multiplier
        if noise_mult > 0.0:
            epsilon_used = (
                max_norm
                * math.sqrt(2.0 * math.log(1.25 / self.config.delta))
                / (noise_mult * n)
            )
        else:
            epsilon_used = float("inf")

        return DPAggregationResult(
            aggregated=noisy,
            noise_added=noise,
            epsilon_used=epsilon_used,
            num_clients=n,
        )

    def privacy_budget_remaining(
        self, steps_taken: int, total_steps: int
    ) -> float:
        if total_steps <= 0:
            return self.config.target_epsilon
        frac = max(0.0, 1.0 - steps_taken / total_steps)
        return frac * self.config.target_epsilon


DP_AGGREGATOR_REGISTRY = {"default": DPAggregator}
