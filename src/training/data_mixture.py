"""
data_mixture.py — Data mixture and curriculum scheduling for Aurelius LLM training.

Controls proportions of data sources (web, books, code, etc.) and provides
curriculum scheduling that ramps up difficulty over training steps.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MixtureConfig:
    """Configuration for data mixture and curriculum scheduling."""

    source_names: list[str] = field(default_factory=lambda: ["web", "books", "code"])
    weights: list[float] | None = None  # None → uniform
    temperature: float = 1.0  # T=1 identity; T→0 argmax; T→∞ uniform
    curriculum_steps: int = 0  # 0 = no curriculum
    total_steps: int = 10_000


# ---------------------------------------------------------------------------
# Weight utilities
# ---------------------------------------------------------------------------


def normalize_weights(weights: list[float]) -> list[float]:
    """Normalize *weights* so they sum to 1.0.

    Raises ValueError if any weight is negative or the total is zero.
    """
    if any(w < 0 for w in weights):
        raise ValueError("All weights must be non-negative.")
    total = sum(weights)
    if total == 0.0:
        raise ValueError("Sum of weights must be greater than zero.")
    return [w / total for w in weights]


def temperature_sample_weights(weights: list[float], temperature: float) -> list[float]:
    """Apply temperature scaling to *weights*.

    w_i' = w_i^(1/T) / sum_j w_j^(1/T)

    - T = 1   → identity (same as normalized input)
    - T → 0   → argmax (all mass on the largest weight)
    - T → inf → uniform
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    inv_t = 1.0 / temperature
    scaled = [w**inv_t for w in weights]
    return normalize_weights(scaled)


def compute_curriculum_weights(
    base_weights: list[float],
    step: int,
    curriculum_steps: int,
) -> list[float]:
    """Linear warmup from uniform to *base_weights* over *curriculum_steps*.

    - step = 0                   → uniform
    - step >= curriculum_steps   → base_weights
    - 0 < step < curriculum_steps → linear interpolation
    """
    n = len(base_weights)
    uniform = [1.0 / n] * n

    if curriculum_steps <= 0 or step >= curriculum_steps:
        return normalize_weights(base_weights)

    if step <= 0:
        return uniform

    alpha = step / curriculum_steps  # 0 < alpha < 1
    interpolated = [(1.0 - alpha) * u + alpha * b for u, b in zip(uniform, base_weights)]
    return normalize_weights(interpolated)


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------


class DataMixtureSampler:
    """Samples data sources according to mixture weights with optional curriculum."""

    def __init__(self, config: MixtureConfig) -> None:
        self.config = config
        n = len(config.source_names)

        # Resolve base weights (uniform if not provided)
        if config.weights is None:
            base_weights = [1.0 / n] * n
        else:
            if len(config.weights) != n:
                raise ValueError(f"len(weights)={len(config.weights)} != len(source_names)={n}")
            base_weights = normalize_weights(config.weights)

        # Apply temperature to get effective base weights
        self._base_weights: list[float] = temperature_sample_weights(
            base_weights, config.temperature
        )
        self._source_names: list[str] = list(config.source_names)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _weights_at_step(self, step: int) -> list[float]:
        """Internal: resolve curriculum-adjusted weights at *step*."""
        return compute_curriculum_weights(self._base_weights, step, self.config.curriculum_steps)

    def sample_source(self, step: int = 0) -> str:
        """Sample a single source name according to weights at *step*."""
        weights = self._weights_at_step(step)
        return random.choices(self._source_names, weights=weights, k=1)[0]  # noqa: S311

    def sample_batch_sources(self, batch_size: int, step: int = 0) -> list[str]:
        """Return *batch_size* source names sampled at *step*."""
        weights = self._weights_at_step(step)
        return random.choices(self._source_names, weights=weights, k=batch_size)  # noqa: S311

    def get_weights_at_step(self, step: int) -> dict[str, float]:
        """Return {source_name: weight} dict at *step*."""
        weights = self._weights_at_step(step)
        return dict(zip(self._source_names, weights))

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update base weights from a {source: weight} dict.

        All source names must be present in *new_weights*.
        """
        missing = set(self._source_names) - set(new_weights.keys())
        if missing:
            raise ValueError(f"Missing sources in new_weights: {missing}")

        raw = [new_weights[name] for name in self._source_names]
        normalized = normalize_weights(raw)
        self._base_weights = temperature_sample_weights(normalized, self.config.temperature)


# ---------------------------------------------------------------------------
# Entropy & epoch estimation
# ---------------------------------------------------------------------------


def compute_mixture_entropy(weights: list[float]) -> float:
    """Compute Shannon entropy H = -sum(w * log(w)) in nats.

    Treats 0 * log(0) as 0 (limit convention).
    """
    entropy = 0.0
    for w in weights:
        if w > 0.0:
            entropy -= w * math.log(w)
    return entropy


def estimate_steps_per_epoch(
    source_sizes: dict[str, int],
    weights: list[float],  # noqa: ARG001  (kept for API consistency)
    batch_size: int,
) -> int:
    """Estimate training steps per epoch.

    Simple formula: sum of all source token counts divided by batch size.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    total_tokens = sum(source_sizes.values())
    return max(1, total_tokens // batch_size)
