"""SmoothQuant implementation for Aurelius.

Reference: Xiao et al. (2022) SmoothQuant: Accurate and Efficient Post-Training
Quantization for Large Language Models (arXiv:2211.10438).

Per-channel activation-to-weight difficulty migration via a learned scale s,
computed as::

    s[c] = act_max[c]^alpha / (weight_max[c]^(1-alpha) + eps)

Multiplying activations by 1/s and weights by s balances quantization difficulty
between the two operands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmoothQuantConfig:
    """Immutable configuration for :class:`SmoothQuantCalibrator`.

    Args:
        alpha: Migration strength in [0, 1].  0 = all difficulty on weights,
               1 = all difficulty on activations.
        eps:   Small constant added to denominator to prevent zero division.
    """

    alpha: float = 0.5
    eps: float = 1e-5


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class SmoothQuantCalibrator:
    """Compute and apply SmoothQuant per-channel scales.

    Usage::

        cfg = SmoothQuantConfig(alpha=0.5)
        cal = SmoothQuantCalibrator(cfg)

        scales = cal.calibrate(act_stats, weight_stats)
        smooth_weights = cal.apply_scale(weights, scales_list)
        smooth_activations = cal.inverse_scale(activations, scales_list)
    """

    def __init__(self, config: SmoothQuantConfig | None = None) -> None:
        self.config = config if config is not None else SmoothQuantConfig()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        activation_stats: Dict[str, float],
        weight_stats: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute per-channel SmoothQuant scales.

        Args:
            activation_stats: Mapping ``channel_name -> act_max[c]``.
            weight_stats:     Mapping ``channel_name -> weight_max[c]``.

        Returns:
            Dict ``channel_name -> scale s[c]`` for every channel present in
            *both* dicts.  Channels missing from either dict are silently
            skipped.

        Notes:
            ``s[c] = act_max[c]^alpha / (weight_max[c]^(1-alpha) + eps)``
        """
        alpha = self.config.alpha
        eps = self.config.eps
        scales: Dict[str, float] = {}

        for channel in activation_stats:
            if channel not in weight_stats:
                continue
            act_max = activation_stats[channel]
            w_max = weight_stats[channel]

            # Guard against negative values (take abs for safety).
            act_max = abs(act_max)
            w_max = abs(w_max)

            numerator = _safe_pow(act_max, alpha)
            denominator = _safe_pow(w_max, 1.0 - alpha) + eps
            scales[channel] = numerator / denominator

        return scales

    def apply_scale(
        self,
        weights: List[float],
        scales: List[float],
    ) -> List[float]:
        """Multiply each weight element by the corresponding scale.

        ``result[i] = weights[i] * scales[i]``

        Raises:
            ValueError: If *weights* and *scales* differ in length.
        """
        if len(weights) != len(scales):
            raise ValueError(
                f"weights (len={len(weights)}) and scales (len={len(scales)}) "
                "must have the same length."
            )
        return [w * s for w, s in zip(weights, scales)]

    def inverse_scale(
        self,
        activations: List[float],
        scales: List[float],
    ) -> List[float]:
        """Divide each activation element by the corresponding scale (+ eps).

        ``result[i] = activations[i] / (scales[i] + eps)``

        The eps term mirrors the one used in :meth:`calibrate` and prevents
        zero-division when a scale collapses to zero.

        Raises:
            ValueError: If *activations* and *scales* differ in length.
        """
        if len(activations) != len(scales):
            raise ValueError(
                f"activations (len={len(activations)}) and scales "
                f"(len={len(scales)}) must have the same length."
            )
        eps = self.config.eps
        return [a / (s + eps) for a, s in zip(activations, scales)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_pow(base: float, exp: float) -> float:
    """Return base**exp, returning 1.0 for the degenerate 0^0 case."""
    if base == 0.0:
        return 0.0 if exp != 0.0 else 1.0
    return math.pow(base, exp)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SMOOTH_QUANT_REGISTRY: Dict[str, type] = {"default": SmoothQuantCalibrator}
