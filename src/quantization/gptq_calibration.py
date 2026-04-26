"""GPTQ-style calibration: Hessian approximation, optimal quantization (Frantar et al. 2210.17323)."""  # noqa: E501

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GPTQConfig:
    bits: int = 4
    group_size: int = 128
    actorder: bool = False
    damp_percent: float = 0.01


@dataclass
class CalibrationStats:
    layer_name: str
    n_samples: int
    input_mean: float
    input_std: float
    hessian_diag: list[float]


class GPTQCalibrator:
    """Calibrates layers for GPTQ quantization using Welford's online algorithm."""

    def __init__(self, config: GPTQConfig | None = None) -> None:
        self.config = config if config is not None else GPTQConfig()
        # Per-layer running stats: {layer_name: {n, mean, M2, sum_sq}}
        self._stats: dict[str, dict] = {}

    def accumulate(self, layer_name: str, inputs: list[list[float]]) -> None:
        """Update running mean/var (Welford) and Hessian diag (mean of squares) per feature dim."""
        if not inputs:
            return

        len(inputs)
        d = len(inputs[0])

        if layer_name not in self._stats:
            self._stats[layer_name] = {
                "n": 0,
                "mean": [0.0] * d,
                "M2": [0.0] * d,
                "sum_sq": [0.0] * d,
            }

        st = self._stats[layer_name]
        # Validate consistent dimensionality
        if len(st["mean"]) != d:
            raise ValueError(
                f"Feature dimension mismatch for layer {layer_name}: "
                f"expected {len(st['mean'])}, got {d}"
            )

        for sample in inputs:
            st["n"] += 1
            for j, x in enumerate(sample):
                # Welford update
                delta = x - st["mean"][j]
                st["mean"][j] += delta / st["n"]
                delta2 = x - st["mean"][j]
                st["M2"][j] += delta * delta2
                # Hessian diagonal: accumulate x^2 for later mean
                st["sum_sq"][j] += x * x

    def finalize(self, layer_name: str) -> CalibrationStats:
        """Return CalibrationStats with hessian_diag = mean(x^2) per feature dimension."""
        if layer_name not in self._stats:
            raise KeyError(f"No calibration data for layer: {layer_name!r}")

        st = self._stats[layer_name]
        n = st["n"]
        d = len(st["mean"])

        # Overall scalar mean and std (average across dims)
        input_mean = sum(st["mean"]) / d if d > 0 else 0.0

        if n > 1:
            variances = [st["M2"][j] / (n - 1) for j in range(d)]
            avg_var = sum(variances) / d if d > 0 else 0.0
            input_std = math.sqrt(avg_var) if avg_var > 0 else 0.0
        else:
            input_std = 0.0

        hessian_diag = [st["sum_sq"][j] / n for j in range(d)]

        return CalibrationStats(
            layer_name=layer_name,
            n_samples=n,
            input_mean=input_mean,
            input_std=input_std,
            hessian_diag=hessian_diag,
        )

    def quantize_weight(
        self,
        weight: list[float],
        scale: float,
        zero_point: float,
        bits: int,
    ) -> list[int]:
        """Quantize weights: round(clamp((w/scale) + zero_point, 0, 2^bits - 1))."""
        max_val = (1 << bits) - 1  # 2^bits - 1
        result: list[int] = []
        for w in weight:
            q_float = (w / scale) + zero_point
            q_clamped = max(0.0, min(float(max_val), q_float))
            result.append(int(round(q_clamped)))
        return result

    def dequantize(
        self,
        quantized: list[int],
        scale: float,
        zero_point: float,
    ) -> list[float]:
        """Dequantize: (q - zero_point) * scale."""
        return [(q - zero_point) * scale for q in quantized]

    def compute_scale(
        self,
        weight_row: list[float],
        bits: int,
    ) -> tuple[float, float]:
        """Compute symmetric quantization scale and zero_point.

        scale = max(|w|) / (2^(bits-1) - 1), zero_point = 0.
        """
        if not weight_row:
            return (1e-8, 0.0)
        max_abs = max(abs(w) for w in weight_row)
        max_q = (1 << (bits - 1)) - 1  # 2^(bits-1) - 1
        scale = max_abs / max_q if max_q > 0 else 1e-8
        if scale < 1e-8:
            scale = 1e-8
        return (scale, 0.0)
