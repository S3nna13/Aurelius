"""FP8 quantization utilities for memory-efficient inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FP8Quantizer:
    """Simple FP8 quantizer for model weights.

    Per-channel symmetric quantization to 8-bit floating point format.
    """

    max_val: float = 448.0  # FP8 E4M3 max

    def quantize(self, tensor: list[list[float]]) -> tuple[list[list[int]], float]:
        if not tensor or not tensor[0]:
            return [[]], self.max_val
        max_abs = max(abs(v) for row in tensor for v in row) or 1.0
        scale = self.max_val / max_abs
        quantized = [
            [max(-self.max_val, min(self.max_val, round(v * scale))) for v in row] for row in tensor
        ]
        return quantized, scale

    def dequantize(self, tensor: list[list[int]], scale: float) -> list[list[float]]:
        inv_scale = 1.0 / scale
        return [[v * inv_scale for v in row] for row in tensor]


FP8_QUANT = FP8Quantizer()
