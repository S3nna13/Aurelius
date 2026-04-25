"""INT4 quantization with symmetric range mapping."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class INT4Quantizer:
    """Quantize/dequantize 32-bit floats to 4-bit integers (symmetric)."""

    def quantize(self, tensor: list[list[float]]) -> tuple[list[list[int]], float]:
        max_abs = max(abs(v) for row in tensor for v in row) if tensor and tensor[0] else 1.0
        scale = 7.0 / max_abs  # 4-bit signed: -8..7
        quantized = []
        for row in tensor:
            q_row = [max(-8, min(7, int(round(v * scale)))) for v in row]
            quantized.append(q_row)
        return quantized, scale

    def dequantize(self, tensor: list[list[int]], scale: float) -> list[list[float]]:
        inv = 1.0 / scale
        return [[v * inv for v in row] for row in tensor]


INT4_QUANT = INT4Quantizer()