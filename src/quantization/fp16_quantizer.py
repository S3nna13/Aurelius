"""FP16 quantization utilities for memory-efficient inference."""
from __future__ import annotations

from dataclasses import dataclass

import torch


FP16_MAX = 65504.0


@dataclass
class FP16Quantizer:
    """Simple FP16 quantizer for model weights.

    Per-tensor quantization to 16-bit floating point using torch half precision.
    """

    def quantize(self, tensor: list[list[float]]) -> tuple[list[list[float]], float]:
        if not tensor or not tensor[0]:
            return [[]], 1.0
        max_abs = max(abs(v) for row in tensor for v in row)
        if max_abs == 0.0:
            scale = 1.0
        else:
            scale = max_abs / FP16_MAX
        t = torch.tensor(tensor, dtype=torch.float32)
        scaled = t / scale
        half = scaled.half()
        return half.tolist(), scale

    def dequantize(self, tensor: list[list[float]], scale: float) -> list[list[float]]:
        t = torch.tensor(tensor, dtype=torch.float32)
        return (t * scale).tolist()
