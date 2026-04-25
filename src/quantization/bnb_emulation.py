"""bitsandbytes-style 8-bit quantization emulation for Aurelius.

Emulates the core INT8 quantization scheme from bitsandbytes without
depending on the bitsandbytes package itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Registry import guard — shared with awq_quantizer if both are loaded
try:
    from .awq_quantizer import QUANTIZATION_REGISTRY
except ImportError:
    QUANTIZATION_REGISTRY: dict[str, object] = {}  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BnBConfig:
    bits: int = 8
    threshold: float = 6.0
    has_fp16_weights: bool = True


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

class OutlierDetector:
    """Detects weight outliers using an absolute-value threshold."""

    def __init__(self, config: BnBConfig | None = None) -> None:
        self.config = config if config is not None else BnBConfig()

    def detect(self, tensor: torch.Tensor) -> torch.BoolTensor:
        """Return a boolean mask that is True where |value| > threshold.

        Args:
            tensor: Any floating-point tensor.

        Returns:
            Boolean tensor of the same shape (True = outlier).
        """
        return (tensor.abs() > self.config.threshold)  # type: ignore[return-value]

    def extract_outliers(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split tensor into sparse outliers and dense normal values.

        Args:
            tensor: Input float tensor.

        Returns:
            sparse_outliers: Same shape as tensor; outlier positions preserved,
                             non-outliers zeroed.
            dense_normal:    Same shape as tensor; outlier positions zeroed,
                             normal values preserved.
        """
        mask = self.detect(tensor)
        sparse_outliers = tensor * mask.to(tensor.dtype)
        dense_normal = tensor * (~mask).to(tensor.dtype)
        return sparse_outliers, dense_normal


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

_INT8_MAX = 127
_INT8_MIN = -127  # symmetric (bitsandbytes uses ±127, not ±128)


class BnBQuantizer:
    """bitsandbytes-style absmax INT8 quantization emulation.

    Each row of the weight matrix is scaled independently so that the
    largest-magnitude element maps to ±127.
    """

    def __init__(self, config: BnBConfig | None = None) -> None:
        self.config = config if config is not None else BnBConfig()
        self.outlier_detector = OutlierDetector(self.config)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def quantize(
        self,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize weight to INT8 using per-row absmax scaling.

        Args:
            weight: (out_features, in_features) float tensor.

        Returns:
            q8_weight:      INT8 quantized tensor (same shape).
            absmax_per_row: (out_features,) float32 absmax values.
        """
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)

        absmax_per_row = weight.abs().max(dim=1).values.clamp(min=1e-8)  # (out,)
        # scale: absmax maps to _INT8_MAX
        scale = absmax_per_row / _INT8_MAX  # (out,)
        # Quantize
        q_float = (weight / scale.unsqueeze(1)).round().clamp(_INT8_MIN, _INT8_MAX)
        q8_weight = q_float.to(torch.int8)
        return q8_weight, absmax_per_row

    def dequantize(
        self,
        q8_weight: torch.Tensor,
        absmax_per_row: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct float weights from INT8 + per-row absmax.

        Args:
            q8_weight:      INT8 tensor (out_features, in_features).
            absmax_per_row: (out_features,) float32 absmax values.

        Returns:
            Reconstructed float32 tensor.
        """
        scale = absmax_per_row / _INT8_MAX  # (out,)
        return q8_weight.to(torch.float32) * scale.unsqueeze(1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
QUANTIZATION_REGISTRY["bnb_int8"] = BnBQuantizer
