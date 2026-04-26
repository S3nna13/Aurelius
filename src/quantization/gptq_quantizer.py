"""GPTQ post-training quantization for Aurelius.

Reference: Frantar et al. (2022) GPTQ: Accurate Post-Training Quantization
for Generative Pre-trained Transformers (arXiv:2210.17323).

This module implements a lightweight round-to-nearest quantizer following the
GPTQ conventions used throughout Aurelius: symmetric or asymmetric integer
quantization with configurable bit-width and group size.  Torch is imported
lazily so that the module can be imported in environments where torch is not
available (an informative ``ImportError`` is raised when the quantizer is
actually used).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import torch


# ---------------------------------------------------------------------------
# Lazy torch import helper
# ---------------------------------------------------------------------------


def _require_torch():
    try:
        import torch  # noqa: WPS433 - intentional lazy import
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "GPTQQuantizer requires PyTorch. Install with `pip install torch`."
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# Config / container dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GPTQConfig:
    """Configuration for :class:`GPTQQuantizer`."""

    bits: int = 4
    group_size: int = 128
    desc_act: bool = False
    damp_percent: float = 0.01
    sym: bool = True


@dataclass(frozen=True)
class QuantizedLayer:
    """Immutable container for a quantized weight tensor."""

    weight_int: torch.Tensor | None
    scale: torch.Tensor | None
    zero_point: torch.Tensor | None
    bits: int
    group_size: int


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------


class GPTQQuantizer:
    """GPTQ-style weight quantizer (round-to-nearest approximation)."""

    def __init__(self, config: GPTQConfig | None = None) -> None:
        self.config = config if config is not None else GPTQConfig()
        if self.config.bits < 2 or self.config.bits > 16:
            raise ValueError(f"bits must be in [2, 16], got {self.config.bits}")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def quantize_weight(self, weight: torch.Tensor) -> QuantizedLayer:
        """Quantize a floating-point weight tensor to integer representation."""
        torch = _require_torch()
        if not isinstance(weight, torch.Tensor):
            raise TypeError("weight must be a torch.Tensor")

        bits = self.config.bits
        if self.config.sym:
            qmax = (1 << (bits - 1)) - 1  # e.g. 7 for 4-bit
            qmin = -(1 << (bits - 1))  # e.g. -8 for 4-bit
            max_abs = weight.abs().max().clamp(min=1e-8)
            scale = max_abs / qmax
            w_int = (weight / scale).round().clamp(qmin, qmax).to(torch.int32)
            zero = torch.zeros((), dtype=weight.dtype, device=weight.device)
        else:
            qmax = (1 << bits) - 1  # e.g. 15 for 4-bit
            w_min = weight.min()
            w_max = weight.max()
            scale = (w_max - w_min).clamp(min=1e-8) / qmax
            zero = (-w_min / scale).round()
            w_int = (weight / scale + zero).round().clamp(0, qmax).to(torch.int32)

        return QuantizedLayer(
            weight_int=w_int,
            scale=scale,
            zero_point=zero,
            bits=bits,
            group_size=self.config.group_size,
        )

    def dequantize(self, layer: QuantizedLayer) -> torch.Tensor:
        """Reconstruct a float tensor from a :class:`QuantizedLayer`."""
        torch = _require_torch()
        if layer.weight_int is None or layer.scale is None:
            raise ValueError("QuantizedLayer has no quantized weights to dequantize")

        w_int = layer.weight_int.to(torch.float32)
        scale = layer.scale.to(torch.float32)

        if self.config.sym:
            return w_int * scale
        zero = (
            layer.zero_point.to(torch.float32)
            if layer.zero_point is not None
            else torch.zeros_like(scale)
        )
        return (w_int - zero) * scale

    def quantize_error(
        self,
        original: torch.Tensor,
        layer: QuantizedLayer,
    ) -> float:
        """Return mean squared reconstruction error."""
        _require_torch()
        recon = self.dequantize(layer).to(original.dtype)
        return float(((recon - original) ** 2).mean().item())

    def bits_saved_ratio(self, original_bits: int = 32) -> float:
        """Ratio of quantized bits to original bits (e.g. 4/32 = 0.125)."""
        if original_bits <= 0:
            raise ValueError("original_bits must be positive")
        return self.config.bits / original_bits


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
GPTQ_QUANTIZER_REGISTRY: dict[str, type] = {"default": GPTQQuantizer}
