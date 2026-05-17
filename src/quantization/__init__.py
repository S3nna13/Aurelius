"""Aurelius quantization package.

Post-training quantization tools for inference optimization.
"""

from src.quantization import quantize as _quantize

QuantMode = _quantize.QuantMode
quantize = _quantize.quantize
load_quantized = _quantize.load_quantized
estimate_memory = _quantize.estimate_memory

__all__ = [
    "QuantMode",
    "quantize",
    "load_quantized",
    "estimate_memory",
]
# ---------------------------------------------------------------------------
# Stable quantization backends (public API)
# ---------------------------------------------------------------------------

from .awq_quantizer import AWQQuantizer
from .gptq_quantizer import GPTQQuantizer

__all__ = [
    "QuantMode",
    "quantize",
    "load_quantized",
    "estimate_memory",
    "AWQQuantizer",
    "GPTQQuantizer",
]
