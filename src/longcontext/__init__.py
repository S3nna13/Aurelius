"""Long-context strategies for Aurelius.

This package holds opt-in mechanisms for extending / compressing long
context windows (KV cache quantization, paged attention, etc.). Nothing
exported here modifies the default forward path; every strategy must be
explicitly selected by the caller.

Public surface:
    LONGCONTEXT_STRATEGY_REGISTRY -- dict mapping strategy name -> class
    KVInt8Compressor              -- per-head INT8 symmetric KV compressor
    CompressedKV                  -- NamedTuple holding packed buffers
    quantize_per_head_symmetric   -- helper quantizer

Importing this package MUST be side-effect-free with respect to existing
modules (model, training, inference). Keep it that way.
"""

from .kv_compression import (
    CompressedKV,
    KVInt8Compressor,
    quantize_per_head_symmetric,
)

LONGCONTEXT_STRATEGY_REGISTRY: dict = {}
LONGCONTEXT_STRATEGY_REGISTRY["kv_int8"] = KVInt8Compressor

__all__ = [
    "LONGCONTEXT_STRATEGY_REGISTRY",
    "KVInt8Compressor",
    "CompressedKV",
    "quantize_per_head_symmetric",
]
