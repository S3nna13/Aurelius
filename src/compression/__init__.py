"""Aurelius compression surface: KV cache, activations, and model pruning."""
__all__ = [
    "KVCacheCompressor", "KV_CACHE_COMPRESSOR",
    "ActivationCompressor", "ACTIVATION_COMPRESSOR",
    "ModelPruner", "PRUNING_REGISTRY",
    "GradCompressMethod", "GradCompressionConfig", "GradientCompressor",
    "SDCConfig", "DraftBuffer", "SpeculativeCompressionMetrics", "SpeculativeDecodingCompressor",
    "PackingStrategy", "LosslessPacker",
    "COMPRESSION_REGISTRY",
]
from .kv_cache_compression import KVCacheCompressor, KV_CACHE_COMPRESSOR
from .activation_compression import ActivationCompressor, ACTIVATION_COMPRESSOR
from .model_pruner import ModelPruner, PRUNING_REGISTRY
from .gradient_compression import GradCompressMethod, GradCompressionConfig, GradientCompressor
from .speculative_decoding_compressor import (
    SDCConfig, DraftBuffer, SpeculativeCompressionMetrics, SpeculativeDecodingCompressor,
)
from .lossless_packer import PackingStrategy, LosslessPacker

COMPRESSION_REGISTRY: dict[str, object] = {
    "kv_cache": KV_CACHE_COMPRESSOR,
    "activation": ACTIVATION_COMPRESSOR,
    "pruning": PRUNING_REGISTRY,
    "gradient_compression": GradientCompressor,
    "speculative_compression": SpeculativeDecodingCompressor,
    "lossless_packer": LosslessPacker,
}
