"""Aurelius compression surface: KV cache, activations, and model pruning."""
__all__ = [
    "KVCacheCompressor", "KV_CACHE_COMPRESSOR",
    "ActivationCompressor", "ACTIVATION_COMPRESSOR",
    "ModelPruner", "PRUNING_REGISTRY",
    "COMPRESSION_REGISTRY",
]
from .kv_cache_compression import KVCacheCompressor, KV_CACHE_COMPRESSOR
from .activation_compression import ActivationCompressor, ACTIVATION_COMPRESSOR
from .model_pruner import ModelPruner, PRUNING_REGISTRY

COMPRESSION_REGISTRY: dict[str, object] = {
    "kv_cache": KV_CACHE_COMPRESSOR,
    "activation": ACTIVATION_COMPRESSOR,
    "pruning": PRUNING_REGISTRY,
}
