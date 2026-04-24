"""Aurelius compression surface: KV cache, activations, model pruning, and merging."""
__all__ = [
    "KVCacheCompressor", "KV_CACHE_COMPRESSOR",
    "ActivationCompressor", "ACTIVATION_COMPRESSOR",
    "ModelPruner", "MODEL_PRUNER_REGISTRY",
    "GradCompressMethod", "GradCompressionConfig", "GradientCompressor",
    "SDCConfig", "DraftBuffer", "SpeculativeCompressionMetrics", "SpeculativeDecodingCompressor",
    "PackingStrategy", "LosslessPacker",
    "DAREConfig", "MergeResult", "DAREMerger",
    "TIESConfig", "TIESMerger",
    # weight sharing
    "SharingGroup", "WeightSharingConfig", "WeightSharing", "WEIGHT_SHARING_REGISTRY",
    # pruning scheduler
    "SparsitySchedule", "PruningConfig", "PruningScheduler", "PRUNING_SCHEDULER_REGISTRY",
    # sparse optimizer
    "SparseUpdate", "SparseOptimizerConfig", "SparseOptimizer", "SPARSE_OPTIMIZER_REGISTRY",
    "COMPRESSION_REGISTRY",
]
from .kv_cache_compression import KVCacheCompressor, KV_CACHE_COMPRESSOR
from .activation_compression import ActivationCompressor, ACTIVATION_COMPRESSOR
from .model_pruner import ModelPruner, MODEL_PRUNER_REGISTRY
from .gradient_compression import GradCompressMethod, GradCompressionConfig, GradientCompressor
from .speculative_decoding_compressor import (
    SDCConfig, DraftBuffer, SpeculativeCompressionMetrics, SpeculativeDecodingCompressor,
)
from .lossless_packer import PackingStrategy, LosslessPacker
from .dare_merge import DAREConfig, MergeResult, DAREMerger
from .ties_merge import TIESConfig, TIESMerger
from .weight_sharing import (
    SharingGroup, WeightSharingConfig, WeightSharing, WEIGHT_SHARING_REGISTRY,
)
from .pruning_scheduler import (
    SparsitySchedule, PruningConfig, PruningScheduler, PRUNING_SCHEDULER_REGISTRY,
)
from .sparse_optimizer import (
    SparseUpdate, SparseOptimizerConfig, SparseOptimizer, SPARSE_OPTIMIZER_REGISTRY,
)

COMPRESSION_REGISTRY: dict[str, object] = {
    "kv_cache": KV_CACHE_COMPRESSOR,
    "activation": ACTIVATION_COMPRESSOR,
    "pruning": MODEL_PRUNER_REGISTRY,
    "gradient_compression": GradientCompressor,
    "speculative_compression": SpeculativeDecodingCompressor,
    "lossless_packer": LosslessPacker,
    "dare_merge": DAREMerger,
    "ties_merge": TIESMerger,
    "weight_sharing": WEIGHT_SHARING_REGISTRY,
    "pruning_scheduler": PRUNING_SCHEDULER_REGISTRY,
    "sparse_optimizer": SPARSE_OPTIMIZER_REGISTRY,
}
