"""Aurelius compression surface: KV cache, activations, model pruning, and merging."""

__all__ = [
    "KVCacheCompressor",
    "KV_CACHE_COMPRESSOR",
    "ActivationCompressor",
    "ACTIVATION_COMPRESSOR",
    "ModelPruner",
    "MODEL_PRUNER_REGISTRY",
    "GradCompressMethod",
    "GradCompressionConfig",
    "GradientCompressor",
    "SDCConfig",
    "DraftBuffer",
    "SpeculativeCompressionMetrics",
    "SpeculativeDecodingCompressor",
    "PackingStrategy",
    "LosslessPacker",
    "DAREConfig",
    "MergeResult",
    "DAREMerger",
    "TIESConfig",
    "TIESMerger",
    # weight sharing
    "SharingGroup",
    "WeightSharingConfig",
    "WeightSharing",
    "WEIGHT_SHARING_REGISTRY",
    # pruning scheduler
    "SparsitySchedule",
    "PruningConfig",
    "PruningScheduler",
    "PRUNING_SCHEDULER_REGISTRY",
    # sparse optimizer
    "SparseUpdate",
    "SparseOptimizerConfig",
    "SparseOptimizer",
    "SPARSE_OPTIMIZER_REGISTRY",
    "COMPRESSION_REGISTRY",
]
from .activation_compression import ACTIVATION_COMPRESSOR, ActivationCompressor
from .dare_merge import DAREConfig, DAREMerger, MergeResult
from .gradient_compression import GradCompressionConfig, GradCompressMethod, GradientCompressor
from .kv_cache_compression import KV_CACHE_COMPRESSOR, KVCacheCompressor
from .lossless_packer import LosslessPacker, PackingStrategy
from .model_pruner import MODEL_PRUNER_REGISTRY, ModelPruner
from .pruning_scheduler import (
    PRUNING_SCHEDULER_REGISTRY,
    PruningConfig,
    PruningScheduler,
    SparsitySchedule,
)
from .sparse_optimizer import (
    SPARSE_OPTIMIZER_REGISTRY,
    SparseOptimizer,
    SparseOptimizerConfig,
    SparseUpdate,
)
from .speculative_decoding_compressor import (
    DraftBuffer,
    SDCConfig,
    SpeculativeCompressionMetrics,
    SpeculativeDecodingCompressor,
)
from .ties_merge import TIESConfig, TIESMerger
from .weight_sharing import (
    WEIGHT_SHARING_REGISTRY,
    SharingGroup,
    WeightSharing,
    WeightSharingConfig,
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
