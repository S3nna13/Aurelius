"""Aurelius 1.3B transformer model."""

from .attention import GroupedQueryAttention, apply_rope, precompute_rope_frequencies
from .chunked_local_attention import ChunkedLocalAttention
from .config import AureliusConfig
from .ffn import SwiGLUFFN
from .parallel_attention import ParallelAttentionBlock
from .rms_norm import RMSNorm
from .transformer import AureliusTransformer, TransformerBlock, count_parameters

__all__ = [
    "AureliusConfig",
    "AureliusTransformer",
    "ChunkedLocalAttention",
    "GroupedQueryAttention",
    "ParallelAttentionBlock",
    "RMSNorm",
    "SwiGLUFFN",
    "TransformerBlock",
    "apply_rope",
    "count_parameters",
    "precompute_rope_frequencies",
]
