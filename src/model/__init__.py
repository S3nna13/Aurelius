"""Aurelius 1.3B transformer model."""

from .attention import GroupedQueryAttention, apply_rope, precompute_rope_frequencies
from .config import AureliusConfig
from .ffn import SwiGLUFFN
from .rms_norm import RMSNorm
from .transformer import AureliusTransformer, TransformerBlock, count_parameters

__all__ = [
    "AureliusConfig",
    "AureliusTransformer",
    "GroupedQueryAttention",
    "RMSNorm",
    "SwiGLUFFN",
    "TransformerBlock",
    "apply_rope",
    "count_parameters",
    "precompute_rope_frequencies",
]
