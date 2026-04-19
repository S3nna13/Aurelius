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
from .attention_sinks import AttentionSinkCache
from .ring_attention import RingAttention, ring_attention
from .context_compaction import ContextCompactor, Turn as CompactionTurn
from .kv_cache_quantization import (
    CompressedKIVI,
    KIVIQuantizer,
    pack_int4,
    unpack_int4,
)
from .infini_attention import InfiniAttention
from .yarn_position_extension import (
    YarnConfig,
    apply_rotary as yarn_apply_rotary,
    build_yarn_rotary_cache,
    yarn_inv_freq,
    yarn_linear_ramp_mask,
    yarn_mscale,
)
from .chunked_prefill import ChunkedPrefill, ChunkedPrefillConfig
from .paged_kv_cache import (
    PagedKVCache,
    PagedKVOutOfMemory,
    PageTable,
)
from .prefix_cache import PrefixCache, PrefixEntry
from .compressive_transformer import CompressiveMemory, CompressiveMemoryState

LONGCONTEXT_STRATEGY_REGISTRY: dict = {}
LONGCONTEXT_STRATEGY_REGISTRY["kv_int8"] = KVInt8Compressor
LONGCONTEXT_STRATEGY_REGISTRY["attention_sinks"] = AttentionSinkCache
LONGCONTEXT_STRATEGY_REGISTRY["ring_attention"] = RingAttention
LONGCONTEXT_STRATEGY_REGISTRY["context_compaction"] = ContextCompactor
LONGCONTEXT_STRATEGY_REGISTRY["kv_kivi_int4"] = KIVIQuantizer
LONGCONTEXT_STRATEGY_REGISTRY["infini"] = InfiniAttention
LONGCONTEXT_STRATEGY_REGISTRY["chunked_prefill"] = ChunkedPrefill
LONGCONTEXT_STRATEGY_REGISTRY["paged_kv"] = PagedKVCache
LONGCONTEXT_STRATEGY_REGISTRY["prefix_cache"] = PrefixCache
LONGCONTEXT_STRATEGY_REGISTRY["compressive_memory"] = CompressiveMemory

__all__ = [
    "LONGCONTEXT_STRATEGY_REGISTRY",
    "KVInt8Compressor",
    "CompressedKV",
    "quantize_per_head_symmetric",
    "AttentionSinkCache",
    "RingAttention",
    "ring_attention",
    "ContextCompactor",
    "CompactionTurn",
    "KIVIQuantizer",
    "CompressedKIVI",
    "pack_int4",
    "unpack_int4",
    "InfiniAttention",
    "YarnConfig",
    "yarn_apply_rotary",
    "build_yarn_rotary_cache",
    "yarn_inv_freq",
    "yarn_linear_ramp_mask",
    "yarn_mscale",
    "ChunkedPrefill",
    "ChunkedPrefillConfig",
    "PagedKVCache",
    "PagedKVOutOfMemory",
    "PageTable",
    "PrefixCache",
    "PrefixEntry",
    "CompressiveMemory",
    "CompressiveMemoryState",
]
