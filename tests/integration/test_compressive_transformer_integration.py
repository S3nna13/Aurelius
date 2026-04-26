"""Integration tests for the compressive_memory strategy surface."""

from __future__ import annotations

import subprocess
import sys

import torch

import src.longcontext as lc
from src.longcontext.compressive_transformer import CompressiveMemory


def test_registry_has_compressive_memory_entry():
    assert "compressive_memory" in lc.LONGCONTEXT_STRATEGY_REGISTRY
    assert lc.LONGCONTEXT_STRATEGY_REGISTRY["compressive_memory"] is CompressiveMemory


def test_registry_retains_prior_entries():
    from src.longcontext.attention_sinks import AttentionSinkCache
    from src.longcontext.chunked_prefill import ChunkedPrefill
    from src.longcontext.context_compaction import ContextCompactor
    from src.longcontext.infini_attention import InfiniAttention
    from src.longcontext.kv_cache_quantization import KIVIQuantizer
    from src.longcontext.kv_compression import KVInt8Compressor
    from src.longcontext.paged_kv_cache import PagedKVCache
    from src.longcontext.prefix_cache import PrefixCache
    from src.longcontext.ring_attention import RingAttention

    reg = lc.LONGCONTEXT_STRATEGY_REGISTRY
    assert reg["kv_int8"] is KVInt8Compressor
    assert reg["attention_sinks"] is AttentionSinkCache
    assert reg["ring_attention"] is RingAttention
    assert reg["context_compaction"] is ContextCompactor
    assert reg["kv_kivi_int4"] is KIVIQuantizer
    assert reg["infini"] is InfiniAttention
    assert reg["chunked_prefill"] is ChunkedPrefill
    assert reg["paged_kv"] is PagedKVCache
    assert reg["prefix_cache"] is PrefixCache


def test_compressive_memory_constructible_from_registry():
    cls = lc.LONGCONTEXT_STRATEGY_REGISTRY["compressive_memory"]
    mem = cls(
        n_heads=2,
        head_dim=8,
        recent_size=8,
        compressed_size=8,
        compression_rate=2,
        compression_fn="mean_pool",
    )
    assert isinstance(mem, CompressiveMemory)
    mem.reset(batch_size=2)
    torch.manual_seed(0)
    k = torch.randn(2, 2, 12, 8)
    v = torch.randn(2, 2, 12, 8)
    state = mem.update(k, v)
    assert state.recent_k.shape == (2, 2, 8, 8)
    assert state.compressed_k.shape == (2, 2, 2, 8)
    ck, cv = mem.concatenated_kv()
    assert ck.shape == (2, 2, 10, 8)
    assert cv.shape == (2, 2, 10, 8)
    assert torch.isfinite(ck).all()


def test_importing_longcontext_does_not_import_model():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.longcontext as lc; import sys; "
            "assert 'src.model' not in sys.modules, sorted(m for m in sys.modules if 'src' in m); "
            "_ = lc.LONGCONTEXT_STRATEGY_REGISTRY['compressive_memory']; "
            "assert 'src.model' not in sys.modules, sorted(m for m in sys.modules if 'src' in m)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
