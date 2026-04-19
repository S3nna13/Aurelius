"""Integration tests for the infini_attention strategy surface."""

from __future__ import annotations

import subprocess
import sys

import torch

import src.longcontext as lc
from src.longcontext.infini_attention import InfiniAttention


def test_registry_has_infini_entry():
    assert "infini" in lc.LONGCONTEXT_STRATEGY_REGISTRY
    assert lc.LONGCONTEXT_STRATEGY_REGISTRY["infini"] is InfiniAttention


def test_registry_retains_prior_entries():
    from src.longcontext.kv_compression import KVInt8Compressor
    from src.longcontext.attention_sinks import AttentionSinkCache
    from src.longcontext.ring_attention import RingAttention
    from src.longcontext.context_compaction import ContextCompactor
    from src.longcontext.kv_cache_quantization import KIVIQuantizer

    reg = lc.LONGCONTEXT_STRATEGY_REGISTRY
    assert reg["kv_int8"] is KVInt8Compressor
    assert reg["attention_sinks"] is AttentionSinkCache
    assert reg["ring_attention"] is RingAttention
    assert reg["context_compaction"] is ContextCompactor
    assert reg["kv_kivi_int4"] is KIVIQuantizer


def test_infini_constructible_from_registry_and_forwards():
    cls = lc.LONGCONTEXT_STRATEGY_REGISTRY["infini"]
    layer = cls(d_model=16, n_heads=2, head_dim=8)
    assert isinstance(layer, InfiniAttention)
    B, H, S, D = 2, 2, 8, 8
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    out = layer(q, k, v)
    assert out.shape == (B, H, S, D)
    assert torch.isfinite(out).all()


def test_importing_longcontext_does_not_import_model():
    # Hermetic subprocess — in-process sys.modules mutation would leak.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.longcontext as lc; import sys; "
            "assert 'src.model' not in sys.modules, sorted(m for m in sys.modules if 'src' in m); "
            "_ = lc.LONGCONTEXT_STRATEGY_REGISTRY['infini']; "
            "assert 'src.model' not in sys.modules, sorted(m for m in sys.modules if 'src' in m)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
