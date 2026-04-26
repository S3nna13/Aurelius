"""Integration tests for the ring_attention strategy surface.

Verifies:
* LONGCONTEXT_STRATEGY_REGISTRY advertises "ring_attention" alongside the
  prior entries (kv_int8, attention_sinks) — additive, non-destructive.
* the registered class instantiates and runs a forward pass end-to-end.
* importing the longcontext package does not pull in src.model (hermetic
  subprocess check — DO NOT mutate sys.modules in-process; that leaked
  stale class refs in cycle-101).
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

import src.longcontext as lc
from src.longcontext.ring_attention import RingAttention


def test_registry_has_ring_attention_entry():
    assert "ring_attention" in lc.LONGCONTEXT_STRATEGY_REGISTRY
    assert lc.LONGCONTEXT_STRATEGY_REGISTRY["ring_attention"] is RingAttention


def test_registry_retains_prior_entries():
    # Additive-only registration: prior strategies must still be reachable.
    from src.longcontext.attention_sinks import AttentionSinkCache
    from src.longcontext.kv_compression import KVInt8Compressor

    assert lc.LONGCONTEXT_STRATEGY_REGISTRY["kv_int8"] is KVInt8Compressor
    assert lc.LONGCONTEXT_STRATEGY_REGISTRY["attention_sinks"] is AttentionSinkCache


def test_ring_attention_constructible_from_registry():
    cls = lc.LONGCONTEXT_STRATEGY_REGISTRY["ring_attention"]
    inst = cls(chunk_size=8, causal=False)
    assert isinstance(inst, RingAttention)
    assert inst.chunk_size == 8
    assert inst.causal is False


def test_ring_attention_end_to_end_forward_pass():
    cls = lc.LONGCONTEXT_STRATEGY_REGISTRY["ring_attention"]
    inst = cls(chunk_size=8, causal=True)
    B, H, S, D = 2, 4, 32, 8
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    out = inst(q, k, v)
    assert out.shape == (B, H, S, D)
    assert torch.isfinite(out).all()


def test_importing_longcontext_does_not_import_model():
    # Hermetic subprocess — mutating sys.modules in-process pollutes
    # downstream tests that cached `from src.model.rms_norm import RMSNorm`.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.longcontext as lc; import sys; "
            "assert 'src.model' not in sys.modules, sorted(m for m in sys.modules if 'src' in m); "
            "_ = lc.LONGCONTEXT_STRATEGY_REGISTRY['ring_attention']; "
            "assert 'src.model' not in sys.modules, sorted(m for m in sys.modules if 'src' in m)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_wrong_chunk_size_fails_loud():
    with pytest.raises(ValueError):
        RingAttention(chunk_size=0)
    with pytest.raises(ValueError):
        RingAttention(chunk_size=-1)
