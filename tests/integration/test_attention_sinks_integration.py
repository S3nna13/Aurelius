"""Integration tests for AttentionSinkCache surface.

Verify:
* registry advertises both "kv_int8" and "attention_sinks"
* the registered class instantiates and works end-to-end with a toy
  attention computation
* importing the longcontext package does not touch src.model
"""

from __future__ import annotations

import importlib
import sys

import pytest
import torch

import src.longcontext as lc
from src.longcontext.attention_sinks import AttentionSinkCache


def test_registry_has_both_strategies():
    assert "kv_int8" in lc.LONGCONTEXT_STRATEGY_REGISTRY
    assert "attention_sinks" in lc.LONGCONTEXT_STRATEGY_REGISTRY
    assert lc.LONGCONTEXT_STRATEGY_REGISTRY["attention_sinks"] is AttentionSinkCache


def test_instantiation_from_registry():
    cls = lc.LONGCONTEXT_STRATEGY_REGISTRY["attention_sinks"]
    cache = cls(n_sinks=2, window_size=8, head_dim=16, n_kv_heads=4)
    assert cache.num_cached_tokens() == 0
    assert cache.budget == 10


def test_toy_attention_uses_cached_kv_and_positions():
    """Run a tiny scaled-dot-product attention over the cached KV.

    This is a smoke test — we only verify the shapes and that the
    output is finite given a long streaming input.
    """
    B, H, D = 2, 4, 16
    cache = AttentionSinkCache(n_sinks=4, window_size=32, head_dim=D, n_kv_heads=H)

    torch.manual_seed(0)
    total = 512
    k_stream = torch.randn(B, H, total, D)
    v_stream = torch.randn(B, H, total, D)
    for t in range(total):
        cache.append(k_stream[:, :, t:t + 1, :], v_stream[:, :, t:t + 1, :], t)

    # Pull the final cached view by appending one more token.
    ck, cv, cp = cache.append(
        torch.randn(B, H, 1, D), torch.randn(B, H, 1, D), total
    )
    assert ck.shape == (B, H, 4 + 32, D)
    assert cp.shape == (4 + 32,)
    # Sinks get their absolute positions, window gets shifted positions.
    assert cp[:4].tolist() == [0, 1, 2, 3]
    assert cp[4:].tolist() == list(range(4, 4 + 32))

    # Toy attention: query from the "current" position attends to cache.
    q = torch.randn(B, H, 1, D)
    scores = torch.matmul(q, ck.transpose(-2, -1)) / (D ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, cv)
    assert out.shape == (B, H, 1, D)
    assert torch.isfinite(out).all()


def test_importing_longcontext_does_not_import_model():
    # Hermetic subprocess check — mutating sys.modules in-process leaves
    # stale class references in already-loaded test modules that hold
    # `from src.model.rms_norm import RMSNorm`, causing isinstance checks
    # in unrelated downstream tests (e.g. test_replace_norms_count) to fail.
    import subprocess
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.longcontext as lc; import sys; "
            "assert 'src.model' not in sys.modules, list(sys.modules); "
            "_ = lc.LONGCONTEXT_STRATEGY_REGISTRY['attention_sinks']; "
            "assert 'src.model' not in sys.modules, list(sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_kv_int8_entry_unchanged():
    # kv_int8 entry must still point to the existing compressor.
    from src.longcontext.kv_compression import KVInt8Compressor

    assert lc.LONGCONTEXT_STRATEGY_REGISTRY["kv_int8"] is KVInt8Compressor


def test_reset_allows_rebatching():
    cache = AttentionSinkCache(n_sinks=2, window_size=4, head_dim=8, n_kv_heads=2)
    k = torch.randn(3, 2, 5, 8)
    v = torch.randn(3, 2, 5, 8)
    cache.append(k, v, 0)
    # Changing batch size without reset must fail loud.
    with pytest.raises(ValueError):
        cache.append(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8), 0)
    cache.reset()
    ck, cv, _ = cache.append(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8), 0)
    assert ck.shape[0] == 1
