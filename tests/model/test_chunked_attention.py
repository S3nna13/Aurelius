"""Tests for src/model/chunked_attention.py"""

from __future__ import annotations

import math

import pytest
import torch

from src.model.chunked_attention import (
    ChunkedAttnConfig,
    ChunkedAttentionBenchmark,
    ChunkedMultiHeadAttention,
    chunked_attention,
    compute_memory_ratio,
    estimate_attention_memory_mb,
    standard_attention,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B, H, T, D = 2, 2, 8, 16  # small tensors for all tests


def _make_qkv(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    return q, k, v


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = ChunkedAttnConfig()
    assert cfg.chunk_size == 64
    assert cfg.causal is True
    assert cfg.scale is None


# ---------------------------------------------------------------------------
# 2. standard_attention shape
# ---------------------------------------------------------------------------


def test_standard_attention_shape():
    q, k, v = _make_qkv()
    out = standard_attention(q, k, v, causal=True)
    assert out.shape == (B, H, T, D)


# ---------------------------------------------------------------------------
# 3. chunked_attention shape
# ---------------------------------------------------------------------------


def test_chunked_attention_shape():
    q, k, v = _make_qkv()
    out = chunked_attention(q, k, v, chunk_size=4, causal=True)
    assert out.shape == (B, H, T, D)


# ---------------------------------------------------------------------------
# 4. Chunked equals standard — non-causal
# ---------------------------------------------------------------------------


def test_chunked_equals_standard_noncausal():
    q, k, v = _make_qkv(seed=1)
    ref = standard_attention(q, k, v, causal=False)
    out = chunked_attention(q, k, v, chunk_size=4, causal=False)
    assert torch.allclose(ref, out, atol=1e-4), f"max diff: {(ref - out).abs().max().item()}"


# ---------------------------------------------------------------------------
# 5. Chunked equals standard — causal
# ---------------------------------------------------------------------------


def test_chunked_equals_standard_causal():
    q, k, v = _make_qkv(seed=2)
    ref = standard_attention(q, k, v, causal=True)
    out = chunked_attention(q, k, v, chunk_size=4, causal=True)
    assert torch.allclose(ref, out, atol=1e-4), f"max diff: {(ref - out).abs().max().item()}"


# ---------------------------------------------------------------------------
# 6. Large chunk (>= T) gives same result
# ---------------------------------------------------------------------------


def test_chunked_equals_standard_large_chunk():
    q, k, v = _make_qkv(seed=3)
    ref = standard_attention(q, k, v, causal=True)
    out = chunked_attention(q, k, v, chunk_size=T, causal=True)
    assert torch.allclose(ref, out, atol=1e-4), f"max diff: {(ref - out).abs().max().item()}"


# ---------------------------------------------------------------------------
# 7. chunk_size=1 gives same result
# ---------------------------------------------------------------------------


def test_chunked_equals_standard_small_chunk():
    q, k, v = _make_qkv(seed=4)
    ref = standard_attention(q, k, v, causal=True)
    out = chunked_attention(q, k, v, chunk_size=1, causal=True)
    assert torch.allclose(ref, out, atol=1e-4), f"max diff: {(ref - out).abs().max().item()}"


# ---------------------------------------------------------------------------
# 8. compute_memory_ratio
# ---------------------------------------------------------------------------


def test_compute_memory_ratio():
    assert compute_memory_ratio(128, 16) == pytest.approx(16 / 128)


# ---------------------------------------------------------------------------
# 9. estimate_attention_memory_mb keys
# ---------------------------------------------------------------------------


def test_estimate_memory_mb_keys():
    result = estimate_attention_memory_mb(
        batch_size=2, n_heads=4, seq_len=128, head_dim=64, chunk_size=32
    )
    assert "full_mb" in result
    assert "chunked_mb" in result
    assert "savings_fraction" in result


# ---------------------------------------------------------------------------
# 10. chunked_mb < full_mb when chunk_size < seq_len
# ---------------------------------------------------------------------------


def test_estimate_memory_mb_savings():
    result = estimate_attention_memory_mb(
        batch_size=2, n_heads=4, seq_len=128, head_dim=64, chunk_size=16
    )
    assert result["chunked_mb"] < result["full_mb"]
    assert result["savings_fraction"] > 0.0


# ---------------------------------------------------------------------------
# 11. ChunkedMultiHeadAttention forward shape
# ---------------------------------------------------------------------------


def test_chunked_mha_shape():
    cfg = ChunkedAttnConfig(chunk_size=4, causal=True)
    mha = ChunkedMultiHeadAttention(d_model=D * H, n_heads=H, cfg=cfg)
    torch.manual_seed(0)
    x = torch.randn(B, T, D * H)
    out = mha(x)
    assert out.shape == (B, T, D * H)


# ---------------------------------------------------------------------------
# 12. Different chunk sizes give same MHA output
# ---------------------------------------------------------------------------


def test_chunked_mha_different_chunk_sizes():
    d_model = D * H
    torch.manual_seed(42)
    x = torch.randn(B, T, d_model)

    cfg1 = ChunkedAttnConfig(chunk_size=1, causal=True)
    mha1 = ChunkedMultiHeadAttention(d_model=d_model, n_heads=H, cfg=cfg1)

    cfg2 = ChunkedAttnConfig(chunk_size=T, causal=True)
    mha2 = ChunkedMultiHeadAttention(d_model=d_model, n_heads=H, cfg=cfg2)

    # Copy weights so the only difference is chunk_size
    mha2.load_state_dict(mha1.state_dict())

    with torch.no_grad():
        out1 = mha1(x)
        out2 = mha2(x)

    assert torch.allclose(out1, out2, atol=1e-4), f"max diff: {(out1 - out2).abs().max().item()}"


# ---------------------------------------------------------------------------
# 13. ChunkedAttentionBenchmark.verify_equivalence returns True
# ---------------------------------------------------------------------------


def test_benchmark_verify_equivalence_true():
    cfg = ChunkedAttnConfig(chunk_size=4, causal=True)
    bench = ChunkedAttentionBenchmark(cfg)
    q, k, v = _make_qkv(seed=5)
    assert bench.verify_equivalence(q, k, v, atol=1e-4) is True


# ---------------------------------------------------------------------------
# 14. ChunkedAttentionBenchmark.estimate_speedup returns 1.0
# ---------------------------------------------------------------------------


def test_benchmark_estimate_speedup():
    cfg = ChunkedAttnConfig(chunk_size=4, causal=True)
    bench = ChunkedAttentionBenchmark(cfg)
    assert bench.estimate_speedup(T=256, chunk_size=64) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 15. Causal mask correctness — position i cannot attend to j > i
# ---------------------------------------------------------------------------


def test_chunked_causal_mask_correctness():
    """Verify chunked causal attention matches manually masked standard attention."""
    q, k, v = _make_qkv(seed=6)

    # Build explicit causal mask and apply to standard attention
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

    # Mask: position i cannot see j > i
    q_pos = torch.arange(T).unsqueeze(1)  # (T, 1)
    k_pos = torch.arange(T).unsqueeze(0)  # (1, T)
    mask = k_pos > q_pos  # True where future
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    ref = torch.softmax(scores, dim=-1) @ v  # (B, H, T, D)

    chunked = chunked_attention(q, k, v, chunk_size=3, causal=True, scale=scale)
    assert torch.allclose(ref, chunked, atol=1e-4), f"max diff: {(ref - chunked).abs().max().item()}"
