"""Tests for src/model/chunked_attention_v2.py

Tiny configurations (small tensors) are used throughout.
All tests use pure PyTorch — no HuggingFace, scipy, or sklearn.

Coverage (≥10 tests):
  1.  config_defaults
  2.  chunked_attention output shape (B, H, T, D_HEAD)
  3.  numerical equivalence with standard attention — causal (atol=1e-4)
  4.  numerical equivalence with standard attention — non-causal (atol=1e-4)
  5.  single chunk (chunk_size >= T)
  6.  chunk_size = 1
  7.  ChunkedAttention output shape
  8.  gradient flows through ChunkedAttention
  9.  ChunkedAttnBlock output shape
 10.  ChunkedAttnBlock residual (output != input)
 11.  compare_chunked_vs_standard < 1e-4
 12.  seq_len not divisible by chunk_size
"""

from __future__ import annotations

import math

import torch

from src.model.chunked_attention_v2 import (
    ChunkedAttention,
    ChunkedAttnBlock,
    ChunkedAttnConfig,
    chunked_attention,
    compare_chunked_vs_standard,
)

# ---------------------------------------------------------------------------
# Shared tiny configuration
# ---------------------------------------------------------------------------

D_MODEL = 16
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS  # 8
CHUNK = 4
BATCH = 2
SEQ = 8


def _make_qkv(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Q, K, V of shape (BATCH, N_HEADS, SEQ, D_HEAD)."""
    torch.manual_seed(seed)
    q = torch.randn(BATCH, N_HEADS, SEQ, D_HEAD)
    k = torch.randn(BATCH, N_HEADS, SEQ, D_HEAD)
    v = torch.randn(BATCH, N_HEADS, SEQ, D_HEAD)
    return q, k, v


def _standard_attention_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """O(T^2) reference attention for comparison inside tests."""
    B, H, T, d = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        q_pos = torch.arange(T).unsqueeze(1)
        k_pos = torch.arange(T).unsqueeze(0)
        mask = (k_pos > q_pos).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = ChunkedAttnConfig()
    assert cfg.d_model == 64
    assert cfg.n_heads == 4
    assert cfg.chunk_size == 64
    assert cfg.causal is True
    assert cfg.scale is None


# ---------------------------------------------------------------------------
# 2. chunked_attention output shape (B, H, T, D_HEAD)
# ---------------------------------------------------------------------------


def test_chunked_attention_output_shape():
    Q, K, V = _make_qkv(seed=0)
    out = chunked_attention(Q, K, V, chunk_size=CHUNK, causal=True)
    assert out.shape == (BATCH, N_HEADS, SEQ, D_HEAD), (
        f"Expected ({BATCH}, {N_HEADS}, {SEQ}, {D_HEAD}), got {tuple(out.shape)}"
    )


# ---------------------------------------------------------------------------
# 3. Numerical equivalence — causal (atol=1e-4)
# ---------------------------------------------------------------------------


def test_numerical_equivalence_causal():
    Q, K, V = _make_qkv(seed=1)
    ref = _standard_attention_ref(Q, K, V, causal=True)
    out = chunked_attention(Q, K, V, chunk_size=CHUNK, causal=True)
    max_diff = (ref - out).abs().max().item()
    assert torch.allclose(ref, out, atol=1e-4), f"max abs diff: {max_diff:.3e}"


# ---------------------------------------------------------------------------
# 4. Numerical equivalence — non-causal (atol=1e-4)
# ---------------------------------------------------------------------------


def test_numerical_equivalence_noncausal():
    Q, K, V = _make_qkv(seed=2)
    ref = _standard_attention_ref(Q, K, V, causal=False)
    out = chunked_attention(Q, K, V, chunk_size=CHUNK, causal=False)
    max_diff = (ref - out).abs().max().item()
    assert torch.allclose(ref, out, atol=1e-4), f"max abs diff: {max_diff:.3e}"


# ---------------------------------------------------------------------------
# 5. Single chunk (chunk_size >= T) — equivalent to standard attention
# ---------------------------------------------------------------------------


def test_single_chunk_equals_standard():
    Q, K, V = _make_qkv(seed=3)
    ref = _standard_attention_ref(Q, K, V, causal=True)
    # chunk_size >= SEQ → single chunk
    out = chunked_attention(Q, K, V, chunk_size=SEQ, causal=True)
    max_diff = (ref - out).abs().max().item()
    assert torch.allclose(ref, out, atol=1e-4), f"max abs diff: {max_diff:.3e}"


# ---------------------------------------------------------------------------
# 6. chunk_size = 1 — still numerically equivalent
# ---------------------------------------------------------------------------


def test_chunk_size_one():
    Q, K, V = _make_qkv(seed=4)
    ref = _standard_attention_ref(Q, K, V, causal=True)
    out = chunked_attention(Q, K, V, chunk_size=1, causal=True)
    max_diff = (ref - out).abs().max().item()
    assert torch.allclose(ref, out, atol=1e-4), f"max abs diff: {max_diff:.3e}"


# ---------------------------------------------------------------------------
# 7. ChunkedAttention output shape (B, T, d_model)
# ---------------------------------------------------------------------------


def test_chunked_attention_module_output_shape():
    cfg = ChunkedAttnConfig(d_model=D_MODEL, n_heads=N_HEADS, chunk_size=CHUNK, causal=True)
    model = ChunkedAttention(cfg)
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = model(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH}, {SEQ}, {D_MODEL}), got {tuple(out.shape)}"
    )


# ---------------------------------------------------------------------------
# 8. Gradient flows through ChunkedAttention
# ---------------------------------------------------------------------------


def test_gradient_flows():
    cfg = ChunkedAttnConfig(d_model=D_MODEL, n_heads=N_HEADS, chunk_size=CHUNK, causal=True)
    model = ChunkedAttention(cfg)
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient on input"
    assert x.grad.shape == x.shape, "Gradient shape mismatch"

    # At least one parameter must have a non-zero gradient
    param_grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "No parameter gradients"
    assert any(g.abs().max().item() > 0 for g in param_grads), "All parameter grads are zero"


# ---------------------------------------------------------------------------
# 9. ChunkedAttnBlock output shape
# ---------------------------------------------------------------------------


def test_chunked_attn_block_output_shape():
    cfg = ChunkedAttnConfig(d_model=D_MODEL, n_heads=N_HEADS, chunk_size=CHUNK, causal=True)
    block = ChunkedAttnBlock(cfg)
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = block(x)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH}, {SEQ}, {D_MODEL}), got {tuple(out.shape)}"
    )


# ---------------------------------------------------------------------------
# 10. ChunkedAttnBlock residual (output != input)
# ---------------------------------------------------------------------------


def test_chunked_attn_block_residual():
    """Block output should differ from input (residual adds attention)."""
    cfg = ChunkedAttnConfig(d_model=D_MODEL, n_heads=N_HEADS, chunk_size=CHUNK, causal=True)
    block = ChunkedAttnBlock(cfg)
    torch.manual_seed(42)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = block(x)
    # Output != input because attention is added
    assert not torch.allclose(out, x, atol=1e-6), "Block output should differ from input"
    # Output has the same shape
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 11. compare_chunked_vs_standard < 1e-4
# ---------------------------------------------------------------------------


def test_compare_chunked_vs_standard_small():
    cfg = ChunkedAttnConfig(d_model=D_MODEL, n_heads=N_HEADS, chunk_size=CHUNK, causal=True)
    attn = ChunkedAttention(cfg)
    torch.manual_seed(0)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    diff = compare_chunked_vs_standard(x, attn, chunk_size=CHUNK)
    assert diff < 1e-4, f"Max abs diff {diff:.3e} exceeds 1e-4"


# ---------------------------------------------------------------------------
# 12. seq_len not divisible by chunk_size
# ---------------------------------------------------------------------------


def test_seq_len_not_divisible_by_chunk():
    """SEQ=7, chunk_size=3 → last chunk has size 1 (7 % 3 != 0)."""
    T_odd = 7
    chunk = 3
    torch.manual_seed(5)
    Q = torch.randn(BATCH, N_HEADS, T_odd, D_HEAD)
    K = torch.randn(BATCH, N_HEADS, T_odd, D_HEAD)
    V = torch.randn(BATCH, N_HEADS, T_odd, D_HEAD)

    ref = _standard_attention_ref(Q, K, V, causal=True)
    out = chunked_attention(Q, K, V, chunk_size=chunk, causal=True)

    assert out.shape == (BATCH, N_HEADS, T_odd, D_HEAD)
    max_diff = (ref - out).abs().max().item()
    assert torch.allclose(ref, out, atol=1e-4), f"max abs diff: {max_diff:.3e}"


# ---------------------------------------------------------------------------
# Bonus: custom scale factor is respected
# ---------------------------------------------------------------------------


def test_custom_scale_factor():
    Q, K, V = _make_qkv(seed=6)
    custom_scale = 0.25
    ref = _standard_attention_ref(Q, K, V, causal=True, scale=custom_scale)
    out = chunked_attention(Q, K, V, chunk_size=CHUNK, causal=True, scale=custom_scale)
    max_diff = (ref - out).abs().max().item()
    assert torch.allclose(ref, out, atol=1e-4), f"max abs diff: {max_diff:.3e}"


# ---------------------------------------------------------------------------
# Bonus: compare_chunked_vs_standard with chunk_size=1
# ---------------------------------------------------------------------------


def test_compare_chunked_vs_standard_chunk_one():
    cfg = ChunkedAttnConfig(d_model=D_MODEL, n_heads=N_HEADS, chunk_size=1, causal=True)
    attn = ChunkedAttention(cfg)
    torch.manual_seed(7)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    diff = compare_chunked_vs_standard(x, attn, chunk_size=1)
    assert diff < 1e-4, f"Max abs diff {diff:.3e} exceeds 1e-4"
