"""Tests for src/model/nystromformer.py — Nyströmformer attention.

Test inventory (15 tests):
 1.  test_segment_mean_shape          — _segment_mean produces (B, H, m, d_head)
 2.  test_nystrom_attn_shape          — NystromAttention output (B, T, d_model)
 3.  test_nystrom_attn_finite         — NystromAttention output contains no NaN/Inf
 4.  test_T_not_divisible_by_m        — T not divisible by num_landmarks works
 5.  test_num_landmarks_ge_T          — num_landmarks >= T degrades gracefully
 6.  test_conv_skip_shape_finite      — conv_kernel_size: shape unchanged, still finite
 7.  test_nystrom_attn_gradients      — gradient flows through NystromAttention
 8.  test_nystrom_attn_batch1         — batch=1 works
 9.  test_block_shape                 — NystromformerBlock output (B, T, d_model)
10.  test_block_finite                — NystromformerBlock output finite
11.  test_block_gradients             — gradient flows through NystromformerBlock
12.  test_model_shape                 — NystromformerModel output (B, T, d_model)
13.  test_model_finite                — NystromformerModel output finite
14.  test_model_gradients             — gradient flows through NystromformerModel
15.  test_seq_len_1                   — seq_len=1 edge case works without crash
"""

import pytest
import torch

from aurelius.model.nystromformer import (
    NystromAttention,
    NystromformerBlock,
    NystromformerModel,
)
from src.model.nystromformer import _segment_mean

# ---------------------------------------------------------------------------
# Shared test dimensions
# ---------------------------------------------------------------------------
B = 2
T = 20
D_MODEL = 64
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS   # 16
D_FF = 128
N_LAYERS = 2
M = 8                          # num_landmarks
VOCAB = 256
MAX_SEQ = 64


# ===========================================================================
# 1. Segment-mean pooling shape
# ===========================================================================

def test_segment_mean_shape():
    x = torch.randn(B, N_HEADS, T, D_HEAD)
    pooled = _segment_mean(x, M)
    assert pooled.shape == (B, N_HEADS, M, D_HEAD), (
        f"Expected {(B, N_HEADS, M, D_HEAD)}, got {pooled.shape}"
    )


# ===========================================================================
# 2. NystromAttention output shape
# ===========================================================================

def test_nystrom_attn_shape():
    attn = NystromAttention(D_MODEL, N_HEADS, num_landmarks=M)
    x = torch.randn(B, T, D_MODEL)
    out = attn(x)
    assert out.shape == (B, T, D_MODEL), (
        f"Expected {(B, T, D_MODEL)}, got {out.shape}"
    )


# ===========================================================================
# 3. NystromAttention output finite
# ===========================================================================

def test_nystrom_attn_finite():
    attn = NystromAttention(D_MODEL, N_HEADS, num_landmarks=M)
    x = torch.randn(B, T, D_MODEL)
    out = attn(x)
    assert torch.isfinite(out).all(), "NystromAttention output contains NaN/Inf"


# ===========================================================================
# 4. T not divisible by num_landmarks
# ===========================================================================

def test_T_not_divisible_by_m():
    """T=17 is not divisible by M=8 — should work and produce correct shape."""
    T_odd = 17
    m = 8
    attn = NystromAttention(D_MODEL, N_HEADS, num_landmarks=m)
    x = torch.randn(B, T_odd, D_MODEL)
    out = attn(x)
    assert out.shape == (B, T_odd, D_MODEL), (
        f"Expected {(B, T_odd, D_MODEL)}, got {out.shape}"
    )
    assert torch.isfinite(out).all(), "Output contains NaN/Inf when T not divisible by m"


# ===========================================================================
# 5. num_landmarks >= T — degrades gracefully
# ===========================================================================

def test_num_landmarks_ge_T():
    """num_landmarks=50 > T=10 — should not crash and produce finite output."""
    T_short = 10
    m_large = 50
    attn = NystromAttention(D_MODEL, N_HEADS, num_landmarks=m_large)
    x = torch.randn(B, T_short, D_MODEL)
    out = attn(x)
    assert out.shape == (B, T_short, D_MODEL), (
        f"Expected {(B, T_short, D_MODEL)}, got {out.shape}"
    )
    assert torch.isfinite(out).all(), "Output contains NaN/Inf when num_landmarks >= T"


# ===========================================================================
# 6. Conv skip connection — shape and finite
# ===========================================================================

def test_conv_skip_shape_finite():
    attn = NystromAttention(D_MODEL, N_HEADS, num_landmarks=M, conv_kernel_size=3)
    x = torch.randn(B, T, D_MODEL)
    out = attn(x)
    assert out.shape == (B, T, D_MODEL), (
        f"With conv skip: expected {(B, T, D_MODEL)}, got {out.shape}"
    )
    assert torch.isfinite(out).all(), "Conv skip output contains NaN/Inf"


# ===========================================================================
# 7. Gradient flows through NystromAttention
# ===========================================================================

def test_nystrom_attn_gradients():
    attn = NystromAttention(D_MODEL, N_HEADS, num_landmarks=M)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out = attn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient at input"
    assert torch.isfinite(x.grad).all(), "Input gradient contains NaN/Inf"
    param_grads = [p.grad for p in attn.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "No parameter gradients computed"


# ===========================================================================
# 8. Batch=1
# ===========================================================================

def test_nystrom_attn_batch1():
    attn = NystromAttention(D_MODEL, N_HEADS, num_landmarks=M)
    x = torch.randn(1, T, D_MODEL)
    out = attn(x)
    assert out.shape == (1, T, D_MODEL)
    assert torch.isfinite(out).all()


# ===========================================================================
# 9. NystromformerBlock output shape
# ===========================================================================

def test_block_shape():
    block = NystromformerBlock(D_MODEL, N_HEADS, D_FF, num_landmarks=M)
    x = torch.randn(B, T, D_MODEL)
    out = block(x)
    assert out.shape == (B, T, D_MODEL), (
        f"Expected {(B, T, D_MODEL)}, got {out.shape}"
    )


# ===========================================================================
# 10. NystromformerBlock output finite
# ===========================================================================

def test_block_finite():
    block = NystromformerBlock(D_MODEL, N_HEADS, D_FF, num_landmarks=M)
    x = torch.randn(B, T, D_MODEL)
    out = block(x)
    assert torch.isfinite(out).all(), "NystromformerBlock output contains NaN/Inf"


# ===========================================================================
# 11. Gradient flows through NystromformerBlock
# ===========================================================================

def test_block_gradients():
    block = NystromformerBlock(D_MODEL, N_HEADS, D_FF, num_landmarks=M)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None, "No gradient at input"
    assert torch.isfinite(x.grad).all(), "Block input gradient contains NaN/Inf"
    param_grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "No block parameter gradients computed"


# ===========================================================================
# 12. NystromformerModel output shape
# ===========================================================================

def test_model_shape():
    model = NystromformerModel(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        num_landmarks=M,
        max_seq_len=MAX_SEQ,
    )
    ids = torch.randint(0, VOCAB, (B, T))
    out = model(ids)
    assert out.shape == (B, T, D_MODEL), (
        f"Expected {(B, T, D_MODEL)}, got {out.shape}"
    )


# ===========================================================================
# 13. NystromformerModel output finite
# ===========================================================================

def test_model_finite():
    model = NystromformerModel(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        num_landmarks=M,
        max_seq_len=MAX_SEQ,
    )
    ids = torch.randint(0, VOCAB, (B, T))
    out = model(ids)
    assert torch.isfinite(out).all(), "NystromformerModel output contains NaN/Inf"


# ===========================================================================
# 14. Gradient flows through NystromformerModel
# ===========================================================================

def test_model_gradients():
    model = NystromformerModel(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        num_landmarks=M,
        max_seq_len=MAX_SEQ,
    )
    ids = torch.randint(0, VOCAB, (B, T))
    out = model(ids)
    out.sum().backward()
    param_grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "No model parameter gradients computed"
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "A model parameter gradient is NaN/Inf"


# ===========================================================================
# 15. seq_len=1 edge case
# ===========================================================================

def test_seq_len_1():
    """T=1 must not crash and output must be finite and correctly shaped."""
    attn = NystromAttention(D_MODEL, N_HEADS, num_landmarks=M)
    x = torch.randn(B, 1, D_MODEL)
    out = attn(x)
    assert out.shape == (B, 1, D_MODEL), (
        f"seq_len=1: expected {(B, 1, D_MODEL)}, got {out.shape}"
    )
    assert torch.isfinite(out).all(), "seq_len=1 output contains NaN/Inf"

    # Also test via the model
    model = NystromformerModel(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=1,
        num_landmarks=M,
        max_seq_len=MAX_SEQ,
    )
    ids = torch.randint(0, VOCAB, (B, 1))
    out_m = model(ids)
    assert out_m.shape == (B, 1, D_MODEL), (
        f"Model seq_len=1: expected {(B, 1, D_MODEL)}, got {out_m.shape}"
    )
    assert torch.isfinite(out_m).all(), "Model seq_len=1 output contains NaN/Inf"
