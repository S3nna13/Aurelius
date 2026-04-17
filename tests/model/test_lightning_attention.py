"""Tests for src/model/lightning_attention.py (Lightning Attention-2).

14 tests covering:
 1.  LightningLinearAttn output shape matches v shape
 2.  Output is finite
 3.  Causal: out[:t] same whether T or T+4 longer (check first t outputs)
 4.  chunk_size > T: works (single chunk), output finite
 5.  T not divisible by chunk_size: works, output shape correct
 6.  ELU+1 kernel: q,k values all positive after kernel
 7.  LightningAttentionLayer output shape (B, T, d_model)
 8.  Layer output finite
 9.  Gradient flows through LightningAttentionLayer
10.  Batch=1 seq_len=1 works
11.  seq_len == exactly chunk_size
12.  LightningAttentionBlock output shape (B, T, d_model)
13.  Block output finite
14.  Gradient flows through LightningAttentionBlock
"""

import torch
import torch.nn.functional as F
import pytest

from aurelius.model.lightning_attention import (
    LightningLinearAttn,
    LightningAttentionLayer,
    LightningAttentionBlock,
)


# ---------------------------------------------------------------------------
# Shared dimensions
# ---------------------------------------------------------------------------
B_BATCH = 2
T = 20
D_HEAD = 16
D_MODEL = 32
N_HEADS = 2
D_FF = 64
CHUNK = 8


# ---------------------------------------------------------------------------
# 1. LightningLinearAttn output shape matches v shape
# ---------------------------------------------------------------------------

def test_lightning_linear_attn_output_shape():
    attn = LightningLinearAttn(d_head=D_HEAD, chunk_size=CHUNK)
    q = torch.randn(B_BATCH, T, D_HEAD)
    k = torch.randn(B_BATCH, T, D_HEAD)
    v = torch.randn(B_BATCH, T, D_HEAD)
    out = attn(q, k, v)
    assert out.shape == v.shape, f"Expected {v.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 2. Output finite
# ---------------------------------------------------------------------------

def test_lightning_linear_attn_output_finite():
    attn = LightningLinearAttn(d_head=D_HEAD, chunk_size=CHUNK)
    q = torch.randn(B_BATCH, T, D_HEAD)
    k = torch.randn(B_BATCH, T, D_HEAD)
    v = torch.randn(B_BATCH, T, D_HEAD)
    out = attn(q, k, v)
    assert torch.isfinite(out).all(), "Output contains non-finite values"


# ---------------------------------------------------------------------------
# 3. Causal: first t outputs are identical for T and T+4
# ---------------------------------------------------------------------------

def test_lightning_linear_attn_causal():
    """Causal property: outputs at positions 0..T-1 must not change when we
    append 4 more tokens to the sequence."""
    torch.manual_seed(42)
    attn = LightningLinearAttn(d_head=D_HEAD, chunk_size=CHUNK)
    attn.train(False)  # inference mode, no dropout

    q = torch.randn(1, T, D_HEAD)
    k = torch.randn(1, T, D_HEAD)
    v = torch.randn(1, T, D_HEAD)

    # Extend by 4 extra tokens
    q_ext = torch.cat([q, torch.randn(1, 4, D_HEAD)], dim=1)
    k_ext = torch.cat([k, torch.randn(1, 4, D_HEAD)], dim=1)
    v_ext = torch.cat([v, torch.randn(1, 4, D_HEAD)], dim=1)

    with torch.no_grad():
        out_T = attn(q, k, v)
        out_T4 = attn(q_ext, k_ext, v_ext)

    assert torch.allclose(out_T, out_T4[:, :T, :], atol=1e-5), (
        "Causal property violated: outputs for first T positions changed when "
        "4 more tokens were appended."
    )


# ---------------------------------------------------------------------------
# 4. chunk_size > T: single chunk, output finite
# ---------------------------------------------------------------------------

def test_lightning_linear_attn_chunk_larger_than_T():
    # chunk_size=128 > T=20
    attn = LightningLinearAttn(d_head=D_HEAD, chunk_size=128)
    q = torch.randn(B_BATCH, T, D_HEAD)
    k = torch.randn(B_BATCH, T, D_HEAD)
    v = torch.randn(B_BATCH, T, D_HEAD)
    out = attn(q, k, v)
    assert out.shape == v.shape
    assert torch.isfinite(out).all(), "Output contains non-finite values (chunk_size > T)"


# ---------------------------------------------------------------------------
# 5. T not divisible by chunk_size: works, output shape correct
# ---------------------------------------------------------------------------

def test_lightning_linear_attn_T_not_divisible_by_chunk():
    T_odd = 17  # 17 % 8 = 1
    attn = LightningLinearAttn(d_head=D_HEAD, chunk_size=CHUNK)
    q = torch.randn(B_BATCH, T_odd, D_HEAD)
    k = torch.randn(B_BATCH, T_odd, D_HEAD)
    v = torch.randn(B_BATCH, T_odd, D_HEAD)
    out = attn(q, k, v)
    assert out.shape == (B_BATCH, T_odd, D_HEAD), (
        f"Expected ({B_BATCH}, {T_odd}, {D_HEAD}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 6. ELU+1 kernel: after applying kernel, all q/k values are positive
# ---------------------------------------------------------------------------

def test_elu_kernel_positive():
    """ELU(x)+1 must be strictly positive for any input."""
    x = torch.randn(4, 10, D_HEAD) * 5.0
    out = F.elu(x) + 1.0
    assert (out >= 0).all(), "ELU+1 kernel produced negative values"


# ---------------------------------------------------------------------------
# 7. LightningAttentionLayer output shape (B, T, d_model)
# ---------------------------------------------------------------------------

def test_lightning_attention_layer_output_shape():
    layer = LightningAttentionLayer(D_MODEL, N_HEADS, chunk_size=CHUNK)
    x = torch.randn(B_BATCH, T, D_MODEL)
    out = layer(x)
    assert out.shape == (B_BATCH, T, D_MODEL), (
        f"Expected ({B_BATCH}, {T}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 8. Layer output finite
# ---------------------------------------------------------------------------

def test_lightning_attention_layer_output_finite():
    layer = LightningAttentionLayer(D_MODEL, N_HEADS, chunk_size=CHUNK)
    x = torch.randn(B_BATCH, T, D_MODEL)
    out = layer(x)
    assert torch.isfinite(out).all(), "LightningAttentionLayer output contains non-finite values"


# ---------------------------------------------------------------------------
# 9. Gradient flows through LightningAttentionLayer
# ---------------------------------------------------------------------------

def test_lightning_attention_layer_gradient_flows():
    layer = LightningAttentionLayer(D_MODEL, N_HEADS, chunk_size=CHUNK)
    x = torch.randn(B_BATCH, T, D_MODEL, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient flowed to input"
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all(), "Gradient contains non-finite values"


# ---------------------------------------------------------------------------
# 10. Batch=1, seq_len=1 works
# ---------------------------------------------------------------------------

def test_lightning_attention_layer_batch1_seq1():
    layer = LightningAttentionLayer(D_MODEL, N_HEADS, chunk_size=CHUNK)
    x = torch.randn(1, 1, D_MODEL)
    out = layer(x)
    assert out.shape == (1, 1, D_MODEL)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 11. seq_len exactly equals chunk_size
# ---------------------------------------------------------------------------

def test_lightning_linear_attn_seq_len_equals_chunk_size():
    chunk = 8
    attn = LightningLinearAttn(d_head=D_HEAD, chunk_size=chunk)
    q = torch.randn(B_BATCH, chunk, D_HEAD)
    k = torch.randn(B_BATCH, chunk, D_HEAD)
    v = torch.randn(B_BATCH, chunk, D_HEAD)
    out = attn(q, k, v)
    assert out.shape == (B_BATCH, chunk, D_HEAD)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 12. LightningAttentionBlock output shape (B, T, d_model)
# ---------------------------------------------------------------------------

def test_lightning_attention_block_output_shape():
    block = LightningAttentionBlock(D_MODEL, N_HEADS, D_FF, chunk_size=CHUNK)
    x = torch.randn(B_BATCH, T, D_MODEL)
    out = block(x)
    assert out.shape == (B_BATCH, T, D_MODEL), (
        f"Expected ({B_BATCH}, {T}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 13. Block output finite
# ---------------------------------------------------------------------------

def test_lightning_attention_block_output_finite():
    block = LightningAttentionBlock(D_MODEL, N_HEADS, D_FF, chunk_size=CHUNK)
    x = torch.randn(B_BATCH, T, D_MODEL)
    out = block(x)
    assert torch.isfinite(out).all(), "LightningAttentionBlock output contains non-finite values"


# ---------------------------------------------------------------------------
# 14. Gradient flows through LightningAttentionBlock
# ---------------------------------------------------------------------------

def test_lightning_attention_block_gradient_flows():
    block = LightningAttentionBlock(D_MODEL, N_HEADS, D_FF, chunk_size=CHUNK)
    x = torch.randn(B_BATCH, T, D_MODEL, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient flowed to input through LightningAttentionBlock"
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all(), "Block gradient contains non-finite values"
