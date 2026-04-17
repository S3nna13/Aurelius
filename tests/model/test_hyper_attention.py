"""Tests for src/model/hyper_attention.py — 15 tests."""

from __future__ import annotations

import pytest
import torch

from aurelius.model.hyper_attention import (
    HammingLSH,
    HyperAttention,
    HyperAttentionLayer,
)

# ---------------------------------------------------------------------------
# Shared test dimensions
# ---------------------------------------------------------------------------
B = 2
T = 8
D_MODEL = 32
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS  # 8
N_HASHES = 4


# ---------------------------------------------------------------------------
# 1. HammingLSH.hash output shape
# ---------------------------------------------------------------------------
def test_hamming_lsh_hash_shape():
    lsh = HammingLSH(d_head=D_HEAD, n_hashes=N_HASHES)
    x = torch.randn(B, T, D_HEAD)
    h = lsh.hash(x)
    assert h.shape == (B, T, N_HASHES), f"Expected {(B, T, N_HASHES)}, got {h.shape}"


# ---------------------------------------------------------------------------
# 2. HammingLSH.hash returns bool tensor
# ---------------------------------------------------------------------------
def test_hamming_lsh_hash_dtype():
    lsh = HammingLSH(d_head=D_HEAD, n_hashes=N_HASHES)
    x = torch.randn(B, T, D_HEAD)
    h = lsh.hash(x)
    assert h.dtype == torch.bool, f"Expected bool tensor, got {h.dtype}"


# ---------------------------------------------------------------------------
# 3. HammingLSH.hash same input → same output (deterministic)
# ---------------------------------------------------------------------------
def test_hamming_lsh_hash_deterministic():
    lsh = HammingLSH(d_head=D_HEAD, n_hashes=N_HASHES)
    x = torch.randn(B, T, D_HEAD)
    h1 = lsh.hash(x)
    h2 = lsh.hash(x)
    assert (h1 == h2).all(), "Same input should produce identical hash codes"


# ---------------------------------------------------------------------------
# 4. HammingLSH.hash different inputs → potentially different hashes
# ---------------------------------------------------------------------------
def test_hamming_lsh_hash_different_inputs():
    lsh = HammingLSH(d_head=D_HEAD, n_hashes=N_HASHES)
    x1 = torch.randn(B, T, D_HEAD)
    x2 = torch.randn(B, T, D_HEAD)
    h1 = lsh.hash(x1)
    h2 = lsh.hash(x2)
    # Very unlikely all bits match for completely random inputs
    assert not (h1 == h2).all(), "Distinct inputs should generally produce different hashes"


# ---------------------------------------------------------------------------
# 5. HyperAttention output shape
# ---------------------------------------------------------------------------
def test_hyper_attention_output_shape():
    attn = HyperAttention(d_head=D_HEAD, n_hashes=N_HASHES)
    q = torch.randn(B, T, D_HEAD)
    k = torch.randn(B, T, D_HEAD)
    v = torch.randn(B, T, D_HEAD)
    out = attn(q, k, v)
    assert out.shape == (B, T, D_HEAD), f"Expected {(B, T, D_HEAD)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 6. HyperAttention output is finite (no NaN/Inf)
# ---------------------------------------------------------------------------
def test_hyper_attention_output_finite():
    attn = HyperAttention(d_head=D_HEAD, n_hashes=N_HASHES)
    q = torch.randn(B, T, D_HEAD)
    k = torch.randn(B, T, D_HEAD)
    v = torch.randn(B, T, D_HEAD)
    out = attn(q, k, v)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"


# ---------------------------------------------------------------------------
# 7. HyperAttention works with T=1 (edge case)
# ---------------------------------------------------------------------------
def test_hyper_attention_single_token():
    attn = HyperAttention(d_head=D_HEAD, n_hashes=N_HASHES)
    q = torch.randn(B, 1, D_HEAD)
    k = torch.randn(B, 1, D_HEAD)
    v = torch.randn(B, 1, D_HEAD)
    out = attn(q, k, v)
    assert out.shape == (B, 1, D_HEAD)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 8. HyperAttention works with B=1, T=1
# ---------------------------------------------------------------------------
def test_hyper_attention_batch1_seq1():
    attn = HyperAttention(d_head=D_HEAD, n_hashes=N_HASHES)
    q = torch.randn(1, 1, D_HEAD)
    k = torch.randn(1, 1, D_HEAD)
    v = torch.randn(1, 1, D_HEAD)
    out = attn(q, k, v)
    assert out.shape == (1, 1, D_HEAD)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 9. HyperAttention gradients flow (loss.backward())
# ---------------------------------------------------------------------------
def test_hyper_attention_gradients():
    attn = HyperAttention(d_head=D_HEAD, n_hashes=N_HASHES)
    q = torch.randn(B, T, D_HEAD, requires_grad=True)
    k = torch.randn(B, T, D_HEAD, requires_grad=True)
    v = torch.randn(B, T, D_HEAD, requires_grad=True)
    out = attn(q, k, v)
    loss = out.sum()
    loss.backward()
    assert q.grad is not None, "No gradient for q"
    assert k.grad is not None, "No gradient for k"
    assert v.grad is not None, "No gradient for v"


# ---------------------------------------------------------------------------
# 10. HyperAttentionLayer output shape
# ---------------------------------------------------------------------------
def test_hyper_attention_layer_output_shape():
    layer = HyperAttentionLayer(d_model=D_MODEL, n_heads=N_HEADS, n_hashes=N_HASHES)
    x = torch.randn(B, T, D_MODEL)
    out = layer(x)
    assert out.shape == (B, T, D_MODEL), f"Expected {(B, T, D_MODEL)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 11. HyperAttentionLayer output finite
# ---------------------------------------------------------------------------
def test_hyper_attention_layer_output_finite():
    layer = HyperAttentionLayer(d_model=D_MODEL, n_heads=N_HEADS, n_hashes=N_HASHES)
    x = torch.randn(B, T, D_MODEL)
    out = layer(x)
    assert torch.isfinite(out).all(), "Layer output contains NaN or Inf"


# ---------------------------------------------------------------------------
# 12. HyperAttentionLayer gradients flow
# ---------------------------------------------------------------------------
def test_hyper_attention_layer_gradients():
    layer = HyperAttentionLayer(d_model=D_MODEL, n_heads=N_HEADS, n_hashes=N_HASHES)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient for input x"
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"


# ---------------------------------------------------------------------------
# 13. HyperAttentionLayer works with B=1, T=4
# ---------------------------------------------------------------------------
def test_hyper_attention_layer_small_batch():
    layer = HyperAttentionLayer(d_model=D_MODEL, n_heads=N_HEADS, n_hashes=N_HASHES)
    x = torch.randn(1, 4, D_MODEL)
    out = layer(x)
    assert out.shape == (1, 4, D_MODEL)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 14. Different inputs produce different outputs
# ---------------------------------------------------------------------------
def test_hyper_attention_different_inputs_different_outputs():
    attn = HyperAttention(d_head=D_HEAD, n_hashes=N_HASHES)
    q1 = torch.randn(B, T, D_HEAD)
    k1 = torch.randn(B, T, D_HEAD)
    v1 = torch.randn(B, T, D_HEAD)
    q2 = torch.randn(B, T, D_HEAD)
    k2 = torch.randn(B, T, D_HEAD)
    v2 = torch.randn(B, T, D_HEAD)
    out1 = attn(q1, k1, v1)
    out2 = attn(q2, k2, v2)
    assert not torch.allclose(out1, out2), "Different inputs should produce different outputs"


# ---------------------------------------------------------------------------
# 15. HyperAttention with sample_size=1 still works
# ---------------------------------------------------------------------------
def test_hyper_attention_sample_size_one():
    attn = HyperAttention(d_head=D_HEAD, n_hashes=N_HASHES, sample_size=1)
    q = torch.randn(B, T, D_HEAD)
    k = torch.randn(B, T, D_HEAD)
    v = torch.randn(B, T, D_HEAD)
    out = attn(q, k, v)
    assert out.shape == (B, T, D_HEAD)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 16. Hash codes vary across sequence positions
# ---------------------------------------------------------------------------
def test_hamming_lsh_hash_varies_across_positions():
    lsh = HammingLSH(d_head=D_HEAD, n_hashes=N_HASHES)
    # Use a longer sequence and distinct random vectors to ensure variety
    x = torch.randn(1, T, D_HEAD)
    h = lsh.hash(x)  # (1, T, N_HASHES)
    # Not all positions should have the same hash code
    codes = h[0]  # (T, N_HASHES)
    # Check that at least 2 distinct hash codes exist
    unique_rows = set(tuple(row.tolist()) for row in codes)
    assert len(unique_rows) > 1, "Hash codes should vary across positions for random input"
