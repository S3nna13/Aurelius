"""Tests for sparse_attention.py — local window + strided global patterns."""
from __future__ import annotations

import pytest
import torch

from src.model.sparse_attention import (
    SparseAttentionConfig,
    SparseAttention,
    SparseTransformer,
    SparseTransformerLayer,
    build_local_mask,
    build_sparse_mask,
    build_strided_mask,
    sparsity_ratio,
)

# ---------------------------------------------------------------------------
# Small test config helpers
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> SparseAttentionConfig:
    defaults = dict(d_model=64, n_heads=4, window_size=4, stride=4, dropout=0.0)
    defaults.update(kwargs)
    return SparseAttentionConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. SparseAttentionConfig defaults
# ---------------------------------------------------------------------------

def test_sparse_attention_config_defaults():
    cfg = SparseAttentionConfig()
    assert cfg.d_model == 256
    assert cfg.n_heads == 4
    assert cfg.window_size == 32
    assert cfg.stride == 8
    assert cfg.dropout == 0.0


# ---------------------------------------------------------------------------
# 2. build_local_mask shape is (T, T)
# ---------------------------------------------------------------------------

def test_build_local_mask_shape():
    T = 16
    mask = build_local_mask(T, window_size=3)
    assert mask.shape == (T, T)


# ---------------------------------------------------------------------------
# 3. build_local_mask is causal (upper triangle is False)
# ---------------------------------------------------------------------------

def test_build_local_mask_causal():
    T = 10
    mask = build_local_mask(T, window_size=3)
    # Upper triangle (j > i) must all be False
    for i in range(T):
        for j in range(i + 1, T):
            assert not mask[i, j].item(), f"Expected mask[{i},{j}] to be False (future token)"


# ---------------------------------------------------------------------------
# 4. build_local_mask diagonal is True (attend to self)
# ---------------------------------------------------------------------------

def test_build_local_mask_diagonal_true():
    T = 8
    mask = build_local_mask(T, window_size=2)
    for i in range(T):
        assert mask[i, i].item(), f"Expected mask[{i},{i}] (self-attention) to be True"


# ---------------------------------------------------------------------------
# 5. build_local_mask token outside window is False
# ---------------------------------------------------------------------------

def test_build_local_mask_outside_window_false():
    T = 20
    window_size = 3
    mask = build_local_mask(T, window_size=window_size)
    # Token 10 should NOT attend to token 0 (distance = 10 > window_size=3)
    assert not mask[10, 0].item()
    # Token 10 should NOT attend to token 6 (distance = 4 > 3)
    assert not mask[10, 6].item()
    # Token 10 SHOULD attend to token 7 (distance = 3 == window_size)
    assert mask[10, 7].item()


# ---------------------------------------------------------------------------
# 6. build_strided_mask shape is (T, T)
# ---------------------------------------------------------------------------

def test_build_strided_mask_shape():
    T = 24
    mask = build_strided_mask(T, stride=4)
    assert mask.shape == (T, T)


# ---------------------------------------------------------------------------
# 7. build_strided_mask position 0 (stride divides 0) is True for all rows
# ---------------------------------------------------------------------------

def test_build_strided_mask_col0_true():
    T = 16
    mask = build_strided_mask(T, stride=4)
    # Column 0: 0 % stride == 0, and 0 <= i for all i, so all rows should be True
    for i in range(T):
        assert mask[i, 0].item(), f"Expected mask[{i},0] to be True (col 0 is strided global)"


# ---------------------------------------------------------------------------
# 8. build_sparse_mask is superset of both local and strided masks
# ---------------------------------------------------------------------------

def test_build_sparse_mask_superset():
    T = 16
    window_size, stride = 3, 4
    local = build_local_mask(T, window_size)
    strided = build_strided_mask(T, stride)
    sparse = build_sparse_mask(T, window_size, stride)

    # Wherever local is True, sparse must be True
    assert (sparse[local]).all(), "Sparse mask must include all local mask True positions"
    # Wherever strided is True, sparse must be True
    assert (sparse[strided]).all(), "Sparse mask must include all strided mask True positions"


# ---------------------------------------------------------------------------
# 9. build_sparse_mask is always causal (no future tokens)
# ---------------------------------------------------------------------------

def test_build_sparse_mask_causal():
    T = 12
    mask = build_sparse_mask(T, window_size=3, stride=4)
    for i in range(T):
        for j in range(i + 1, T):
            assert not mask[i, j].item(), f"Sparse mask must be causal: mask[{i},{j}] should be False"


# ---------------------------------------------------------------------------
# 10. SparseAttention output shape matches input (B, T, d_model)
# ---------------------------------------------------------------------------

def test_sparse_attention_output_shape():
    cfg = small_config()
    attn = SparseAttention(cfg)
    B, T = 2, 20
    x = torch.randn(B, T, cfg.d_model)
    out = attn(x)
    assert out.shape == (B, T, cfg.d_model)


# ---------------------------------------------------------------------------
# 11. SparseAttention with window_size >= T equivalent to full causal attention
# ---------------------------------------------------------------------------

def test_sparse_attention_full_window():
    T = 8
    cfg = small_config(window_size=T, stride=1)  # stride=1 => every col is strided
    attn = SparseAttention(cfg)
    attn.train(False)

    x = torch.randn(1, T, cfg.d_model)
    with torch.no_grad():
        out = attn(x)

    # With window_size >= T, every causal pair is attended
    assert out.shape == (1, T, cfg.d_model)
    assert not torch.isnan(out).any(), "Output should not contain NaN with full window"


# ---------------------------------------------------------------------------
# 12. SparseTransformerLayer output shape matches input
# ---------------------------------------------------------------------------

def test_sparse_transformer_layer_output_shape():
    cfg = small_config()
    layer = SparseTransformerLayer(cfg)
    B, T = 3, 16
    x = torch.randn(B, T, cfg.d_model)
    out = layer(x)
    assert out.shape == (B, T, cfg.d_model)


# ---------------------------------------------------------------------------
# 13. SparseTransformer output shape is (B, T, vocab_size)
# ---------------------------------------------------------------------------

def test_sparse_transformer_output_shape():
    cfg = small_config()
    vocab_size = 128
    model = SparseTransformer(cfg, n_layers=2, vocab_size=vocab_size)
    B, T = 2, 12
    input_ids = torch.randint(0, vocab_size, (B, T))
    logits = model(input_ids)
    assert logits.shape == (B, T, vocab_size)


# ---------------------------------------------------------------------------
# 14. SparseTransformer is differentiable (backward works)
# ---------------------------------------------------------------------------

def test_sparse_transformer_backward():
    cfg = small_config()
    vocab_size = 128
    model = SparseTransformer(cfg, n_layers=2, vocab_size=vocab_size)
    B, T = 2, 10
    input_ids = torch.randint(0, vocab_size, (B, T))
    logits = model(input_ids)  # (B, T, vocab_size)
    loss = logits.mean()
    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "Backward pass should compute gradients for model parameters"


# ---------------------------------------------------------------------------
# 15. sparsity_ratio returns float in (0, 1]
# ---------------------------------------------------------------------------

def test_sparsity_ratio_range():
    ratio = sparsity_ratio(seq_len=32, window_size=4, stride=4)
    assert isinstance(ratio, float)
    assert 0.0 < ratio <= 1.0, f"sparsity_ratio should be in (0, 1], got {ratio}"


# ---------------------------------------------------------------------------
# 16. sparsity_ratio increases with larger window_size
# ---------------------------------------------------------------------------

def test_sparsity_ratio_increases_with_window():
    seq_len = 64
    stride = 8
    ratio_small = sparsity_ratio(seq_len, window_size=2, stride=stride)
    ratio_large = sparsity_ratio(seq_len, window_size=16, stride=stride)
    assert ratio_large > ratio_small, (
        f"Larger window_size should yield higher sparsity_ratio: "
        f"{ratio_large} > {ratio_small}"
    )
