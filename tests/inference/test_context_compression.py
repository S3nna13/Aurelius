"""Tests for src/inference/context_compression.py."""
from __future__ import annotations

import pytest
import torch

from src.inference.context_compression import (
    CompressionConfig,
    ContextCompressor,
    compute_compression_ratio,
    local_pooling_compression,
    reconstruct_from_mask,
    score_tokens_by_attention,
    score_tokens_by_norm,
    select_top_tokens,
    strided_compression,
)

# ---------------------------------------------------------------------------
# Shared tiny dimensions
# ---------------------------------------------------------------------------
B = 2
T = 12
D = 8
K = 6       # T * 0.5
POOL = 2
STRIDE = 2


# ---------------------------------------------------------------------------
# 1. CompressionConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = CompressionConfig()
    assert cfg.target_ratio == 0.5
    assert cfg.min_tokens == 64
    assert cfg.method == "attention_score"
    assert cfg.stride == 2


# ---------------------------------------------------------------------------
# 2. score_tokens_by_attention — shape
# ---------------------------------------------------------------------------

def test_score_tokens_by_attention_shape():
    hidden = torch.randn(B, T, D)
    query = torch.randn(B, D)
    scores = score_tokens_by_attention(hidden, query)
    assert scores.shape == (B, T), f"Expected ({B},{T}), got {scores.shape}"


# ---------------------------------------------------------------------------
# 3. score_tokens_by_attention — finite values
# ---------------------------------------------------------------------------

def test_score_tokens_by_attention_finite():
    hidden = torch.randn(B, T, D)
    query = torch.randn(B, D)
    scores = score_tokens_by_attention(hidden, query)
    assert torch.isfinite(scores).all(), "Attention scores contain non-finite values"


# ---------------------------------------------------------------------------
# 4. score_tokens_by_norm — shape
# ---------------------------------------------------------------------------

def test_score_tokens_by_norm_shape():
    hidden = torch.randn(B, T, D)
    scores = score_tokens_by_norm(hidden)
    assert scores.shape == (B, T), f"Expected ({B},{T}), got {scores.shape}"


# ---------------------------------------------------------------------------
# 5. score_tokens_by_norm — non-negative
# ---------------------------------------------------------------------------

def test_score_tokens_by_norm_non_negative():
    hidden = torch.randn(B, T, D)
    scores = score_tokens_by_norm(hidden)
    assert (scores >= 0).all(), "Norm scores should be non-negative"


# ---------------------------------------------------------------------------
# 6. select_top_tokens — indices shape
# ---------------------------------------------------------------------------

def test_select_top_tokens_indices_shape():
    scores = torch.randn(B, T)
    indices, _ = select_top_tokens(scores, K)
    assert indices.shape == (B, K), f"Expected ({B},{K}), got {indices.shape}"


# ---------------------------------------------------------------------------
# 7. select_top_tokens — mask shape
# ---------------------------------------------------------------------------

def test_select_top_tokens_mask_shape():
    scores = torch.randn(B, T)
    _, mask = select_top_tokens(scores, K)
    assert mask.shape == (B, T), f"Expected ({B},{T}), got {mask.shape}"


# ---------------------------------------------------------------------------
# 8. select_top_tokens — exactly k True per row
# ---------------------------------------------------------------------------

def test_select_top_tokens_mask_count():
    scores = torch.randn(B, T)
    _, mask = select_top_tokens(scores, K)
    counts = mask.sum(dim=-1)
    assert (counts == K).all(), f"Each row should have exactly {K} True entries, got {counts}"


# ---------------------------------------------------------------------------
# 9. strided_compression — output shape
# ---------------------------------------------------------------------------

def test_strided_compression_shape():
    hidden = torch.randn(B, T, D)
    out = strided_compression(hidden, STRIDE)
    expected_t = T // STRIDE
    assert out.shape == (B, expected_t, D), f"Expected ({B},{expected_t},{D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 10. local_pooling_compression — output shape
# ---------------------------------------------------------------------------

def test_local_pooling_compression_shape():
    hidden = torch.randn(B, T, D)
    out = local_pooling_compression(hidden, POOL)
    expected_t = T // POOL
    assert out.shape == (B, expected_t, D), f"Expected ({B},{expected_t},{D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 11. ContextCompressor — compressed hidden shape
# ---------------------------------------------------------------------------

def test_context_compressor_output_hidden_shape():
    cfg = CompressionConfig(target_ratio=0.5, min_tokens=1)
    compressor = ContextCompressor(D, cfg)
    hidden = torch.randn(B, T, D)
    compressed, _ = compressor(hidden)
    assert compressed.shape == (B, K, D), f"Expected ({B},{K},{D}), got {compressed.shape}"


# ---------------------------------------------------------------------------
# 12. ContextCompressor — selection mask shape
# ---------------------------------------------------------------------------

def test_context_compressor_mask_shape():
    cfg = CompressionConfig(target_ratio=0.5, min_tokens=1)
    compressor = ContextCompressor(D, cfg)
    hidden = torch.randn(B, T, D)
    _, mask = compressor(hidden)
    assert mask.shape == (B, T), f"Expected ({B},{T}), got {mask.shape}"


# ---------------------------------------------------------------------------
# 13. compute_compression_ratio
# ---------------------------------------------------------------------------

def test_compute_compression_ratio():
    ratio = compute_compression_ratio(original_len=12, compressed_len=6)
    assert abs(ratio - 0.5) < 1e-6, f"Expected 0.5, got {ratio}"


# ---------------------------------------------------------------------------
# 14. reconstruct_from_mask — output shape
# ---------------------------------------------------------------------------

def test_reconstruct_from_mask_shape():
    cfg = CompressionConfig(target_ratio=0.5, min_tokens=1)
    compressor = ContextCompressor(D, cfg)
    hidden = torch.randn(B, T, D)
    compressed, mask = compressor(hidden)
    reconstructed = reconstruct_from_mask(compressed, mask)
    assert reconstructed.shape == (B, T, D), (
        f"Expected ({B},{T},{D}), got {reconstructed.shape}"
    )


# ---------------------------------------------------------------------------
# 15. reconstruct_from_mask — fill value at unselected positions
# ---------------------------------------------------------------------------

def test_reconstruct_from_mask_fill_value():
    fill = -999.0
    scores = torch.randn(B, T)
    _, mask = select_top_tokens(scores, K)
    compressed = torch.ones(B, K, D)
    reconstructed = reconstruct_from_mask(compressed, mask, fill_value=fill)
    unselected = reconstructed[~mask]
    assert (unselected == fill).all(), "Unselected positions should contain fill_value"
