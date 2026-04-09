"""Tests for src/inference/kv_compression.py"""

from __future__ import annotations

import math

import pytest
import torch

from src.inference.kv_compression import (
    KVCompressionConfig,
    KVCacheCompressor,
    compress_kv_cache,
    compute_token_importance_from_attn,
    estimate_compression_quality,
    scissorhands_score,
    select_tokens_to_keep,
    streaming_kv_drop,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B, H, T, D = 1, 2, 16, 8
V = 32  # vocab size for logit tests


def make_kv_cache(layers: int = 2) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create a dummy KV cache with shape (B, H, T, D)."""
    return [
        (torch.randn(B, H, T, D), torch.randn(B, H, T, D))
        for _ in range(layers)
    ]


def make_attn_weights_batched() -> torch.Tensor:
    """(B, H, T, T) attention weights (positive, normalized over last dim)."""
    raw = torch.rand(B, H, T, T)
    return raw / raw.sum(dim=-1, keepdim=True)


def make_attn_weights_unbatched() -> torch.Tensor:
    """(H, T, T) attention weights."""
    raw = torch.rand(H, T, T)
    return raw / raw.sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = KVCompressionConfig()
    assert cfg.strategy == "heavy_hitter"
    assert cfg.keep_ratio == 0.5


# ---------------------------------------------------------------------------
# 2. compute_token_importance_from_attn — sum, shape
# ---------------------------------------------------------------------------


def test_compute_importance_sum_shape():
    attn = make_attn_weights_unbatched()  # (H, T, T)
    scores = compute_token_importance_from_attn(attn, method="sum")
    assert scores.shape == (T,), f"Expected ({T},), got {scores.shape}"


# ---------------------------------------------------------------------------
# 3. compute_token_importance_from_attn — max, shape
# ---------------------------------------------------------------------------


def test_compute_importance_max_shape():
    attn = make_attn_weights_unbatched()  # (H, T, T)
    scores = compute_token_importance_from_attn(attn, method="max")
    assert scores.shape == (T,), f"Expected ({T},), got {scores.shape}"


# ---------------------------------------------------------------------------
# 4. Importance scores are non-negative
# ---------------------------------------------------------------------------


def test_compute_importance_nonneg():
    attn = make_attn_weights_unbatched()
    scores_sum = compute_token_importance_from_attn(attn, method="sum")
    scores_max = compute_token_importance_from_attn(attn, method="max")
    assert (scores_sum >= 0).all(), "sum scores should be non-negative"
    assert (scores_max >= 0).all(), "max scores should be non-negative"


# ---------------------------------------------------------------------------
# 5. select_tokens_to_keep — always keeps init tokens
# ---------------------------------------------------------------------------


def test_select_tokens_always_keeps_init():
    n_init = 4
    importance = torch.rand(T)
    indices = select_tokens_to_keep(importance, keep_ratio=0.5, local_window=4, n_init_tokens=n_init)
    idx_set = set(indices.tolist())
    for i in range(n_init):
        assert i in idx_set, f"Init token {i} not in kept indices"


# ---------------------------------------------------------------------------
# 6. select_tokens_to_keep — always keeps local window
# ---------------------------------------------------------------------------


def test_select_tokens_always_keeps_local():
    local_window = 4
    importance = torch.rand(T)
    indices = select_tokens_to_keep(importance, keep_ratio=0.5, local_window=local_window, n_init_tokens=2)
    idx_set = set(indices.tolist())
    for i in range(T - local_window, T):
        assert i in idx_set, f"Local token {i} not in kept indices"


# ---------------------------------------------------------------------------
# 7. select_tokens_to_keep — indices are sorted
# ---------------------------------------------------------------------------


def test_select_tokens_sorted():
    importance = torch.rand(T)
    indices = select_tokens_to_keep(importance, keep_ratio=0.5, local_window=4, n_init_tokens=4)
    idx_list = indices.tolist()
    assert idx_list == sorted(idx_list), "Indices should be sorted"


# ---------------------------------------------------------------------------
# 8. select_tokens_to_keep — approximately correct count
# ---------------------------------------------------------------------------


def test_select_tokens_count():
    n_init = 4
    local_window = 4
    keep_ratio = 0.5
    importance = torch.rand(T)
    indices = select_tokens_to_keep(
        importance, keep_ratio=keep_ratio, local_window=local_window, n_init_tokens=n_init
    )
    # Should keep at most T tokens and at least mandatory tokens
    mandatory_count = len(set(range(n_init)) | set(range(T - local_window, T)))
    assert len(indices) >= mandatory_count
    assert len(indices) <= T


# ---------------------------------------------------------------------------
# 9. compress_kv_cache — output tensors have T_keep on T dimension
# ---------------------------------------------------------------------------


def test_compress_kv_cache_shape():
    kv = make_kv_cache(layers=2)
    importance = torch.rand(T)
    keep_indices = select_tokens_to_keep(importance, keep_ratio=0.5, local_window=4, n_init_tokens=4)
    T_keep = len(keep_indices)
    compressed = compress_kv_cache(kv, keep_indices)
    for k, v in compressed:
        assert k.shape[2] == T_keep, f"K T dim: expected {T_keep}, got {k.shape[2]}"
        assert v.shape[2] == T_keep, f"V T dim: expected {T_keep}, got {v.shape[2]}"


# ---------------------------------------------------------------------------
# 10. compress_kv_cache — same number of layer pairs
# ---------------------------------------------------------------------------


def test_compress_kv_cache_layer_count():
    n_layers = 3
    kv = make_kv_cache(layers=n_layers)
    keep_indices = torch.arange(8)  # keep first 8 tokens
    compressed = compress_kv_cache(kv, keep_indices)
    assert len(compressed) == n_layers


# ---------------------------------------------------------------------------
# 11. streaming_kv_drop — output has n_sink + window_size tokens on T dim
# ---------------------------------------------------------------------------


def test_streaming_kv_drop_shape():
    kv = make_kv_cache(layers=2)
    n_sink = 4
    window_size = 6
    result = streaming_kv_drop(kv, n_sink=n_sink, window_size=window_size)
    expected_T = n_sink + window_size
    for k, v in result:
        assert k.shape[2] == expected_T, f"K T dim: expected {expected_T}, got {k.shape[2]}"
        assert v.shape[2] == expected_T, f"V T dim: expected {expected_T}, got {v.shape[2]}"


# ---------------------------------------------------------------------------
# 12. scissorhands_score — output shape (H, T)
# ---------------------------------------------------------------------------


def test_scissorhands_score_shape():
    attn = make_attn_weights_unbatched()  # (H, T, T)
    history = torch.zeros(H, T)
    result = scissorhands_score(attn, history, decay=0.9)
    assert result.shape == (H, T), f"Expected ({H}, {T}), got {result.shape}"


# ---------------------------------------------------------------------------
# 13. scissorhands_score — with decay=0, returns only current attn contribution
# ---------------------------------------------------------------------------


def test_scissorhands_score_decay():
    attn = make_attn_weights_unbatched()  # (H, T, T)
    history = torch.ones(H, T) * 100.0  # large values to be zeroed out by decay=0
    result = scissorhands_score(attn, history, decay=0.0)
    expected = attn.sum(dim=-2)  # (H, T)
    assert torch.allclose(result, expected, atol=1e-5), (
        "With decay=0, result should equal current attn summed over queries"
    )


# ---------------------------------------------------------------------------
# 14. estimate_compression_quality — dict has all 3 keys
# ---------------------------------------------------------------------------


def test_estimate_compression_quality_keys():
    original_logits = torch.randn(B, T, V)
    compressed_logits = torch.randn(B, T, V)
    result = estimate_compression_quality(original_logits, compressed_logits)
    assert "kl_divergence" in result
    assert "top1_agreement" in result
    assert "perplexity_ratio" in result


# ---------------------------------------------------------------------------
# 15. KVCacheCompressor — heavy_hitter compress() reduces T dimension
# ---------------------------------------------------------------------------


def test_kv_compressor_heavy_hitter():
    cfg = KVCompressionConfig(strategy="heavy_hitter", keep_ratio=0.5, local_window=4, n_init_tokens=4)
    compressor = KVCacheCompressor(cfg)
    kv = make_kv_cache(layers=2)
    attn = make_attn_weights_batched()  # (B, H, T, T)
    compressed = compressor.compress(kv, attention_weights=attn)
    original_T = T
    for k, v in compressed:
        assert k.shape[2] < original_T, (
            f"Expected compressed T < {original_T}, got {k.shape[2]}"
        )


# ---------------------------------------------------------------------------
# 16. KVCacheCompressor — get_compression_stats returns correct keys
# ---------------------------------------------------------------------------


def test_kv_compressor_stats_keys():
    cfg = KVCompressionConfig()
    compressor = KVCacheCompressor(cfg)
    stats = compressor.get_compression_stats(original_len=16, compressed_len=8)
    assert "compression_ratio" in stats
    assert "kept_tokens" in stats
    assert "dropped_tokens" in stats
    assert stats["kept_tokens"] == 8.0
    assert stats["dropped_tokens"] == 8.0
    assert math.isclose(stats["compression_ratio"], 0.5)
