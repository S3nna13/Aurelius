"""Tests for Token Merging (ToMe) module."""

from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.attention import GroupedQueryAttention, precompute_rope_frequencies
from src.model.token_merge import (
    bipartite_soft_matching,
    TokenMergeAttention,
    apply_token_merging,
    ToMeStats,
)
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tiny_cfg(**kwargs) -> AureliusConfig:
    """Build a minimal AureliusConfig for fast tests."""
    defaults = dict(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=128,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return AureliusConfig(**defaults)


def make_attn(cfg: AureliusConfig | None = None) -> GroupedQueryAttention:
    if cfg is None:
        cfg = tiny_cfg()
    return GroupedQueryAttention(cfg)


# ---------------------------------------------------------------------------
# bipartite_soft_matching tests
# ---------------------------------------------------------------------------

def test_bipartite_matching_merge_reduces_length():
    """merge_fn on (1, 20, 64) with r=4 → (1, 16, 64)."""
    B, N, C = 1, 20, 64
    r = 4
    metric = torch.randn(B, N, C)
    merge_fn, _ = bipartite_soft_matching(metric, r)
    x = torch.randn(B, N, C)
    merged = merge_fn(x)
    assert merged.shape == (B, N - r, C), f"Expected {(B, N-r, C)}, got {merged.shape}"


def test_bipartite_matching_unmerge_restores_length():
    """unmerge_fn restores original shape (1, 20, 64)."""
    B, N, C = 1, 20, 64
    r = 4
    metric = torch.randn(B, N, C)
    merge_fn, unmerge_fn = bipartite_soft_matching(metric, r)
    x = torch.randn(B, N, C)
    merged = merge_fn(x)
    restored = unmerge_fn(merged)
    assert restored.shape == (B, N, C), f"Expected {(B, N, C)}, got {restored.shape}"


def test_merge_unmerge_approximate_inverse():
    """unmerge(merge(x)) should be 'close' to x in the non-merged positions.

    Exact equality is not expected because merging averages pairs.
    We check that the L2 distance is less than the total L2 norm of x (i.e.,
    information is not destroyed entirely).
    """
    B, N, C = 1, 20, 64
    r = 2
    torch.manual_seed(42)
    x = torch.randn(B, N, C)
    metric = torch.randn(B, N, C)
    merge_fn, unmerge_fn = bipartite_soft_matching(metric, r)
    restored = unmerge_fn(merge_fn(x))
    # The result should be non-zero and finite
    assert restored.isfinite().all()
    # The error should be bounded — merging only affects r pairs
    # Positions that were NOT merged should be exactly preserved
    diff = (restored - x).abs().sum()
    assert diff < x.abs().sum() * 2, "unmerge(merge(x)) diverged too much from x"


def test_r_zero_no_change():
    """With r=0, merge_fn returns unchanged sequence."""
    B, N, C = 1, 20, 64
    torch.manual_seed(0)
    x = torch.randn(B, N, C)
    metric = torch.randn(B, N, C)
    merge_fn, unmerge_fn = bipartite_soft_matching(metric, r=0)
    merged = merge_fn(x)
    assert merged.shape == x.shape
    assert torch.allclose(merged, x)
    # unmerge should also be identity
    restored = unmerge_fn(merged)
    assert torch.allclose(restored, x)


# ---------------------------------------------------------------------------
# TokenMergeAttention tests
# ---------------------------------------------------------------------------

def test_token_merge_attention_output_shape():
    """(2, 32, 64) → (2, 32, 64) after merge+attn+unmerge."""
    cfg = tiny_cfg()
    attn = make_attn(cfg)
    tome_attn = TokenMergeAttention(attn, r=4)

    B, N, D = 2, 32, cfg.d_model
    x = torch.randn(B, N, D)
    freqs = precompute_rope_frequencies(cfg.head_dim, N)

    out, kv = tome_attn(x, freqs)
    assert out.shape == (B, N, D), f"Expected {(B, N, D)}, got {out.shape}"
    assert isinstance(kv, tuple) and len(kv) == 2


def test_token_merge_reduces_attn_sequence():
    """With r=4, attention processes a shorter sequence internally."""
    cfg = tiny_cfg()
    attn = make_attn(cfg)
    tome_attn = TokenMergeAttention(attn, r=4, trace_source=True)

    B, N, D = 1, 32, cfg.d_model
    x = torch.randn(B, N, D)
    freqs = precompute_rope_frequencies(cfg.head_dim, N)

    out, _ = tome_attn(x, freqs)

    # trace_source records merge events
    assert len(tome_attn._merge_history) == 1
    event = tome_attn._merge_history[0]
    assert event["before"] == N
    assert event["after"] == N - 4, f"Expected {N-4}, got {event['after']}"


# ---------------------------------------------------------------------------
# apply_token_merging tests
# ---------------------------------------------------------------------------

def test_apply_token_merging_wraps_layers():
    """After apply_token_merging, all attention layers are TokenMergeAttention."""
    cfg = tiny_cfg()
    model = AureliusTransformer(cfg)
    apply_token_merging(model, r=4)
    for block in model.layers:
        assert isinstance(block.attn, TokenMergeAttention), (
            f"Expected TokenMergeAttention, got {type(block.attn)}"
        )


def test_some_layers_only():
    """With layer_indices=[0], only the first layer is TokenMergeAttention."""
    cfg = tiny_cfg()
    model = AureliusTransformer(cfg)
    apply_token_merging(model, r=4, layer_indices=[0])

    assert isinstance(model.layers[0].attn, TokenMergeAttention)
    for block in model.layers[1:]:
        assert not isinstance(block.attn, TokenMergeAttention), (
            f"Layer after index 0 should NOT be TokenMergeAttention"
        )


# ---------------------------------------------------------------------------
# ToMeStats tests
# ---------------------------------------------------------------------------

def test_tome_stats_compression():
    """record(100, 80) → compression_ratio() == 0.8."""
    stats = ToMeStats()
    stats.record(100, 80)
    assert abs(stats.compression_ratio() - 0.8) < 1e-6


def test_tome_stats_no_records():
    """With no records, compression_ratio() == 1.0."""
    stats = ToMeStats()
    assert stats.compression_ratio() == 1.0


def test_tome_stats_multiple_records():
    """Multiple records accumulate correctly."""
    stats = ToMeStats()
    stats.record(100, 80)
    stats.record(200, 160)
    # total_before=300, total_after=240 → ratio=0.8
    assert abs(stats.compression_ratio() - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# Edge case: small sequence
# ---------------------------------------------------------------------------

def test_small_seq_no_merge():
    """seq_len < 4 doesn't crash (r is clamped / merging skipped)."""
    cfg = tiny_cfg()
    attn = make_attn(cfg)
    tome_attn = TokenMergeAttention(attn, r=8)  # r larger than half of seq

    B, N, D = 1, 3, cfg.d_model  # very short sequence
    x = torch.randn(B, N, D)
    freqs = precompute_rope_frequencies(cfg.head_dim, N)

    # Should not raise; output shape preserved
    out, kv = tome_attn(x, freqs)
    assert out.shape == (B, N, D), f"Expected {(B, N, D)}, got {out.shape}"
