"""
Tests for kv_cache_compression_v2.py

Tiny configs: budget=4, B=2, H=2, T=8, D_head=8, recent_window=2
Every test runs actual forward (and where relevant backward) passes.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import pytest

from src.inference.kv_cache_compression_v2 import (
    AttentionScoreAccumulator,
    H2OEvictionPolicy,
    SnapKVPolicy,
    CompressedKVCache,
    KVCompressionAnalyzer,
)

# ---------------------------------------------------------------------------
# Shared tiny dimensions
# ---------------------------------------------------------------------------
B, H, T, D = 2, 2, 8, 8
BUDGET = 4
RECENT = 2


def _rand(*shape) -> torch.Tensor:
    return torch.randn(*shape)


def _attn_weights(b=B, h=H, t_q=4, t_kv=T) -> torch.Tensor:
    """Random valid attention weight matrix (sums to 1 along t_kv)."""
    raw = torch.rand(b, h, t_q, t_kv)
    return raw / raw.sum(dim=-1, keepdim=True)


# ===========================================================================
# 1. AttentionScoreAccumulator.update — shape check
# ===========================================================================
def test_accumulator_update_shape():
    acc = AttentionScoreAccumulator(recent_window=RECENT)
    w = _attn_weights()
    acc.update(w)
    assert acc.accumulated is not None
    assert acc.accumulated.shape == (B, H, T), (
        f"Expected (B={B}, H={H}, T={T}), got {acc.accumulated.shape}"
    )


# ===========================================================================
# 2. AttentionScoreAccumulator.importance_scores — sums to 1 along T_kv
# ===========================================================================
def test_accumulator_importance_scores_normalized():
    acc = AttentionScoreAccumulator(recent_window=RECENT)
    acc.update(_attn_weights())
    scores = acc.importance_scores()  # (B, H, T)
    assert scores.shape == (B, H, T)
    # Each (b, h) slice should sum to 1
    sums = scores.sum(dim=-1)  # (B, H)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"Scores do not sum to 1; max deviation {(sums - 1).abs().max().item()}"
    )


# ===========================================================================
# 3. AttentionScoreAccumulator.reset — clears state
# ===========================================================================
def test_accumulator_reset_clears_state():
    acc = AttentionScoreAccumulator(recent_window=RECENT)
    acc.update(_attn_weights())
    assert acc.accumulated is not None

    acc.reset()
    assert acc.accumulated is None

    # After reset, a fresh update with different T should work cleanly
    new_w = _attn_weights(t_q=2, t_kv=3)
    acc.update(new_w)
    assert acc.accumulated.shape == (B, H, 3)


# ===========================================================================
# 4. H2OEvictionPolicy.select_keep_indices — always includes last recent_window
# ===========================================================================
def test_h2o_select_keep_indices_recency():
    policy = H2OEvictionPolicy(budget=BUDGET, recent_window=RECENT)
    scores = torch.rand(B, H, T)
    keep = policy.select_keep_indices(scores, current_len=T)
    # Last RECENT indices must appear in every row
    for b in range(B):
        kept = keep[b].tolist()
        for idx in range(T - RECENT, T):
            assert idx in kept, f"Recent index {idx} missing from batch {b}: {kept}"


# ===========================================================================
# 5. H2OEvictionPolicy.evict — output shape (B,H,budget,D) when T > budget
# ===========================================================================
def test_h2o_evict_shape():
    policy = H2OEvictionPolicy(budget=BUDGET, recent_window=RECENT)
    keys = _rand(B, H, T, D)
    vals = _rand(B, H, T, D)
    scores = torch.rand(B, H, T)
    k_out, v_out = policy.evict(keys, vals, scores)
    assert k_out.shape == (B, H, BUDGET, D), f"keys shape {k_out.shape}"
    assert v_out.shape == (B, H, BUDGET, D), f"values shape {v_out.shape}"


# ===========================================================================
# 6. H2OEvictionPolicy — no eviction when T ≤ budget (return all indices)
# ===========================================================================
def test_h2o_no_eviction_when_short():
    policy = H2OEvictionPolicy(budget=BUDGET, recent_window=RECENT)
    short_T = BUDGET - 1  # 3 < budget=4
    scores = torch.rand(B, H, short_T)
    keep = policy.select_keep_indices(scores, current_len=short_T)
    # Should return all indices 0..short_T-1
    expected = set(range(short_T))
    for b in range(B):
        assert set(keep[b].tolist()) == expected, (
            f"Expected all {expected}, got {set(keep[b].tolist())}"
        )


# ===========================================================================
# 7. SnapKVPolicy.compress — output shape (B,H,budget,D)
# ===========================================================================
def test_snapkv_compress_shape():
    policy = SnapKVPolicy(budget=BUDGET, observation_window=RECENT)
    keys = _rand(B, H, T, D)
    vals = _rand(B, H, T, D)
    qw = _rand(B, H, RECENT, D)
    k_out, v_out = policy.compress(keys, vals, qw)
    assert k_out.shape == (B, H, BUDGET, D), f"keys shape {k_out.shape}"
    assert v_out.shape == (B, H, BUDGET, D), f"values shape {v_out.shape}"


# ===========================================================================
# 8. SnapKVPolicy — importance scores derived from query-key attention
# ===========================================================================
def test_snapkv_importance_from_query_key():
    """
    Place strong signal in key position 0 so the query strongly attends to it.
    After compression that position should be retained.
    """
    torch.manual_seed(0)
    policy = SnapKVPolicy(budget=BUDGET, observation_window=RECENT, pooling="mean")
    D_local = 8

    # Craft keys: position 0 is a very large magnitude vector
    keys = torch.zeros(B, H, T, D_local)
    keys[:, :, 0, :] = 10.0  # dominant key

    vals = _rand(B, H, T, D_local)

    # Query aligned with key[0]
    qw = torch.zeros(B, H, RECENT, D_local)
    qw[:, :, :, :] = 10.0  # will strongly attend to keys[:,:,0,:]

    k_out, v_out = policy.compress(keys, vals, qw)
    # Position 0 should be among the kept positions for at least batch 0
    # We check by comparing the kept key values against the large-value key
    max_val = k_out.abs().max().item()
    assert max_val >= 5.0, (
        f"Expected dominant key (value≈10) to be kept; max in output: {max_val}"
    )


# ===========================================================================
# 9. CompressedKVCache "h2o" — cache_size stays ≤ budget after enough updates
# ===========================================================================
def test_compressed_cache_h2o_bounded():
    cache = CompressedKVCache(policy="h2o", budget=BUDGET, recent_window=RECENT)
    for step in range(5):
        t_new = 2
        k = _rand(B, H, t_new, D)
        v = _rand(B, H, t_new, D)
        w = _attn_weights(t_q=t_new, t_kv=(step + 1) * t_new)
        # pad/trim w so t_kv matches actual cache size after cat
        cache.update(0, k, v, attn_weights=w)
    assert cache.cache_size(0) <= BUDGET, (
        f"cache_size {cache.cache_size(0)} exceeds budget {BUDGET}"
    )


# ===========================================================================
# 10. CompressedKVCache "none" — cache grows unbounded
# ===========================================================================
def test_compressed_cache_none_unbounded():
    cache = CompressedKVCache(policy="none", budget=BUDGET)
    total = 0
    steps = 4
    t_new = 2
    for _ in range(steps):
        k = _rand(B, H, t_new, D)
        v = _rand(B, H, t_new, D)
        cache.update(0, k, v)
        total += t_new
    assert cache.cache_size(0) == total, (
        f"Expected {total}, got {cache.cache_size(0)}"
    )


# ===========================================================================
# 11. CompressedKVCache.reset — cache_size becomes 0
# ===========================================================================
def test_compressed_cache_reset():
    cache = CompressedKVCache(policy="h2o", budget=BUDGET, recent_window=RECENT)
    k = _rand(B, H, T, D)
    v = _rand(B, H, T, D)
    cache.update(0, k, v)
    assert cache.cache_size(0) > 0
    cache.reset()
    assert cache.cache_size(0) == 0


# ===========================================================================
# 12. KVCompressionAnalyzer.memory_reduction — in [0,1], 0 if no compression
# ===========================================================================
def test_analyzer_memory_reduction():
    analyzer = KVCompressionAnalyzer()
    # No compression
    r = analyzer.memory_reduction(original_len=8, compressed_len=8)
    assert r == pytest.approx(0.0), f"Expected 0.0, got {r}"
    # Half compression
    r2 = analyzer.memory_reduction(original_len=8, compressed_len=4)
    assert r2 == pytest.approx(0.5), f"Expected 0.5, got {r2}"
    # Fully compressed (1 token)
    r3 = analyzer.memory_reduction(original_len=8, compressed_len=1)
    assert 0.0 <= r3 <= 1.0


# ===========================================================================
# 13. KVCompressionAnalyzer.attention_approximation_error — 0.0 for identical
# ===========================================================================
def test_analyzer_attention_error_identical():
    analyzer = KVCompressionAnalyzer()
    out = _rand(B, T, D)
    err = analyzer.attention_approximation_error(out, out)
    assert err == pytest.approx(0.0, abs=1e-6), f"Expected 0.0, got {err}"


# ===========================================================================
# 14. KVCompressionAnalyzer.perplexity_degradation — 0.0 for identical dists
# ===========================================================================
def test_analyzer_perplexity_degradation_identical():
    analyzer = KVCompressionAnalyzer()
    V = 16
    logits = torch.randn(B, T, V)
    logprobs = F.log_softmax(logits, dim=-1)
    kl = analyzer.perplexity_degradation(logprobs, logprobs)
    assert kl == pytest.approx(0.0, abs=1e-5), f"Expected 0.0, got {kl}"


# ===========================================================================
# 15. Full cache update cycle — update 3 times → correct shape maintained
# ===========================================================================
def test_full_cache_update_cycle():
    """
    Simulate 3 decode steps with SnapKV cache; after each step shape is valid.
    """
    obs_window = RECENT
    cache = CompressedKVCache(
        policy="snapkv",
        budget=BUDGET,
        observation_window=obs_window,
        pooling="mean",
    )

    for step in range(3):
        t_new = 3  # generate 3 tokens per step
        k_new = _rand(B, H, t_new, D)
        v_new = _rand(B, H, t_new, D)
        # attn_weights supplied as a dummy (not used by snapkv path directly)
        k_stored, v_stored = cache.update(0, k_new, v_new)
        sz = cache.cache_size(0)

        # After compression the size must be ≤ budget (or ≤ total if still small)
        assert v_stored.shape[2] == sz, "stored shape mismatch with reported size"
        assert sz <= max(BUDGET, (step + 1) * t_new), (
            f"Step {step}: unexpected cache size {sz}"
        )

    # Final state: must be ≤ budget (9 tokens accumulated, budget=4)
    assert cache.cache_size(0) <= BUDGET, (
        f"Final cache size {cache.cache_size(0)} exceeds budget {BUDGET}"
    )
