"""Tests for src/inference/token_merging.py"""

from __future__ import annotations

import torch
import pytest

from src.inference.token_merging import (
    ToMeConfig,
    compute_token_similarity,
    bipartite_matching,
    merge_tokens,
    unmerge_tokens,
    ToMeLayer,
    apply_tome_to_hidden_states,
)

# Fixed dimensions for all tests
B, T, D = 2, 16, 32


# ---------------------------------------------------------------------------
# 1. ToMeConfig defaults
# ---------------------------------------------------------------------------

def test_tome_config_defaults():
    cfg = ToMeConfig()
    assert cfg.r == 8
    assert cfg.merge_mode == "mean"
    assert cfg.similarity_threshold == 0.0


# ---------------------------------------------------------------------------
# 2. compute_token_similarity — output shape
# ---------------------------------------------------------------------------

def test_compute_token_similarity_shape():
    x = torch.randn(B, T, D)
    sim = compute_token_similarity(x)
    assert sim.shape == (B, T - 1), f"Expected ({B}, {T - 1}), got {sim.shape}"


# ---------------------------------------------------------------------------
# 3. compute_token_similarity — identical tokens → similarity ≈ 1.0
# ---------------------------------------------------------------------------

def test_compute_token_similarity_identical():
    # All tokens are the same vector → cosim = 1.0
    vec = torch.randn(D)
    x = vec.unsqueeze(0).unsqueeze(0).expand(B, T, D).clone()
    sim = compute_token_similarity(x)
    assert torch.allclose(sim, torch.ones_like(sim), atol=1e-5), \
        f"Expected all 1.0, got min={sim.min().item():.4f}"


# ---------------------------------------------------------------------------
# 4. compute_token_similarity — orthogonal tokens → similarity ≈ 0.0
# ---------------------------------------------------------------------------

def test_compute_token_similarity_orthogonal():
    # Alternate between two orthogonal unit vectors
    e0 = torch.zeros(D)
    e0[0] = 1.0
    e1 = torch.zeros(D)
    e1[1] = 1.0

    tokens = []
    for i in range(T):
        tokens.append(e0 if i % 2 == 0 else e1)
    x = torch.stack(tokens).unsqueeze(0).expand(B, -1, -1).clone()  # (B, T, D)
    sim = compute_token_similarity(x)
    # Adjacent pairs (e0,e1) are orthogonal → cosim = 0
    assert torch.allclose(sim, torch.zeros_like(sim), atol=1e-5), \
        f"Expected all 0.0, got max={sim.abs().max().item():.4f}"


# ---------------------------------------------------------------------------
# 5. bipartite_matching — output shapes
# ---------------------------------------------------------------------------

def test_bipartite_matching_output_shapes():
    r = 4
    sim = torch.rand(B, T - 1)
    merge_idx, keep_idx = bipartite_matching(sim, r=r)
    r_actual = merge_idx.shape[1]
    assert merge_idx.shape == (B, r_actual)
    assert keep_idx.shape == (B, T - r_actual)


# ---------------------------------------------------------------------------
# 6. bipartite_matching — r=0 → empty merge, all kept
# ---------------------------------------------------------------------------

def test_bipartite_matching_r0():
    sim = torch.rand(B, T - 1)
    merge_idx, keep_idx = bipartite_matching(sim, r=0)
    assert merge_idx.shape == (B, 0)
    assert keep_idx.shape == (B, T)


# ---------------------------------------------------------------------------
# 7. bipartite_matching — threshold filters low-similarity pairs
# ---------------------------------------------------------------------------

def test_bipartite_matching_threshold_filters():
    # All similarities are 0.3 — below threshold of 0.5 → nothing merged
    sim = torch.full((B, T - 1), 0.3)
    merge_idx, keep_idx = bipartite_matching(sim, r=4, threshold=0.5)
    assert merge_idx.shape[1] == 0, "All pairs below threshold should be filtered"
    assert keep_idx.shape == (B, T)


# ---------------------------------------------------------------------------
# 8. merge_tokens — output shape
# ---------------------------------------------------------------------------

def test_merge_tokens_output_shape():
    r = 4
    x = torch.randn(B, T, D)
    sim = torch.rand(B, T - 1)
    merge_idx, keep_idx = bipartite_matching(sim, r=r)
    r_actual = merge_idx.shape[1]
    merged = merge_tokens(x, merge_idx, keep_idx, mode="mean")
    assert merged.shape == (B, T - r_actual, D), \
        f"Expected ({B}, {T - r_actual}, {D}), got {merged.shape}"


# ---------------------------------------------------------------------------
# 9. merge_tokens — mode="mean" averages correctly
# ---------------------------------------------------------------------------

def test_merge_tokens_mean_averages_correctly():
    # Simple case: B=1, T=4, D=2, r=1
    # tokens: [a, b, c, d]; merge pair (b→a), i.e. merge_index=1 (b), kept predecessor=0 (a)
    # After mean merge: position 0 = (a+b)/2, positions {1,2,3} kept minus 1
    B1, T1, D1 = 1, 4, 2
    x = torch.tensor([[[1.0, 1.0],   # token 0: a
                        [3.0, 3.0],   # token 1: b  ← will be merged into 0
                        [5.0, 5.0],   # token 2: c
                        [7.0, 7.0]]], # token 3: d
                      dtype=torch.float32)
    # Force merge_indices = [[1]], keep_indices = [[0, 2, 3]]
    merge_idx = torch.tensor([[1]], dtype=torch.long)
    keep_idx  = torch.tensor([[0, 2, 3]], dtype=torch.long)

    merged = merge_tokens(x, merge_idx, keep_idx, mode="mean")
    assert merged.shape == (1, 3, 2)
    # Token 0 should be average of [1,1] and [3,3] = [2,2]
    assert torch.allclose(merged[0, 0], torch.tensor([2.0, 2.0])), \
        f"Expected [2, 2], got {merged[0, 0]}"
    # Token 1 (original 2) and token 2 (original 3) unchanged
    assert torch.allclose(merged[0, 1], torch.tensor([5.0, 5.0]))
    assert torch.allclose(merged[0, 2], torch.tensor([7.0, 7.0]))


# ---------------------------------------------------------------------------
# 10. unmerge_tokens — output shape
# ---------------------------------------------------------------------------

def test_unmerge_tokens_output_shape():
    r = 4
    x = torch.randn(B, T, D)
    sim = torch.rand(B, T - 1)
    merge_idx, keep_idx = bipartite_matching(sim, r=r)
    merged = merge_tokens(x, merge_idx, keep_idx, mode="mean")

    reconstructed = unmerge_tokens(merged, merge_idx, keep_idx, T_orig=T)
    assert reconstructed.shape == (B, T, D), \
        f"Expected ({B}, {T}, {D}), got {reconstructed.shape}"


# ---------------------------------------------------------------------------
# 11. ToMeLayer forward — returns tuple with correct keys
# ---------------------------------------------------------------------------

def test_tome_layer_forward_keys():
    cfg = ToMeConfig(r=4)
    layer = ToMeLayer(cfg)
    x = torch.randn(B, T, D)
    merged_x, merge_info = layer(x)

    assert isinstance(merged_x, torch.Tensor)
    assert isinstance(merge_info, dict)
    for key in ("merge_indices", "keep_indices", "compression_ratio"):
        assert key in merge_info, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 12. ToMeLayer — compression_ratio < 1.0 when r > 0
# ---------------------------------------------------------------------------

def test_tome_layer_compression_ratio():
    # Use high similarities so that threshold=0.0 causes actual merges
    cfg = ToMeConfig(r=4, similarity_threshold=0.0)
    layer = ToMeLayer(cfg)

    # Build x with identical adjacent tokens (similarity=1.0) to guarantee merges
    base = torch.randn(B, 1, D)
    x = base.expand(B, T, D).clone()  # all tokens identical → similarity=1.0

    merged_x, merge_info = layer(x)
    ratio = merge_info["compression_ratio"]
    assert ratio < 1.0, f"Expected compression_ratio < 1.0, got {ratio}"
