"""Tests for src/model/token_merging_v2.py — ToMe bipartite soft matching.

Tiny config: d_model=16, n_heads=2, seq_len=8, r=2, batch=2, n_layers=2,
             vocab=16. Every test runs a forward or forward+backward pass.
"""

from __future__ import annotations

import math

import torch

from src.model.token_merging_v2 import (
    BipartiteSoftMatching,
    TokenUnmerger,
    ToMeAttention,
    ToMeEfficiencyAnalyzer,
    ToMeModel,
    ToMeTransformerBlock,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------
B = 2  # batch
T = 8  # seq_len
D = 16  # d_model
H = 2  # n_heads
R = 2  # r (pairs merged per layer)
NL = 2  # n_layers
V = 16  # vocab_size


def _rand(B, T, D, requires_grad=False):
    x = torch.randn(B, T, D)
    if requires_grad:
        x.requires_grad_(True)
    return x


# ===========================================================================
# 1. BipartiteSoftMatching.match — merge_indices shape (B, r, 2)
# ===========================================================================
def test_match_indices_shape():
    x = _rand(B, T, D)
    bsm = BipartiteSoftMatching(R)
    merge_indices, unmerge_weights = bsm.match(x)
    assert merge_indices.shape == (B, R, 2), f"Expected ({B}, {R}, 2), got {merge_indices.shape}"


# ===========================================================================
# 2. BipartiteSoftMatching.match — valid index ranges
# ===========================================================================
def test_match_indices_valid_ranges():
    x = _rand(B, T, D)
    bsm = BipartiteSoftMatching(R)
    merge_indices, _ = bsm.match(x)
    T_a = math.ceil(T / 2)
    T_b = T // 2
    a_idx = merge_indices[:, :, 0]
    b_idx = merge_indices[:, :, 1]
    assert (a_idx >= 0).all() and (a_idx < T_a).all(), "A indices out of range"
    assert (b_idx >= 0).all() and (b_idx < T_b).all(), "B indices out of range"


# ===========================================================================
# 3. BipartiteSoftMatching.merge — output shape (B, T-r, D)
# ===========================================================================
def test_merge_output_shape():
    x = _rand(B, T, D)
    bsm = BipartiteSoftMatching(R)
    merge_indices, _ = bsm.match(x)
    merged = bsm.merge(x, merge_indices)
    assert merged.shape == (B, T - R, D), f"Expected ({B}, {T - R}, {D}), got {merged.shape}"


# ===========================================================================
# 4. BipartiteSoftMatching.merge — merged tokens are mean of their pair (r=1)
# ===========================================================================
def test_merge_is_mean_of_pair():
    """For r=1, the single merged token must equal (a + b) / 2."""
    B1, T1, D1, R1 = 1, 6, 8, 1

    # Create tokens where A[0] and B[0] are maximally similar (identical)
    x = torch.randn(B1, T1, D1)
    # Force A[0] == B[0] so they will be matched (highest cosine sim)
    x[0, 0] = x[0, 1]  # position 0 (A[0]) = position 1 (B[0])

    bsm = BipartiteSoftMatching(R1)
    merge_indices, _ = bsm.match(x)

    a_idx = merge_indices[0, 0, 0].item()
    b_idx = merge_indices[0, 0, 1].item()

    expected_merge = (x[0, a_idx * 2] + x[0, b_idx * 2 + 1]) * 0.5

    merged = bsm.merge(x, merge_indices)  # (1, T1-1, D1)

    # The merged value for the pair is stored at position a_idx in the merged seq
    actual_merge = merged[0, a_idx]

    assert torch.allclose(actual_merge, expected_merge, atol=1e-5), (
        f"Merged token != mean of pair.\n  expected: {expected_merge}\n  got: {actual_merge}"
    )


# ===========================================================================
# 5. TokenUnmerger.unmerge — output shape (B, original_T, D)
# ===========================================================================
def test_unmerger_output_shape():
    x = _rand(B, T, D)
    bsm = BipartiteSoftMatching(R)
    merge_indices, unmerge_weights = bsm.match(x)
    merged = bsm.merge(x, merge_indices)

    unmerger = TokenUnmerger()
    restored = unmerger.unmerge(merged, merge_indices, T, unmerge_weights)
    assert restored.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {restored.shape}"


# ===========================================================================
# 6. ToMeAttention.forward — output shape (B, T, D), compression_ratio < 1
# ===========================================================================
def test_tome_attention_output_shape_and_compression():
    x = _rand(B, T, D)
    attn = ToMeAttention(D, H, r=R)
    out, info = attn(x)
    assert out.shape == (B, T, D), f"Expected ({B},{T},{D}), got {out.shape}"
    assert info["compression_ratio"] < 1.0, (
        f"compression_ratio should be < 1 when r>0, got {info['compression_ratio']}"
    )


# ===========================================================================
# 7. ToMeAttention: r=0 → output same shape as standard attention (no merging)
# ===========================================================================
def test_tome_attention_r0_no_merging():
    """r=0 must pass through with compression_ratio == 1.0."""
    x = _rand(B, T, D)
    attn = ToMeAttention(D, H, r=0)
    out, info = attn(x)
    assert out.shape == (B, T, D)
    assert info["compression_ratio"] == 1.0
    assert info["original_T"] == info["merged_T"] == T


# ===========================================================================
# 8. ToMeTransformerBlock — output shape preserved (B, T, D), ratio in (0,1)
# ===========================================================================
def test_transformer_block_shape_and_ratio():
    x = _rand(B, T, D)
    block = ToMeTransformerBlock(D, H, r=R)
    out, cr = block(x)
    assert out.shape == (B, T, D), f"Expected ({B},{T},{D}), got {out.shape}"
    assert 0.0 < cr <= 1.0, f"compression_ratio should be in (0,1], got {cr}"


# ===========================================================================
# 9. ToMeModel — logits shape (B, T, V), mean_compression_ratio in (0,1)
# ===========================================================================
def test_model_logits_shape_and_ratio():
    ids = torch.randint(0, V, (B, T))
    model = ToMeModel(D, NL, H, V, r_per_layer=R)
    logits, mean_cr = model(ids)
    assert logits.shape == (B, T, V), f"Expected ({B},{T},{V}), got {logits.shape}"
    assert 0.0 < mean_cr <= 1.0, f"mean_compression_ratio out of range: {mean_cr}"


# ===========================================================================
# 10. ToMeModel — backward pass succeeds, gradients reach embedding
# ===========================================================================
def test_model_backward():
    ids = torch.randint(0, V, (B, T))
    model = ToMeModel(D, NL, H, V, r_per_layer=R)
    logits, _ = model(ids)
    loss = logits.mean()
    loss.backward()
    emb_grad = model.embedding.weight.grad
    assert emb_grad is not None, "Embedding gradient is None after backward"
    assert not torch.isnan(emb_grad).any(), "NaN in embedding gradient"


# ===========================================================================
# 11. ToMeModel.set_r — updates r in all blocks
# ===========================================================================
def test_set_r_updates_all_blocks():
    model = ToMeModel(D, NL, H, V, r_per_layer=R)
    new_r = 1
    model.set_r(new_r)
    for i, block in enumerate(model.blocks):
        assert block.attn.r == new_r, f"Block {i} has r={block.attn.r}, expected {new_r}"
    # Verify forward still works after changing r
    ids = torch.randint(0, V, (B, T))
    logits, _ = model(ids)
    assert logits.shape == (B, T, V)


# ===========================================================================
# 12. ToMeEfficiencyAnalyzer.theoretical_speedup — > 1.0 when r > 0
# ===========================================================================
def test_efficiency_speedup_positive():
    analyzer = ToMeEfficiencyAnalyzer()
    speedup = analyzer.theoretical_speedup(T=64, r_per_layer=4, n_layers=6)
    assert speedup > 1.0, f"Speedup should be > 1.0 when r>0, got {speedup}"


# ===========================================================================
# 13. ToMeEfficiencyAnalyzer.compression_per_layer — descending, final >= T-r*n
# ===========================================================================
def test_compression_per_layer_shape_and_monotone():
    analyzer = ToMeEfficiencyAnalyzer()
    T0 = 16
    r = 2
    n = 4
    lengths = analyzer.compression_per_layer(T=T0, r=r, n_layers=n)
    assert len(lengths) == n, f"Expected {n} lengths, got {len(lengths)}"
    # Non-increasing
    for i in range(1, len(lengths)):
        assert lengths[i] <= lengths[i - 1], (
            f"lengths[{i}]={lengths[i]} > lengths[{i - 1}]={lengths[i - 1]}"
        )
    # Final >= T - r*n_layers (could be clamped to 1)
    expected_min = max(1, T0 - r * n)
    assert lengths[-1] >= expected_min, (
        f"Final length {lengths[-1]} < expected minimum {expected_min}"
    )


# ===========================================================================
# 14. ToMeEfficiencyAnalyzer.flop_reduction — in (0, 1) for valid r
# ===========================================================================
def test_flop_reduction_range():
    analyzer = ToMeEfficiencyAnalyzer()
    reduction = analyzer.flop_reduction(T=32, r=4, d_model=64, n_heads=4)
    assert 0.0 < reduction < 1.0, f"flop_reduction should be in (0,1), got {reduction}"


# ===========================================================================
# 15. Higher r → lower compression_ratio (more merging = shorter sequences)
# ===========================================================================
def test_higher_r_lower_compression():
    """A model with larger r should produce a lower (or equal) compression ratio."""
    ids = torch.randint(0, V, (B, T))

    model_small_r = ToMeModel(D, NL, H, V, r_per_layer=1)
    model_large_r = ToMeModel(D, NL, H, V, r_per_layer=3)

    with torch.no_grad():
        _, cr_small = model_small_r(ids)
        _, cr_large = model_large_r(ids)

    assert cr_large <= cr_small, (
        f"Larger r should give lower (or equal) compression_ratio. "
        f"r=1 → {cr_small:.4f}, r=3 → {cr_large:.4f}"
    )


# ===========================================================================
# 16 (bonus). ToMe with same weights: larger r gives different output
# ===========================================================================
def test_larger_r_different_output():
    """Same weights, different r values should produce different logit outputs."""
    ids = torch.randint(0, V, (B, T))

    model = ToMeModel(D, NL, H, V, r_per_layer=1)

    with torch.no_grad():
        logits_r1, _ = model(ids)

    model.set_r(3)
    with torch.no_grad():
        logits_r3, _ = model(ids)

    # They should not be identical (merging different tokens changes the output)
    assert not torch.allclose(logits_r1, logits_r3, atol=1e-6), (
        "Logits with r=1 and r=3 should differ"
    )
