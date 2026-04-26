"""Tests for src/model/flash_sliding_window.py

14 tests covering SlidingWindowConfig, build_sliding_window_mask,
online_softmax, tiled_attention, SlidingWindowAttention,
GlobalLocalAttention, and compute_attention_flops.
"""

import math

import torch
import torch.nn.functional as F

from src.model.flash_sliding_window import (
    GlobalLocalAttention,
    SlidingWindowAttention,
    SlidingWindowConfig,
    build_sliding_window_mask,
    compute_attention_flops,
    online_softmax,
    tiled_attention,
)

# ---------------------------------------------------------------------------
# 1. SlidingWindowConfig defaults
# ---------------------------------------------------------------------------


def test_sliding_window_config_defaults():
    cfg = SlidingWindowConfig()
    assert cfg.window_size == 512
    assert cfg.global_tokens == 64
    assert cfg.causal is True
    assert cfg.block_size == 64


# ---------------------------------------------------------------------------
# 2. build_sliding_window_mask shape
# ---------------------------------------------------------------------------


def test_build_mask_shape():
    T = 16
    mask = build_sliding_window_mask(T, window_size=8, causal=True)
    assert mask.shape == (T, T), f"Expected ({T}, {T}), got {mask.shape}"
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 3. build_sliding_window_mask causal: upper triangle is False
# ---------------------------------------------------------------------------


def test_build_mask_causal_upper_triangle_false():
    T = 12
    mask = build_sliding_window_mask(T, window_size=T, causal=True)
    # Upper-strict-triangular positions (j > i) must all be False
    i_idx = torch.arange(T).unsqueeze(1)
    j_idx = torch.arange(T).unsqueeze(0)
    upper = j_idx > i_idx  # (T, T) bool, strict upper triangle
    assert not mask[upper].any(), "Causal mask has True values in upper triangle"


# ---------------------------------------------------------------------------
# 4. build_sliding_window_mask window_size=T: all causal positions True
# ---------------------------------------------------------------------------


def test_build_mask_full_window_all_causal_true():
    T = 10
    mask = build_sliding_window_mask(T, window_size=T, causal=True)
    # Every lower-triangular position (j <= i) should be True
    i_idx = torch.arange(T).unsqueeze(1)
    j_idx = torch.arange(T).unsqueeze(0)
    lower = j_idx <= i_idx
    assert mask[lower].all(), "Full-window causal mask has False in lower triangle"


# ---------------------------------------------------------------------------
# 5. build_sliding_window_mask: distant tokens are masked
# ---------------------------------------------------------------------------


def test_build_mask_distant_tokens_masked():
    T = 20
    W = 4
    mask = build_sliding_window_mask(T, window_size=W, causal=True)
    # Token at row 15 should NOT attend to token at col 0 (distance 15 >= W)
    assert not mask[15, 0].item(), "Token 15 should not attend to token 0 with window=4"
    # But it SHOULD attend to token 12 (distance 3 < 4)
    assert mask[15, 12].item(), "Token 15 should attend to token 12 with window=4"


# ---------------------------------------------------------------------------
# 6. online_softmax: output sums to 1 along last dim
# ---------------------------------------------------------------------------


def test_online_softmax_sums_to_one():
    torch.manual_seed(0)
    scores = torch.randn(2, 4, 8, 16)  # (B, H, T, S)
    probs = online_softmax(scores)
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"Softmax rows do not sum to 1; max deviation: {(row_sums - 1).abs().max()}"
    )


# ---------------------------------------------------------------------------
# 7. online_softmax: masked positions near zero
# ---------------------------------------------------------------------------


def test_online_softmax_masked_positions_near_zero():
    torch.manual_seed(1)
    S = 16
    scores = torch.randn(1, 1, 1, S)
    # Mask that allows only positions 0..7
    mask = torch.zeros(1, 1, 1, S, dtype=torch.bool)
    mask[..., :8] = True

    probs = online_softmax(scores, mask=mask)
    # Positions 8..15 should have probability ≈ 0
    masked_out_probs = probs[..., 8:]
    assert (masked_out_probs < 1e-6).all(), (
        f"Masked positions not near zero: {masked_out_probs.max()}"
    )
    # Remaining positions should sum to 1
    assert torch.allclose(probs[..., :8].sum(dim=-1), torch.ones(1, 1, 1), atol=1e-5)


# ---------------------------------------------------------------------------
# 8. tiled_attention output shape
# ---------------------------------------------------------------------------


def test_tiled_attention_output_shape():
    torch.manual_seed(2)
    B, H, T, D = 2, 4, 24, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    mask = build_sliding_window_mask(T, window_size=8, causal=True)
    out = tiled_attention(q, k, v, mask, block_size=8)
    assert out.shape == (B, H, T, D), f"Expected {(B, H, T, D)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 9. tiled_attention numerically close to standard attention (small sequences)
# ---------------------------------------------------------------------------


def test_tiled_attention_matches_standard():
    torch.manual_seed(3)
    B, H, T, D = 1, 2, 16, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    # Full causal mask (window_size = T)
    mask = build_sliding_window_mask(T, window_size=T, causal=True)  # (T, T)

    # Reference: standard scaled dot-product attention
    scale = 1.0 / math.sqrt(D)
    scores = scale * torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
    # Apply additive mask: forbidden positions → -1e9
    additive = torch.where(mask, torch.zeros(T, T), torch.full((T, T), -1e9))
    scores = scores + additive.unsqueeze(0).unsqueeze(0)
    ref = torch.matmul(F.softmax(scores, dim=-1), v)

    # Tiled attention
    tiled = tiled_attention(q, k, v, mask, block_size=8)

    assert torch.allclose(ref, tiled, atol=1e-4), f"Max diff: {(ref - tiled).abs().max().item()}"


# ---------------------------------------------------------------------------
# 10. SlidingWindowAttention output shape
# ---------------------------------------------------------------------------


def test_sliding_window_attention_output_shape():
    torch.manual_seed(4)
    B, T, d_model, n_heads = 2, 16, 32, 4
    cfg = SlidingWindowConfig(window_size=8, block_size=8)
    attn = SlidingWindowAttention(d_model, n_heads, cfg)
    x = torch.randn(B, T, d_model)
    out = attn(x)
    assert out.shape == (B, T, d_model), f"Expected {(B, T, d_model)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 11. SlidingWindowAttention causal vs non-causal produce different outputs
# ---------------------------------------------------------------------------


def test_sliding_window_attention_causal_vs_noncausal():
    torch.manual_seed(5)
    B, T, d_model, n_heads = 1, 12, 16, 2

    cfg_causal = SlidingWindowConfig(window_size=6, causal=True, block_size=6)
    cfg_noncausal = SlidingWindowConfig(window_size=6, causal=False, block_size=6)

    # Share weights for a fair comparison
    attn_causal = SlidingWindowAttention(d_model, n_heads, cfg_causal)
    attn_noncausal = SlidingWindowAttention(d_model, n_heads, cfg_noncausal)

    # Copy weights
    attn_noncausal.load_state_dict(attn_causal.state_dict())

    x = torch.randn(B, T, d_model)
    with torch.no_grad():
        out_causal = attn_causal(x)
        out_noncausal = attn_noncausal(x)

    assert not torch.allclose(out_causal, out_noncausal, atol=1e-4), (
        "Causal and non-causal outputs should differ"
    )


# ---------------------------------------------------------------------------
# 12. GlobalLocalAttention output shape
# ---------------------------------------------------------------------------


def test_global_local_attention_output_shape():
    torch.manual_seed(6)
    B, T, d_model, n_heads = 2, 20, 32, 4
    cfg = SlidingWindowConfig(window_size=6, global_tokens=4, block_size=8)
    gla = GlobalLocalAttention(d_model, n_heads, cfg)
    x = torch.randn(B, T, d_model)
    out = gla(x)
    assert out.shape == (B, T, d_model), f"Expected {(B, T, d_model)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 13. compute_attention_flops: sliding < full for long sequences
# ---------------------------------------------------------------------------


def test_compute_flops_sliding_less_than_full():
    result = compute_attention_flops(seq_len=1024, d_model=512, n_heads=8, window_size=64)
    assert result["sliding_window_flops"] < result["full_attention_flops"], (
        "Sliding-window FLOPs should be less than full-attention FLOPs"
    )


# ---------------------------------------------------------------------------
# 14. compute_attention_flops: speedup_ratio > 1
# ---------------------------------------------------------------------------


def test_compute_flops_speedup_ratio_greater_than_one():
    result = compute_attention_flops(seq_len=2048, d_model=256, n_heads=4, window_size=128)
    assert result["speedup_ratio"] > 1.0, (
        f"Expected speedup_ratio > 1, got {result['speedup_ratio']}"
    )
