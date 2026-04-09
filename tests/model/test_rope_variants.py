"""Tests for rope_variants.py: ALiBi, T5-relative bias, dynamic NTK scaling, position interpolation."""

import pytest
import torch

from src.model.rope_variants import (
    ALiBiAttention,
    PositionConfig,
    T5RelativeAttentionBias,
    build_alibi_slopes,
    compute_alibi_bias,
    dynamic_ntk_scaling,
    interpolate_rope_positions,
    t5_relative_position_bucket,
)

torch.manual_seed(0)

# Common test dimensions
D_MODEL = 64
N_HEADS = 4
T = 16
B = 2


# ---------------------------------------------------------------------------
# 1. PositionConfig defaults
# ---------------------------------------------------------------------------

def test_position_config_defaults():
    cfg = PositionConfig()
    assert cfg.max_seq_len == 4096
    assert cfg.n_heads == 8
    assert cfg.alibi_bias_max == 8.0
    assert cfg.t5_num_buckets == 32
    assert cfg.t5_max_distance == 128
    assert cfg.rope_base == 10000.0
    assert cfg.dynamic_scale_factor == 1.0


# ---------------------------------------------------------------------------
# 2. build_alibi_slopes — shape
# ---------------------------------------------------------------------------

def test_build_alibi_slopes_shape():
    slopes = build_alibi_slopes(N_HEADS)
    assert slopes.shape == (N_HEADS,), f"Expected ({N_HEADS},), got {slopes.shape}"


# ---------------------------------------------------------------------------
# 3. build_alibi_slopes — slopes decrease with head index
# ---------------------------------------------------------------------------

def test_build_alibi_slopes_decreasing():
    slopes = build_alibi_slopes(N_HEADS)
    # slope[h] = 2^(-8 * (h+1) / N_HEADS); larger h → smaller slope
    for i in range(len(slopes) - 1):
        assert slopes[i] > slopes[i + 1], (
            f"Slope at {i} ({slopes[i]:.6f}) should be > slope at {i+1} ({slopes[i+1]:.6f})"
        )


# ---------------------------------------------------------------------------
# 4. compute_alibi_bias — shape
# ---------------------------------------------------------------------------

def test_compute_alibi_bias_shape():
    slopes = build_alibi_slopes(N_HEADS)
    bias = compute_alibi_bias(T, N_HEADS, slopes)
    # Accepts (1, n_heads, T, T) or (n_heads, T, T)
    assert bias.shape in {(1, N_HEADS, T, T), (N_HEADS, T, T)}, (
        f"Unexpected shape {bias.shape}"
    )


# ---------------------------------------------------------------------------
# 5. ALiBiAttention — output shape
# ---------------------------------------------------------------------------

def test_alibi_attention_output_shape():
    torch.manual_seed(0)
    cfg = PositionConfig(n_heads=N_HEADS)
    model = ALiBiAttention(D_MODEL, N_HEADS, cfg)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# 6. ALiBiAttention — gradient flow
# ---------------------------------------------------------------------------

def test_alibi_attention_gradient_flow():
    torch.manual_seed(0)
    cfg = PositionConfig(n_heads=N_HEADS)
    model = ALiBiAttention(D_MODEL, N_HEADS, cfg)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient did not flow back to input"
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# 7. t5_relative_position_bucket — shape
# ---------------------------------------------------------------------------

def test_t5_relative_position_bucket_shape():
    T_q, T_k = 8, 12
    positions = torch.arange(T_q).unsqueeze(1) - torch.arange(T_k).unsqueeze(0)  # (T_q, T_k)
    buckets = t5_relative_position_bucket(positions, num_buckets=32, max_distance=128)
    assert buckets.shape == (T_q, T_k), f"Expected ({T_q}, {T_k}), got {buckets.shape}"


# ---------------------------------------------------------------------------
# 8. t5_relative_position_bucket — values in [0, num_buckets)
# ---------------------------------------------------------------------------

def test_t5_relative_position_bucket_range():
    num_buckets = 32
    T_q, T_k = 16, 16
    positions = torch.arange(T_q).unsqueeze(1) - torch.arange(T_k).unsqueeze(0)
    buckets = t5_relative_position_bucket(positions, num_buckets=num_buckets, max_distance=128)
    assert buckets.min().item() >= 0, "Bucket values must be >= 0"
    assert buckets.max().item() < num_buckets, (
        f"Bucket values must be < {num_buckets}, got {buckets.max().item()}"
    )


# ---------------------------------------------------------------------------
# 9. T5RelativeAttentionBias — output shape
# ---------------------------------------------------------------------------

def test_t5_relative_attention_bias_shape():
    cfg = PositionConfig(n_heads=N_HEADS, t5_num_buckets=32, t5_max_distance=128)
    module = T5RelativeAttentionBias(cfg)
    T_q, T_k = 8, 12
    bias = module(T_q, T_k)
    assert bias.shape == (1, N_HEADS, T_q, T_k), (
        f"Expected (1, {N_HEADS}, {T_q}, {T_k}), got {bias.shape}"
    )


# ---------------------------------------------------------------------------
# 10. dynamic_ntk_scaling — shape
# ---------------------------------------------------------------------------

def test_dynamic_ntk_scaling_shape():
    freqs = dynamic_ntk_scaling(seq_len=T, base=10000.0, d_model=D_MODEL, scale_factor=1.0)
    assert freqs.shape == (D_MODEL // 2,), (
        f"Expected ({D_MODEL // 2},), got {freqs.shape}"
    )


# ---------------------------------------------------------------------------
# 11. interpolate_rope_positions — values are positions / scale
# ---------------------------------------------------------------------------

def test_interpolate_rope_positions_scaled():
    positions = torch.arange(T).float()
    scale = 2.0
    scaled = interpolate_rope_positions(positions, scale)
    expected = positions / scale
    assert scaled.shape == (T,), f"Expected ({T},), got {scaled.shape}"
    assert torch.allclose(scaled, expected), "Scaled positions do not match positions / scale"


# ---------------------------------------------------------------------------
# 12. ALiBiAttention works without any positional embeddings
# ---------------------------------------------------------------------------

def test_alibi_no_positional_embedding_needed():
    """ALiBiAttention should produce valid output with raw token embeddings (no pos embed added)."""
    torch.manual_seed(0)
    cfg = PositionConfig(n_heads=N_HEADS)
    model = ALiBiAttention(D_MODEL, N_HEADS, cfg)

    # Simulate raw token embeddings — no positional embedding added
    token_embeddings = torch.randn(B, T, D_MODEL)
    out = model(token_embeddings)

    # Output must be finite and correct shape
    assert out.shape == (B, T, D_MODEL)
    assert torch.isfinite(out).all(), "ALiBiAttention output contains non-finite values"
