"""Tests for rope_variants.py: ALiBi, T5-relative bias, dynamic NTK scaling, position interpolation,
plus new variants: RoPEVariantConfig, alibi_bias, fire_position_encoding, cope_gate, RoPEVariantLayer."""

import pytest
import torch

from src.model.rope_variants import (
    ALiBiAttention,
    CoPEGate,
    FirePositionEncoding,
    PositionConfig,
    RoPEVariantConfig,
    RoPEVariantLayer,
    T5RelativeAttentionBias,
    alibi_bias,
    build_alibi_slopes,
    compute_alibi_bias,
    cope_gate,
    dynamic_ntk_scaling,
    fire_position_encoding,
    interpolate_rope_positions,
    t5_relative_position_bucket,
)

torch.manual_seed(0)

# Common test dimensions
D_MODEL = 64
N_HEADS = 4
T = 16
B = 2
HEAD_DIM = D_MODEL // N_HEADS  # 16


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
# 2. build_alibi_slopes -- shape
# ---------------------------------------------------------------------------

def test_build_alibi_slopes_shape():
    slopes = build_alibi_slopes(N_HEADS)
    assert slopes.shape == (N_HEADS,), f"Expected ({N_HEADS},), got {slopes.shape}"


# ---------------------------------------------------------------------------
# 3. build_alibi_slopes -- slopes decrease with head index
# ---------------------------------------------------------------------------

def test_build_alibi_slopes_decreasing():
    slopes = build_alibi_slopes(N_HEADS)
    for i in range(len(slopes) - 1):
        assert slopes[i] > slopes[i + 1], (
            f"Slope at {i} ({slopes[i]:.6f}) should be > slope at {i+1} ({slopes[i+1]:.6f})"
        )


# ---------------------------------------------------------------------------
# 4. compute_alibi_bias -- shape
# ---------------------------------------------------------------------------

def test_compute_alibi_bias_shape():
    slopes = build_alibi_slopes(N_HEADS)
    bias = compute_alibi_bias(T, N_HEADS, slopes)
    assert bias.shape in {(1, N_HEADS, T, T), (N_HEADS, T, T)}, (
        f"Unexpected shape {bias.shape}"
    )


# ---------------------------------------------------------------------------
# 5. ALiBiAttention -- output shape
# ---------------------------------------------------------------------------

def test_alibi_attention_output_shape():
    torch.manual_seed(0)
    cfg = PositionConfig(n_heads=N_HEADS)
    model = ALiBiAttention(D_MODEL, N_HEADS, cfg)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# 6. ALiBiAttention -- gradient flow
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
# 7. t5_relative_position_bucket -- shape
# ---------------------------------------------------------------------------

def test_t5_relative_position_bucket_shape():
    T_q, T_k = 8, 12
    positions = torch.arange(T_q).unsqueeze(1) - torch.arange(T_k).unsqueeze(0)
    buckets = t5_relative_position_bucket(positions, num_buckets=32, max_distance=128)
    assert buckets.shape == (T_q, T_k), f"Expected ({T_q}, {T_k}), got {buckets.shape}"


# ---------------------------------------------------------------------------
# 8. t5_relative_position_bucket -- values in [0, num_buckets)
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
# 9. T5RelativeAttentionBias -- output shape
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
# 10. dynamic_ntk_scaling -- shape
# ---------------------------------------------------------------------------

def test_dynamic_ntk_scaling_shape():
    freqs = dynamic_ntk_scaling(seq_len=T, base=10000.0, d_model=D_MODEL, scale_factor=1.0)
    assert freqs.shape == (D_MODEL // 2,), (
        f"Expected ({D_MODEL // 2},), got {freqs.shape}"
    )


# ---------------------------------------------------------------------------
# 11. interpolate_rope_positions -- values are positions / scale
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
    torch.manual_seed(0)
    cfg = PositionConfig(n_heads=N_HEADS)
    model = ALiBiAttention(D_MODEL, N_HEADS, cfg)
    token_embeddings = torch.randn(B, T, D_MODEL)
    out = model(token_embeddings)
    assert out.shape == (B, T, D_MODEL)
    assert torch.isfinite(out).all(), "ALiBiAttention output contains non-finite values"


# ===========================================================================
# NEW TESTS: RoPEVariantConfig, alibi_bias, fire_position_encoding,
#            cope_gate, RoPEVariantLayer
# ===========================================================================


# ---------------------------------------------------------------------------
# 13. RoPEVariantConfig defaults
# ---------------------------------------------------------------------------

def test_rope_variant_config_defaults():
    cfg = RoPEVariantConfig()
    assert cfg.variant == "alibi"
    assert cfg.n_heads == 2
    assert cfg.max_seq_len == 512
    assert cfg.d_model == 64


# ---------------------------------------------------------------------------
# 14. RoPEVariantConfig custom values
# ---------------------------------------------------------------------------

def test_rope_variant_config_custom():
    cfg = RoPEVariantConfig(variant="fire", n_heads=8, max_seq_len=1024, d_model=128)
    assert cfg.variant == "fire"
    assert cfg.n_heads == 8
    assert cfg.max_seq_len == 1024
    assert cfg.d_model == 128


# ---------------------------------------------------------------------------
# 15. build_alibi_slopes -- known values for 2 heads
# ---------------------------------------------------------------------------

def test_build_alibi_slopes_known_values():
    slopes = build_alibi_slopes(2)
    # slope[0] = 2^(-8/2 * 1) = 2^(-4) = 0.0625
    # slope[1] = 2^(-8/2 * 2) = 2^(-8) = 0.00390625
    assert torch.allclose(slopes, torch.tensor([0.0625, 0.00390625]), atol=1e-7)


# ---------------------------------------------------------------------------
# 16. alibi_bias -- output shape
# ---------------------------------------------------------------------------

def test_alibi_bias_shape():
    slopes = build_alibi_slopes(N_HEADS)
    bias = alibi_bias(T, slopes)
    assert bias.shape == (N_HEADS, T, T), f"Expected ({N_HEADS}, {T}, {T}), got {bias.shape}"


# ---------------------------------------------------------------------------
# 17. alibi_bias -- diagonal is zero
# ---------------------------------------------------------------------------

def test_alibi_bias_diagonal_zero():
    slopes = build_alibi_slopes(N_HEADS)
    bias = alibi_bias(T, slopes)
    for h in range(N_HEADS):
        diag = torch.diagonal(bias[h])
        assert torch.allclose(diag, torch.zeros_like(diag)), (
            f"Diagonal for head {h} should be all zeros"
        )


# ---------------------------------------------------------------------------
# 18. alibi_bias -- symmetry (|i-j| is symmetric)
# ---------------------------------------------------------------------------

def test_alibi_bias_symmetric():
    slopes = build_alibi_slopes(N_HEADS)
    bias = alibi_bias(T, slopes)
    for h in range(N_HEADS):
        assert torch.allclose(bias[h], bias[h].T, atol=1e-6), (
            f"ALiBi bias for head {h} should be symmetric"
        )


# ---------------------------------------------------------------------------
# 19. alibi_bias -- non-negative (slopes * |i-j| >= 0)
# ---------------------------------------------------------------------------

def test_alibi_bias_nonnegative():
    slopes = build_alibi_slopes(N_HEADS)
    bias = alibi_bias(T, slopes)
    assert (bias >= 0).all(), "ALiBi bias values should be non-negative"


# ---------------------------------------------------------------------------
# 20. fire_position_encoding -- output shape
# ---------------------------------------------------------------------------

def test_fire_position_encoding_shape():
    positions = torch.arange(T)
    out = fire_position_encoding(positions, d_model=D_MODEL)
    assert out.shape == (T, D_MODEL), f"Expected ({T}, {D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# 21. FirePositionEncoding module -- output shape and finite
# ---------------------------------------------------------------------------

def test_fire_module_output():
    torch.manual_seed(42)
    fire = FirePositionEncoding(d_model=D_MODEL, max_seq_len=512)
    positions = torch.arange(T)
    out = fire(positions)
    assert out.shape == (T, D_MODEL)
    assert torch.isfinite(out).all(), "FIRE output contains non-finite values"


# ---------------------------------------------------------------------------
# 22. FirePositionEncoding -- gradient flow
# ---------------------------------------------------------------------------

def test_fire_gradient_flow():
    torch.manual_seed(42)
    fire = FirePositionEncoding(d_model=D_MODEL, max_seq_len=512)
    positions = torch.arange(T)
    out = fire(positions)
    loss = out.sum()
    loss.backward()
    assert fire.log_freqs.grad is not None, "Gradient did not flow to log_freqs"
    assert fire.proj.weight.grad is not None, "Gradient did not flow to proj"


# ---------------------------------------------------------------------------
# 23. cope_gate -- output shape
# ---------------------------------------------------------------------------

def test_cope_gate_shape():
    q = torch.randn(B, N_HEADS, T, HEAD_DIM)
    k = torch.randn(B, N_HEADS, T, HEAD_DIM)
    out = cope_gate(q, k)
    assert out.shape == (B, N_HEADS, T, T), f"Expected ({B}, {N_HEADS}, {T}, {T}), got {out.shape}"


# ---------------------------------------------------------------------------
# 24. CoPEGate module -- output shape and finite
# ---------------------------------------------------------------------------

def test_cope_module_output():
    torch.manual_seed(42)
    cope = CoPEGate(head_dim=HEAD_DIM, max_seq_len=512)
    q = torch.randn(B, N_HEADS, T, HEAD_DIM)
    k = torch.randn(B, N_HEADS, T, HEAD_DIM)
    out = cope(q, k)
    assert out.shape == (B, N_HEADS, T, T)
    assert torch.isfinite(out).all(), "CoPE output contains non-finite values"


# ---------------------------------------------------------------------------
# 25. CoPEGate -- gradient flow
# ---------------------------------------------------------------------------

def test_cope_gradient_flow():
    torch.manual_seed(42)
    cope = CoPEGate(head_dim=HEAD_DIM, max_seq_len=512)
    q = torch.randn(B, N_HEADS, T, HEAD_DIM, requires_grad=True)
    k = torch.randn(B, N_HEADS, T, HEAD_DIM, requires_grad=True)
    out = cope(q, k)
    loss = out.sum()
    loss.backward()
    assert q.grad is not None, "Gradient did not flow to query"
    assert k.grad is not None, "Gradient did not flow to key"


# ---------------------------------------------------------------------------
# 26. RoPEVariantLayer alibi -- output shape
# ---------------------------------------------------------------------------

def test_rope_variant_layer_alibi_shape():
    torch.manual_seed(0)
    cfg = RoPEVariantConfig(variant="alibi", n_heads=N_HEADS, d_model=D_MODEL)
    layer = RoPEVariantLayer(cfg)
    Q = torch.randn(B, N_HEADS, T, HEAD_DIM)
    K = torch.randn(B, N_HEADS, T, HEAD_DIM)
    V = torch.randn(B, N_HEADS, T, HEAD_DIM)
    out = layer(Q, K, V)
    assert out.shape == (B, N_HEADS, T, HEAD_DIM), f"Got {out.shape}"


# ---------------------------------------------------------------------------
# 27. RoPEVariantLayer fire -- output shape
# ---------------------------------------------------------------------------

def test_rope_variant_layer_fire_shape():
    torch.manual_seed(0)
    cfg = RoPEVariantConfig(variant="fire", n_heads=N_HEADS, d_model=D_MODEL)
    layer = RoPEVariantLayer(cfg)
    Q = torch.randn(B, N_HEADS, T, HEAD_DIM)
    K = torch.randn(B, N_HEADS, T, HEAD_DIM)
    V = torch.randn(B, N_HEADS, T, HEAD_DIM)
    out = layer(Q, K, V)
    assert out.shape == (B, N_HEADS, T, HEAD_DIM), f"Got {out.shape}"


# ---------------------------------------------------------------------------
# 28. RoPEVariantLayer cope -- output shape
# ---------------------------------------------------------------------------

def test_rope_variant_layer_cope_shape():
    torch.manual_seed(0)
    cfg = RoPEVariantConfig(variant="cope", n_heads=N_HEADS, d_model=D_MODEL)
    layer = RoPEVariantLayer(cfg)
    Q = torch.randn(B, N_HEADS, T, HEAD_DIM)
    K = torch.randn(B, N_HEADS, T, HEAD_DIM)
    V = torch.randn(B, N_HEADS, T, HEAD_DIM)
    out = layer(Q, K, V)
    assert out.shape == (B, N_HEADS, T, HEAD_DIM), f"Got {out.shape}"


# ---------------------------------------------------------------------------
# 29. RoPEVariantLayer -- invalid variant raises ValueError
# ---------------------------------------------------------------------------

def test_rope_variant_layer_invalid_variant():
    cfg = RoPEVariantConfig(variant="unknown", n_heads=N_HEADS, d_model=D_MODEL)
    with pytest.raises(ValueError, match="Unknown variant"):
        RoPEVariantLayer(cfg)


# ---------------------------------------------------------------------------
# 30. RoPEVariantLayer alibi -- gradient flow
# ---------------------------------------------------------------------------

def test_rope_variant_layer_alibi_gradient():
    torch.manual_seed(0)
    cfg = RoPEVariantConfig(variant="alibi", n_heads=N_HEADS, d_model=D_MODEL)
    layer = RoPEVariantLayer(cfg)
    Q = torch.randn(B, N_HEADS, T, HEAD_DIM, requires_grad=True)
    K = torch.randn(B, N_HEADS, T, HEAD_DIM, requires_grad=True)
    V = torch.randn(B, N_HEADS, T, HEAD_DIM, requires_grad=True)
    out = layer(Q, K, V)
    out.sum().backward()
    assert Q.grad is not None
    assert K.grad is not None
    assert V.grad is not None


# ---------------------------------------------------------------------------
# 31. RoPEVariantLayer fire -- gradient flow
# ---------------------------------------------------------------------------

def test_rope_variant_layer_fire_gradient():
    torch.manual_seed(0)
    cfg = RoPEVariantConfig(variant="fire", n_heads=N_HEADS, d_model=D_MODEL)
    layer = RoPEVariantLayer(cfg)
    Q = torch.randn(B, N_HEADS, T, HEAD_DIM, requires_grad=True)
    K = torch.randn(B, N_HEADS, T, HEAD_DIM)
    V = torch.randn(B, N_HEADS, T, HEAD_DIM)
    out = layer(Q, K, V)
    out.sum().backward()
    assert Q.grad is not None


# ---------------------------------------------------------------------------
# 32. RoPEVariantLayer -- output is finite for all variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["alibi", "fire", "cope"])
def test_rope_variant_layer_finite_output(variant):
    torch.manual_seed(0)
    cfg = RoPEVariantConfig(variant=variant, n_heads=N_HEADS, d_model=D_MODEL)
    layer = RoPEVariantLayer(cfg)
    Q = torch.randn(B, N_HEADS, T, HEAD_DIM)
    K = torch.randn(B, N_HEADS, T, HEAD_DIM)
    V = torch.randn(B, N_HEADS, T, HEAD_DIM)
    out = layer(Q, K, V)
    assert torch.isfinite(out).all(), f"Non-finite output for variant {variant}"
