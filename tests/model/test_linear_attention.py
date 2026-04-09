"""Tests for src/model/linear_attention.py.

Covers:
 1. test_config_defaults             — default field values
 2. test_elu_feature_map_positive    — ELU+1 always > 0
 3. test_relu_feature_map_nonneg     — ReLU+ε always >= 0
 4. test_random_fourier_features_shape — output has n_features in last dim
 5. test_linear_attention_causal_shape  — (B, H, T, D) output
 6. test_linear_attention_noncausal_shape — (B, H, T, D) output
 7. test_causal_lower_triangular     — zero keys after T//2 leaves first half unchanged
 8. test_noncausal_symmetric         — swapping Q and K gives different result (non-trivial)
 9. test_linear_attn_module_shape    — LinearAttention forward → (B, T, D)
10. test_linear_attn_elu_map         — works with feature_map="elu"
11. test_linear_attn_relu_map        — works with feature_map="relu"
12. test_linear_attn_rff_map         — works with feature_map="random_fourier"
13. test_compute_complexity_keys     — dict has "linear_ops" and "standard_ops"
14. test_compute_complexity_ordering — for large T, linear_ops < standard_ops
15. test_linear_attention_block_shape — (B, T, D) output
"""

import pytest
import torch

from src.model.linear_attention import (
    LinearAttnConfig,
    LinearAttention,
    LinearAttentionBlock,
    compute_linear_attention_complexity,
    elu_feature_map,
    linear_attention_causal,
    linear_attention_noncausal,
    random_fourier_features,
    relu_feature_map,
)


# ---------------------------------------------------------------------------
# Shared test dimensions
# ---------------------------------------------------------------------------

B, H, T, D = 2, 2, 8, 16   # head_dim = D = 16

# Small config used for module-level tests
SMALL_CFG = LinearAttnConfig(d_model=32, n_heads=2, head_dim=16, n_features=8)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = LinearAttnConfig()
    assert cfg.d_model == 64
    assert cfg.feature_map == "elu"
    assert cfg.causal is True


# ---------------------------------------------------------------------------
# 2. test_elu_feature_map_positive
# ---------------------------------------------------------------------------

def test_elu_feature_map_positive():
    x = torch.randn(B, H, T, D) * 5.0   # wide range of values
    out = elu_feature_map(x)
    assert out.shape == x.shape
    assert (out > 0).all(), "ELU+1 feature map must be strictly positive"


# ---------------------------------------------------------------------------
# 3. test_relu_feature_map_nonneg
# ---------------------------------------------------------------------------

def test_relu_feature_map_nonneg():
    x = torch.randn(B, H, T, D) * 5.0
    out = relu_feature_map(x)
    assert out.shape == x.shape
    assert (out >= 0).all(), "ReLU feature map must be non-negative"


# ---------------------------------------------------------------------------
# 4. test_random_fourier_features_shape
# ---------------------------------------------------------------------------

def test_random_fourier_features_shape():
    n_features = 32
    x = torch.randn(B, H, T, D)
    omega = torch.randn(D, n_features)
    bias = torch.rand(n_features) * 2 * 3.14159
    out = random_fourier_features(x, omega, bias)
    assert out.shape == (B, H, T, n_features)


# ---------------------------------------------------------------------------
# 5. test_linear_attention_causal_shape
# ---------------------------------------------------------------------------

def test_linear_attention_causal_shape():
    q = elu_feature_map(torch.randn(B, H, T, D))
    k = elu_feature_map(torch.randn(B, H, T, D))
    v = torch.randn(B, H, T, D)
    out = linear_attention_causal(q, k, v)
    assert out.shape == (B, H, T, D)


# ---------------------------------------------------------------------------
# 6. test_linear_attention_noncausal_shape
# ---------------------------------------------------------------------------

def test_linear_attention_noncausal_shape():
    q = elu_feature_map(torch.randn(B, H, T, D))
    k = elu_feature_map(torch.randn(B, H, T, D))
    v = torch.randn(B, H, T, D)
    out = linear_attention_noncausal(q, k, v)
    assert out.shape == (B, H, T, D)


# ---------------------------------------------------------------------------
# 7. test_causal_lower_triangular
# ---------------------------------------------------------------------------

def test_causal_lower_triangular():
    """Zero keys/values after T//2; output for t < T//2 must be identical."""
    torch.manual_seed(42)
    q = elu_feature_map(torch.randn(B, H, T, D))
    k = elu_feature_map(torch.randn(B, H, T, D))
    v = torch.randn(B, H, T, D)

    half = T // 2

    out_full = linear_attention_causal(q, k, v)

    # Zero out keys and values after the midpoint
    k_masked = k.clone()
    v_masked = v.clone()
    k_masked[:, :, half:, :] = 0.0
    v_masked[:, :, half:, :] = 0.0

    out_masked = linear_attention_causal(q, k_masked, v_masked)

    # For positions before the midpoint the two outputs must match exactly,
    # because those positions haven't seen any of the zeroed keys yet.
    assert torch.allclose(out_full[:, :, :half, :], out_masked[:, :, :half, :], atol=1e-5), (
        "Causal attention: output for t < T//2 should be unchanged when "
        "keys/values after T//2 are zeroed out."
    )


# ---------------------------------------------------------------------------
# 8. test_noncausal_symmetric
# ---------------------------------------------------------------------------

def test_noncausal_symmetric():
    """Swapping Q and K should generally produce a different output (Q/K are not symmetric)."""
    torch.manual_seed(0)
    q = elu_feature_map(torch.randn(B, H, T, D))
    k = elu_feature_map(torch.randn(B, H, T, D))
    v = torch.randn(B, H, T, D)

    out_qk = linear_attention_noncausal(q, k, v)
    out_kq = linear_attention_noncausal(k, q, v)   # swapped

    # They should differ (non-trivial test that Q and K play different roles)
    assert not torch.allclose(out_qk, out_kq, atol=1e-3), (
        "Swapping Q and K in non-causal linear attention should change the output."
    )


# ---------------------------------------------------------------------------
# 9. test_linear_attn_module_shape
# ---------------------------------------------------------------------------

def test_linear_attn_module_shape():
    cfg = SMALL_CFG
    model = LinearAttention(cfg)
    x = torch.randn(B, T, cfg.d_model)
    out = model(x)
    assert out.shape == (B, T, cfg.d_model)


# ---------------------------------------------------------------------------
# 10. test_linear_attn_elu_map
# ---------------------------------------------------------------------------

def test_linear_attn_elu_map():
    cfg = LinearAttnConfig(d_model=32, n_heads=2, head_dim=16, feature_map="elu")
    model = LinearAttention(cfg)
    x = torch.randn(B, T, cfg.d_model)
    out = model(x)
    assert out.shape == (B, T, cfg.d_model)


# ---------------------------------------------------------------------------
# 11. test_linear_attn_relu_map
# ---------------------------------------------------------------------------

def test_linear_attn_relu_map():
    cfg = LinearAttnConfig(d_model=32, n_heads=2, head_dim=16, feature_map="relu")
    model = LinearAttention(cfg)
    x = torch.randn(B, T, cfg.d_model)
    out = model(x)
    assert out.shape == (B, T, cfg.d_model)


# ---------------------------------------------------------------------------
# 12. test_linear_attn_rff_map
# ---------------------------------------------------------------------------

def test_linear_attn_rff_map():
    cfg = LinearAttnConfig(d_model=32, n_heads=2, head_dim=16, feature_map="random_fourier")
    model = LinearAttention(cfg)
    x = torch.randn(B, T, cfg.d_model)
    out = model(x)
    assert out.shape == (B, T, cfg.d_model)


# ---------------------------------------------------------------------------
# 13. test_compute_complexity_keys
# ---------------------------------------------------------------------------

def test_compute_complexity_keys():
    result = compute_linear_attention_complexity(T=128, D=64, H=8)
    assert "linear_ops" in result
    assert "standard_ops" in result


# ---------------------------------------------------------------------------
# 14. test_compute_complexity_ordering
# ---------------------------------------------------------------------------

def test_compute_complexity_ordering():
    """For T >> D, linear attention should be cheaper than standard."""
    result = compute_linear_attention_complexity(T=4096, D=64, H=8)
    assert result["linear_ops"] < result["standard_ops"], (
        f"Expected linear_ops ({result['linear_ops']}) < "
        f"standard_ops ({result['standard_ops']}) for large T"
    )


# ---------------------------------------------------------------------------
# 15. test_linear_attention_block_shape
# ---------------------------------------------------------------------------

def test_linear_attention_block_shape():
    cfg = SMALL_CFG
    block = LinearAttentionBlock(cfg)
    x = torch.randn(B, T, cfg.d_model)
    out = block(x)
    assert out.shape == (B, T, cfg.d_model)
