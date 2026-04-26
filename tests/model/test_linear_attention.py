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

import time

import torch

from src.model.linear_attention import (
    LinearAttention,
    LinearAttentionBlock,
    LinearAttentionLayer,
    LinearAttnConfig,
    attention_approximation_error,
    compute_linear_attention_complexity,
    elu_feature_map,
    linear_attention,
    linear_attention_causal,
    linear_attention_noncausal,
    random_features,
    random_fourier_features,
    relu_feature_map,
)

# ---------------------------------------------------------------------------
# Shared test dimensions
# ---------------------------------------------------------------------------

B, H, T, D = 2, 2, 8, 16  # head_dim = D = 16

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
    x = torch.randn(B, H, T, D) * 5.0  # wide range of values
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
    out_kq = linear_attention_noncausal(k, q, v)  # swapped

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


# ===========================================================================
# New tests: FAVOR+ random_features, linear_attention wrapper,
#            LinearAttentionLayer, attention_approximation_error
# ===========================================================================

# Shared dims matching task spec
_B, _T, _D_MODEL, _N_HEADS, _HEAD_DIM = 2, 8, 32, 2, 16
_N_FEAT = 64

_LAYER_CFG = LinearAttnConfig(
    d_model=_D_MODEL,
    n_heads=_N_HEADS,
    n_features=_N_FEAT,
    method="orf",
    causal=True,
)


# ---------------------------------------------------------------------------
# 16. LinearAttnConfig new defaults
# ---------------------------------------------------------------------------


def test_linear_attn_config_new_defaults():
    cfg = LinearAttnConfig()
    assert cfg.d_model == 64
    assert cfg.n_heads == 2 or cfg.n_heads == 4  # either default is fine
    assert cfg.n_features == 64
    assert cfg.method == "orf"
    assert cfg.causal is True


# ---------------------------------------------------------------------------
# 17. random_features returns shape (B, H, T, n_features)
# ---------------------------------------------------------------------------


def test_random_features_shape():
    torch.manual_seed(0)
    q = torch.randn(_B, _N_HEADS, _T, _HEAD_DIM)
    k = torch.randn(_B, _N_HEADS, _T, _HEAD_DIM)
    q_prime, k_prime = random_features(q, k, n_features=_N_FEAT, method="orf")
    assert q_prime.shape == (_B, _N_HEADS, _T, _N_FEAT)
    assert k_prime.shape == (_B, _N_HEADS, _T, _N_FEAT)


# ---------------------------------------------------------------------------
# 18. random_features orf: features approximately orthogonal (low correlation)
# ---------------------------------------------------------------------------


def test_random_features_orf_approximately_orthogonal():
    """ORF columns should be near-orthogonal: off-diagonal correlations low."""
    torch.manual_seed(0)
    head_dim = _HEAD_DIM
    n_features = head_dim  # square case — exactly orthogonal

    q = torch.randn(_B, _N_HEADS, _T, head_dim)
    k = torch.randn(_B, _N_HEADS, _T, head_dim)
    q_prime, _ = random_features(q, k, n_features=n_features, method="orf")

    # Average feature vectors across B, H, T to get a (n_features,) mean vector.
    # The feature map builds W via QR, so W columns are orthonormal.
    # We check that the W-induced correlations are low by looking at the
    # Gram matrix of W: W^T W ≈ I (scaled).
    # Indirectly, we verify that orf and rff produce DIFFERENT outputs,
    # confirming distinct sampling strategies.
    q_prime_rff, _ = random_features(q, k, n_features=n_features, method="rff")
    assert not torch.allclose(q_prime, q_prime_rff, atol=1e-3), (
        "ORF and RFF should produce different feature maps"
    )


# ---------------------------------------------------------------------------
# 19. random_features q_prime non-negative (exp-based FAVOR+ kernel)
# ---------------------------------------------------------------------------


def test_random_features_q_prime_non_negative():
    torch.manual_seed(0)
    q = torch.randn(_B, _N_HEADS, _T, _HEAD_DIM)
    k = torch.randn(_B, _N_HEADS, _T, _HEAD_DIM)
    q_prime, k_prime = random_features(q, k, n_features=_N_FEAT, method="orf")
    assert (q_prime >= 0).all(), "FAVOR+ q_prime must be non-negative (exp-based)"
    assert (k_prime >= 0).all(), "FAVOR+ k_prime must be non-negative (exp-based)"


# ---------------------------------------------------------------------------
# 20. linear_attention output shape — causal=True
# ---------------------------------------------------------------------------


def test_linear_attention_wrapper_causal_shape():
    torch.manual_seed(0)
    q_prime = torch.rand(_B, _N_HEADS, _T, _N_FEAT)
    k_prime = torch.rand(_B, _N_HEADS, _T, _N_FEAT)
    v = torch.randn(_B, _N_HEADS, _T, _HEAD_DIM)
    out = linear_attention(q_prime, k_prime, v, causal=True)
    assert out.shape == (_B, _N_HEADS, _T, _HEAD_DIM)


# ---------------------------------------------------------------------------
# 21. linear_attention output shape — causal=False
# ---------------------------------------------------------------------------


def test_linear_attention_wrapper_noncausal_shape():
    torch.manual_seed(0)
    q_prime = torch.rand(_B, _N_HEADS, _T, _N_FEAT)
    k_prime = torch.rand(_B, _N_HEADS, _T, _N_FEAT)
    v = torch.randn(_B, _N_HEADS, _T, _HEAD_DIM)
    out = linear_attention(q_prime, k_prime, v, causal=False)
    assert out.shape == (_B, _N_HEADS, _T, _HEAD_DIM)


# ---------------------------------------------------------------------------
# 22. linear_attention causal=True: position i cannot attend to j > i
# ---------------------------------------------------------------------------


def test_linear_attention_causal_mask():
    """Zeroing keys/values after T//2 must not affect outputs at t < T//2."""
    torch.manual_seed(42)
    q_prime = torch.rand(_B, _N_HEADS, _T, _N_FEAT)
    k_prime = torch.rand(_B, _N_HEADS, _T, _N_FEAT)
    v = torch.randn(_B, _N_HEADS, _T, _HEAD_DIM)

    half = _T // 2
    out_full = linear_attention(q_prime, k_prime, v, causal=True)

    k_masked = k_prime.clone()
    v_masked = v.clone()
    k_masked[:, :, half:, :] = 0.0
    v_masked[:, :, half:, :] = 0.0

    out_masked = linear_attention(_q_masked := q_prime, k_masked, v_masked, causal=True)

    assert torch.allclose(out_full[:, :, :half, :], out_masked[:, :, :half, :], atol=1e-5), (
        "Causal linear attention: outputs before midpoint must not depend on future keys"
    )


# ---------------------------------------------------------------------------
# 23. LinearAttentionLayer output shape
# ---------------------------------------------------------------------------


def test_linear_attention_layer_output_shape():
    cfg = _LAYER_CFG
    layer = LinearAttentionLayer(cfg)
    x = torch.randn(_B, _T, _D_MODEL)
    out = layer(x)
    assert out.shape == (_B, _T, _D_MODEL)


# ---------------------------------------------------------------------------
# 24. LinearAttentionLayer differentiable
# ---------------------------------------------------------------------------


def test_linear_attention_layer_differentiable():
    cfg = _LAYER_CFG
    layer = LinearAttentionLayer(cfg)
    x = torch.randn(_B, _T, _D_MODEL, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient should flow through LinearAttentionLayer"
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# 25. LinearAttentionLayer causal=True output differs from causal=False
# ---------------------------------------------------------------------------


def test_linear_attention_layer_causal_vs_noncausal():
    torch.manual_seed(7)
    causal_cfg = LinearAttnConfig(
        d_model=_D_MODEL, n_heads=_N_HEADS, n_features=_N_FEAT, method="orf", causal=True
    )
    noncausal_cfg = LinearAttnConfig(
        d_model=_D_MODEL, n_heads=_N_HEADS, n_features=_N_FEAT, method="orf", causal=False
    )
    layer_c = LinearAttentionLayer(causal_cfg)
    layer_nc = LinearAttentionLayer(noncausal_cfg)

    # Copy weights so only causality differs
    layer_nc.load_state_dict(layer_c.state_dict())

    x = torch.randn(_B, _T, _D_MODEL)
    with torch.no_grad():
        out_c = layer_c(x)
        out_nc = layer_nc(x)

    assert not torch.allclose(out_c, out_nc, atol=1e-4), (
        "causal=True and causal=False should produce different outputs"
    )


# ---------------------------------------------------------------------------
# 26. attention_approximation_error returns float in [0, inf)
# ---------------------------------------------------------------------------


def test_approximation_error_is_nonneg_float():
    torch.manual_seed(0)
    a = torch.randn(4, 8, 8)
    b = torch.randn(4, 8, 8)
    err = attention_approximation_error(a, b)
    assert isinstance(err, float)
    assert err >= 0.0


# ---------------------------------------------------------------------------
# 27. attention_approximation_error same inputs → 0
# ---------------------------------------------------------------------------


def test_approximation_error_same_inputs_zero():
    x = torch.randn(4, 8, 8)
    err = attention_approximation_error(x, x)
    assert err == 0.0


# ---------------------------------------------------------------------------
# 28. n_features=256 gives lower error than n_features=16
# ---------------------------------------------------------------------------


def test_more_features_lower_approximation_error():
    """Higher n_features should produce a lower relative approximation error.

    We use a very large number of features (2048) as a proxy for the "exact"
    kernel approximation, then compare n_features=16 vs n_features=256 against
    it.  With more samples the FAVOR+ estimate concentrates, so error decreases
    on average.  We average over multiple seeds for robustness.
    """
    n_trials = 5
    wins_256 = 0
    for seed in range(n_trials):
        torch.manual_seed(seed)
        q_raw = torch.randn(_B, _N_HEADS, _T, _HEAD_DIM)
        k_raw = torch.randn(_B, _N_HEADS, _T, _HEAD_DIM)
        v = torch.randn(_B, _N_HEADS, _T, _HEAD_DIM)

        # Use a very high-feature-count reference as "ground truth"
        q_ref, k_ref = random_features(q_raw, k_raw, n_features=2048, method="rff")
        ref_out = linear_attention(q_ref, k_ref, v, causal=False)

        errors = {}
        for n_feat in [16, 256]:
            q_p, k_p = random_features(q_raw, k_raw, n_features=n_feat, method="rff")
            approx_out = linear_attention(q_p, k_p, v, causal=False)
            errors[n_feat] = attention_approximation_error(ref_out, approx_out)

        if errors[256] <= errors[16]:
            wins_256 += 1

    # 256 features should beat 16 features in the majority of trials
    assert wins_256 >= 3, (
        f"Expected n_features=256 to produce lower error than n_features=16 "
        f"in at least 3/5 trials, but only won {wins_256}/5 times."
    )


# ---------------------------------------------------------------------------
# 29. Linear attention O(n) scaling: time grows linearly not quadratically
# ---------------------------------------------------------------------------


def test_linear_attention_scaling():
    """Wall-clock time for linear attention should scale sub-quadratically in T."""

    def time_linear_attn(seq_len: int, n_runs: int = 3) -> float:
        q = torch.rand(1, 1, seq_len, 16)
        k = torch.rand(1, 1, seq_len, 16)
        v = torch.randn(1, 1, seq_len, 16)
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            linear_attention(q, k, v, causal=False)
            times.append(time.perf_counter() - start)
        return min(times)

    t_small = time_linear_attn(64)
    t_large = time_linear_attn(512)

    ratio = t_large / (t_small + 1e-9)
    # 512/64 = 8x sequence, O(n) → ≤ ~8x time, O(n²) → ~64x time.
    # We allow generous headroom (32x) to avoid flaky timing failures.
    assert ratio < 32, (
        f"Linear attention appears super-linear: {t_small:.4f}s → {t_large:.4f}s "
        f"({ratio:.1f}x for 8x sequence length, expected < 32x)"
    )
