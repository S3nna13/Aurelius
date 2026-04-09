"""Tests for src/model/long_rope.py — LongRoPE position encoding."""

import torch
import pytest

from src.model.long_rope import (
    LongRoPEConfig,
    compute_longrope_freqs,
    build_longrope_cos_sin,
    apply_longrope_rotation,
    LongRoPEAttention,
    extrapolation_quality,
)

# Common test dimensions (kept small for fast CPU tests)
HEAD_DIM = 32
D_MODEL = 64
N_HEADS = 2
MAX_TRAIN_LEN = 16
TARGET_LEN = 64


def make_config(**kwargs) -> LongRoPEConfig:
    defaults = dict(
        head_dim=HEAD_DIM,
        max_train_len=MAX_TRAIN_LEN,
        target_len=TARGET_LEN,
    )
    defaults.update(kwargs)
    return LongRoPEConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------

def test_longrope_config_defaults():
    cfg = LongRoPEConfig()
    assert cfg.head_dim == 32
    assert cfg.base_theta == 10000.0
    assert cfg.max_train_len == 4096
    assert cfg.target_len == 32768
    assert cfg.n_rescale_factors == 16
    assert cfg.lambda_min == 1.0
    assert cfg.lambda_max == 8.0


# ---------------------------------------------------------------------------
# 2. compute_longrope_freqs shape
# ---------------------------------------------------------------------------

def test_compute_longrope_freqs_shape():
    cfg = make_config()
    freqs = compute_longrope_freqs(cfg)
    assert freqs.shape == (HEAD_DIM // 2,), f"Expected ({HEAD_DIM // 2},), got {freqs.shape}"


# ---------------------------------------------------------------------------
# 3. compute_longrope_freqs all positive
# ---------------------------------------------------------------------------

def test_compute_longrope_freqs_positive():
    cfg = make_config()
    freqs = compute_longrope_freqs(cfg)
    assert (freqs > 0).all(), "All frequencies must be positive"


# ---------------------------------------------------------------------------
# 4. compute_longrope_freqs monotone (higher-lambda dims have lower freq)
#    Because lambda increases with dim index, effective freq = base_freq / lambda
#    decreases faster than base_freq alone -> freqs are strictly decreasing.
# ---------------------------------------------------------------------------

def test_compute_longrope_freqs_monotone():
    cfg = make_config()
    freqs = compute_longrope_freqs(cfg)
    # Effective freqs should be monotonically non-increasing
    diffs = freqs[1:] - freqs[:-1]
    assert (diffs <= 0).all(), (
        "Frequencies should be non-increasing (higher-lambda dims -> lower freq)"
    )


# ---------------------------------------------------------------------------
# 5. build_longrope_cos_sin shape
# ---------------------------------------------------------------------------

def test_build_longrope_cos_sin_shape():
    cfg = make_config()
    T = 8
    cos, sin = build_longrope_cos_sin(T, cfg)
    assert cos.shape == (T, HEAD_DIM), f"cos shape mismatch: {cos.shape}"
    assert sin.shape == (T, HEAD_DIM), f"sin shape mismatch: {sin.shape}"


# ---------------------------------------------------------------------------
# 6. build_longrope_cos_sin values in [-1, 1]
# ---------------------------------------------------------------------------

def test_build_longrope_cos_sin_range():
    cfg = make_config()
    T = 8
    cos, sin = build_longrope_cos_sin(T, cfg)
    assert cos.min() >= -1.0 - 1e-6 and cos.max() <= 1.0 + 1e-6, "cos out of [-1, 1]"
    assert sin.min() >= -1.0 - 1e-6 and sin.max() <= 1.0 + 1e-6, "sin out of [-1, 1]"


# ---------------------------------------------------------------------------
# 7. apply_longrope_rotation output shape
# ---------------------------------------------------------------------------

def test_apply_longrope_rotation_shape():
    torch.manual_seed(0)
    cfg = make_config()
    B, T = 1, 8
    x = torch.randn(B, N_HEADS, T, HEAD_DIM)
    cos, sin = build_longrope_cos_sin(T, cfg)
    out = apply_longrope_rotation(x, cos, sin)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"


# ---------------------------------------------------------------------------
# 8. apply_longrope_rotation L2 norm preserved (rotation is isometry)
# ---------------------------------------------------------------------------

def test_apply_longrope_rotation_norm_preserved():
    torch.manual_seed(0)
    cfg = make_config()
    B, T = 1, 8
    x = torch.randn(B, N_HEADS, T, HEAD_DIM)
    cos, sin = build_longrope_cos_sin(T, cfg)
    out = apply_longrope_rotation(x, cos, sin)

    # L2 norm along head_dim should be preserved for each (B, h, t) position
    norm_in = x.norm(dim=-1)
    norm_out = out.norm(dim=-1)
    assert torch.allclose(norm_in, norm_out, atol=1e-5), (
        f"Norm not preserved: max diff = {(norm_in - norm_out).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# 9. LongRoPEAttention output shape
# ---------------------------------------------------------------------------

def test_longrope_attention_output_shape():
    torch.manual_seed(0)
    cfg = make_config()
    model = LongRoPEAttention(D_MODEL, N_HEADS, cfg)
    x = torch.randn(1, 8, D_MODEL)
    out = model(x)
    assert out.shape == (1, 8, D_MODEL), f"Output shape mismatch: {out.shape}"


# ---------------------------------------------------------------------------
# 10. LongRoPEAttention works for T > max_train_len (extrapolation)
# ---------------------------------------------------------------------------

def test_longrope_attention_long_seq():
    torch.manual_seed(0)
    cfg = make_config()
    model = LongRoPEAttention(D_MODEL, N_HEADS, cfg)
    # T=32 > max_train_len=16, within target_len=64
    x = torch.randn(1, 32, D_MODEL)
    out = model(x)
    assert out.shape == (1, 32, D_MODEL), f"Extrapolation output shape mismatch: {out.shape}"


# ---------------------------------------------------------------------------
# 11. extrapolation_quality returns expected keys
# ---------------------------------------------------------------------------

def test_extrapolation_quality_keys():
    cfg = make_config()
    # Standard RoPE freqs
    half = HEAD_DIM // 2
    i = torch.arange(0, half, dtype=torch.float32)
    model_freqs = 1.0 / (cfg.base_theta ** (2 * i / HEAD_DIM))

    longrope_freqs = compute_longrope_freqs(cfg)

    result = extrapolation_quality(model_freqs, longrope_freqs)
    expected_keys = {"mean_rescale", "max_rescale", "freq_spread"}
    assert set(result.keys()) == expected_keys, (
        f"Keys mismatch: got {set(result.keys())}, expected {expected_keys}"
    )
    # Values should be finite floats
    for k, v in result.items():
        assert isinstance(v, float) and not (v != v), f"{k} is not a valid float: {v}"
