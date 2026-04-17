"""Tests for src/model/rope_extensions.py — Extended RoPE variants."""

from __future__ import annotations

import math

import pytest
import torch

from aurelius.model.rope_extensions import (
    BaseRoPE,
    DynamicNTKRoPE,
    LinearScaledRoPE,
    NTKScaledRoPE,
    RoPEConfig,
    YaRNRoPE,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HEAD_DIM = 64
SEQ_LEN = 128
ORIGINAL_MAX_LEN = 128
DEVICE = torch.device("cpu")


def make_config(
    head_dim: int = HEAD_DIM,
    scale_factor: float = 1.0,
    original_max_len: int = ORIGINAL_MAX_LEN,
    max_seq_len: int = 512,
) -> RoPEConfig:
    return RoPEConfig(
        head_dim=head_dim,
        base=10000.0,
        max_seq_len=max_seq_len,
        scale_factor=scale_factor,
        original_max_len=original_max_len,
    )


def make_input(B: int = 2, T: int = SEQ_LEN, n_heads: int = 4, head_dim: int = HEAD_DIM) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(B, T, n_heads, head_dim)


# ===========================================================================
# 1. BaseRoPE.get_freqs — shape
# ===========================================================================

def test_base_rope_get_freqs_shape():
    cfg = make_config()
    rope = BaseRoPE(cfg)
    freqs = rope.get_freqs(SEQ_LEN, DEVICE)
    assert freqs.shape == (SEQ_LEN, HEAD_DIM // 2), (
        f"Expected ({SEQ_LEN}, {HEAD_DIM // 2}), got {freqs.shape}"
    )


# ===========================================================================
# 2. BaseRoPE.apply — output shape matches input shape
# ===========================================================================

def test_base_rope_apply_output_shape():
    cfg = make_config()
    rope = BaseRoPE(cfg)
    x = make_input()
    out = rope.apply(x, SEQ_LEN)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ===========================================================================
# 3. BaseRoPE.apply — rotation is norm-preserving
# ===========================================================================

def test_base_rope_apply_norm_preserving():
    cfg = make_config()
    rope = BaseRoPE(cfg)
    x = make_input()
    out = rope.apply(x, SEQ_LEN)
    # Compare per-vector norms along head_dim axis
    norm_in = x.norm(dim=-1)
    norm_out = out.norm(dim=-1)
    assert torch.allclose(norm_in, norm_out, atol=1e-5), (
        "RoPE application should preserve L2 norm (rotation)."
    )


# ===========================================================================
# 4. BaseRoPE — different positions produce different rotations
# ===========================================================================

def test_base_rope_different_positions_differ():
    cfg = make_config()
    rope = BaseRoPE(cfg)
    # Use T=2 so we compare position 0 vs position 1
    x = make_input(T=2)
    out = rope.apply(x, seq_len=2)
    # Output at position 0 and position 1 should differ (same input vector)
    x_uniform = torch.ones(1, 2, 1, HEAD_DIM)
    out_uniform = rope.apply(x_uniform, seq_len=2)
    assert not torch.allclose(out_uniform[:, 0], out_uniform[:, 1], atol=1e-6), (
        "RoPE should produce different rotations for different positions."
    )


# ===========================================================================
# 5. LinearScaledRoPE freqs differ from BaseRoPE when scale_factor != 1
# ===========================================================================

def test_linear_scaled_rope_freqs_differ_from_base():
    base_cfg = make_config(scale_factor=1.0)
    scaled_cfg = make_config(scale_factor=2.0)

    base_rope = BaseRoPE(base_cfg)
    linear_rope = LinearScaledRoPE(scaled_cfg)

    freqs_base = base_rope.get_freqs(SEQ_LEN, DEVICE)
    freqs_linear = linear_rope.get_freqs(SEQ_LEN, DEVICE)

    assert not torch.allclose(freqs_base, freqs_linear, atol=1e-6), (
        "LinearScaledRoPE freqs should differ from BaseRoPE when scale_factor != 1."
    )


# ===========================================================================
# 6. NTKScaledRoPE freqs differ from BaseRoPE
# ===========================================================================

def test_ntk_scaled_rope_freqs_differ_from_base():
    base_cfg = make_config(scale_factor=1.0)
    ntk_cfg = make_config(scale_factor=4.0)

    base_rope = BaseRoPE(base_cfg)
    ntk_rope = NTKScaledRoPE(ntk_cfg)

    freqs_base = base_rope.get_freqs(SEQ_LEN, DEVICE)
    freqs_ntk = ntk_rope.get_freqs(SEQ_LEN, DEVICE)

    assert not torch.allclose(freqs_base, freqs_ntk, atol=1e-6), (
        "NTKScaledRoPE freqs should differ from BaseRoPE."
    )


# ===========================================================================
# 7. YaRNRoPE output shape matches input
# ===========================================================================

def test_yarn_rope_output_shape():
    cfg = make_config(scale_factor=2.0)
    rope = YaRNRoPE(cfg)
    x = make_input()
    out = rope.apply(x, SEQ_LEN)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ===========================================================================
# 8. YaRNRoPE with alpha=0, beta=1 (nearly all NTK) differs from LinearScaledRoPE
# ===========================================================================

def test_yarn_rope_all_ntk_differs_from_linear():
    cfg = make_config(scale_factor=2.0)
    yarn_rope = YaRNRoPE(cfg, alpha=0.0, beta=1.0)
    linear_rope = LinearScaledRoPE(cfg)

    freqs_yarn = yarn_rope.get_freqs(SEQ_LEN, DEVICE)
    freqs_linear = linear_rope.get_freqs(SEQ_LEN, DEVICE)

    assert not torch.allclose(freqs_yarn, freqs_linear, atol=1e-6), (
        "YaRNRoPE (all-NTK ramp) should differ from LinearScaledRoPE."
    )


# ===========================================================================
# 9. DynamicNTKRoPE at seq_len <= original_max_len matches BaseRoPE
# ===========================================================================

def test_dynamic_ntk_rope_short_seq_matches_base():
    cfg = make_config(original_max_len=ORIGINAL_MAX_LEN)
    base_rope = BaseRoPE(cfg)
    dyn_rope = DynamicNTKRoPE(cfg)

    short_len = ORIGINAL_MAX_LEN  # exactly at the boundary — should use standard base
    freqs_base = base_rope.get_freqs(short_len, DEVICE)
    freqs_dyn = dyn_rope.get_freqs(short_len, DEVICE)

    assert torch.allclose(freqs_base, freqs_dyn, atol=1e-6), (
        "DynamicNTKRoPE should match BaseRoPE when seq_len <= original_max_len."
    )


# ===========================================================================
# 10. DynamicNTKRoPE at seq_len > original_max_len differs from BaseRoPE
# ===========================================================================

def test_dynamic_ntk_rope_long_seq_differs_from_base():
    cfg = make_config(original_max_len=ORIGINAL_MAX_LEN)
    base_rope = BaseRoPE(cfg)
    dyn_rope = DynamicNTKRoPE(cfg)

    long_len = ORIGINAL_MAX_LEN * 2
    freqs_base = base_rope.get_freqs(long_len, DEVICE)
    freqs_dyn = dyn_rope.get_freqs(long_len, DEVICE)

    assert not torch.allclose(freqs_base, freqs_dyn, atol=1e-6), (
        "DynamicNTKRoPE should differ from BaseRoPE when seq_len > original_max_len."
    )


# ===========================================================================
# 11. All variants produce finite values
# ===========================================================================

@pytest.mark.parametrize("rope_cls,kwargs", [
    (BaseRoPE, {}),
    (LinearScaledRoPE, {"scale_factor": 2.0}),
    (NTKScaledRoPE, {"scale_factor": 4.0}),
    (YaRNRoPE, {"scale_factor": 2.0}),
    (DynamicNTKRoPE, {}),
])
def test_all_variants_finite(rope_cls, kwargs):
    cfg = make_config(**kwargs)
    if rope_cls is YaRNRoPE:
        rope = YaRNRoPE(cfg)
    else:
        rope = rope_cls(cfg)
    freqs = rope.get_freqs(SEQ_LEN, DEVICE)
    assert torch.isfinite(freqs).all(), (
        f"{rope_cls.__name__} produced non-finite frequency values."
    )
    x = make_input()
    out = rope.apply(x, SEQ_LEN)
    assert torch.isfinite(out).all(), (
        f"{rope_cls.__name__} produced non-finite output values."
    )


# ===========================================================================
# 12. RoPE applied once is NOT the identity (rotation is actually applied)
# ===========================================================================

def test_rope_applied_is_not_identity():
    cfg = make_config()
    rope = BaseRoPE(cfg)
    x = make_input(T=16)
    out = rope.apply(x, seq_len=16)
    # The output should differ from the input (at least for non-zero positions)
    assert not torch.allclose(x, out, atol=1e-6), (
        "RoPE output should differ from the input — rotation must be applied."
    )


# ===========================================================================
# 13. head_dim=64 works for all variants
# ===========================================================================

@pytest.mark.parametrize("rope_cls", [
    BaseRoPE,
    LinearScaledRoPE,
    NTKScaledRoPE,
    YaRNRoPE,
    DynamicNTKRoPE,
])
def test_head_dim_64(rope_cls):
    cfg = make_config(head_dim=64, scale_factor=2.0)
    if rope_cls is YaRNRoPE:
        rope = YaRNRoPE(cfg)
    else:
        rope = rope_cls(cfg)
    x = torch.randn(1, 16, 2, 64)
    out = rope.apply(x, seq_len=16)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# ===========================================================================
# 14. Works with seq_len=1
# ===========================================================================

@pytest.mark.parametrize("rope_cls", [
    BaseRoPE,
    LinearScaledRoPE,
    NTKScaledRoPE,
    YaRNRoPE,
    DynamicNTKRoPE,
])
def test_seq_len_1(rope_cls):
    cfg = make_config(scale_factor=2.0, original_max_len=128)
    if rope_cls is YaRNRoPE:
        rope = YaRNRoPE(cfg)
    else:
        rope = rope_cls(cfg)
    freqs = rope.get_freqs(1, DEVICE)
    assert freqs.shape == (1, HEAD_DIM // 2)

    x = torch.randn(1, 1, 2, HEAD_DIM)
    out = rope.apply(x, seq_len=1)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
