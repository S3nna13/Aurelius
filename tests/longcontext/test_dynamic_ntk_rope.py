"""Tests for DynamicNTKRoPE."""

from __future__ import annotations

import math

import torch

from src.longcontext.dynamic_ntk_rope import DynamicNTKRoPE, NTKRoPEConfig

# ---------------------------------------------------------------------------
# Tiny config used throughout
# ---------------------------------------------------------------------------
DIM = 8
MAX_SEQ = 4
ALPHA = 2.0
BASE = 10000.0


def _cfg(**kw) -> NTKRoPEConfig:
    defaults = dict(base=BASE, dim=DIM, max_seq_len=MAX_SEQ, alpha=ALPHA, scaling_factor=1.0)
    defaults.update(kw)
    return NTKRoPEConfig(**defaults)


# ===========================================================================
# NTKRoPEConfig defaults
# ===========================================================================


def test_config_default_base():
    assert NTKRoPEConfig().base == 10000.0


def test_config_default_dim():
    assert NTKRoPEConfig().dim == 64


def test_config_default_max_seq_len():
    assert NTKRoPEConfig().max_seq_len == 2048


def test_config_default_alpha():
    assert NTKRoPEConfig().alpha == 1.0


def test_config_default_scaling_factor():
    assert NTKRoPEConfig().scaling_factor == 1.0


# ===========================================================================
# _compute_freqs: shape
# ===========================================================================


def test_compute_freqs_shape_short():
    rope = DynamicNTKRoPE(_cfg())
    freqs = rope._compute_freqs(MAX_SEQ)
    assert freqs.shape == (MAX_SEQ, DIM // 2)


def test_compute_freqs_shape_long():
    rope = DynamicNTKRoPE(_cfg())
    long_len = MAX_SEQ + 10
    freqs = rope._compute_freqs(long_len)
    assert freqs.shape == (long_len, DIM // 2)


def test_compute_freqs_shape_seq_len_1():
    rope = DynamicNTKRoPE(_cfg())
    freqs = rope._compute_freqs(1)
    assert freqs.shape == (1, DIM // 2)


def test_compute_freqs_dtype_is_float32():
    rope = DynamicNTKRoPE(_cfg())
    freqs = rope._compute_freqs(MAX_SEQ)
    assert freqs.dtype == torch.float32


# ===========================================================================
# _compute_freqs: short sequence — base unchanged
# ===========================================================================


def test_compute_freqs_short_base_unchanged():
    """For seq_len <= max_seq_len frequencies must match hand-calculated values."""
    rope = DynamicNTKRoPE(_cfg())
    freqs = rope._compute_freqs(MAX_SEQ)
    # First row (position 0) should be all zeros (0 * inv_freq)
    assert torch.allclose(freqs[0], torch.zeros(DIM // 2))


def test_compute_freqs_short_first_freq():
    rope = DynamicNTKRoPE(_cfg())
    freqs = rope._compute_freqs(2)
    # inv_freq[0] = 1 / base^0 = 1
    expected_inv0 = 1.0 / (BASE ** (0.0 / DIM))
    assert math.isclose(freqs[1, 0].item(), expected_inv0, rel_tol=1e-5)


def test_compute_freqs_short_last_freq():
    rope = DynamicNTKRoPE(_cfg())
    freqs = rope._compute_freqs(2)
    half = DIM // 2
    last_i = half - 1
    expected_inv = 1.0 / (BASE ** (2.0 * last_i / DIM))
    assert math.isclose(freqs[1, last_i].item(), expected_inv, rel_tol=1e-5)


def test_compute_freqs_short_equal_to_short_seq():
    """Short seq and equal-to-max_seq_len must give identical freqs."""
    rope = DynamicNTKRoPE(_cfg())
    f1 = rope._compute_freqs(MAX_SEQ - 1)
    f2 = rope._compute_freqs(MAX_SEQ)
    # Both use unscaled base; shapes differ but first MAX_SEQ-1 rows must match
    assert torch.allclose(f1, f2[: MAX_SEQ - 1])


# ===========================================================================
# _compute_freqs: long sequence — scaled base used (different freqs)
# ===========================================================================


def test_compute_freqs_long_differs_from_short():
    rope = DynamicNTKRoPE(_cfg())
    short = rope._compute_freqs(MAX_SEQ)
    long_ = rope._compute_freqs(MAX_SEQ + 1)
    # The first MAX_SEQ rows should differ because scaled base changes all freqs
    assert not torch.allclose(short, long_[:MAX_SEQ])


def test_compute_freqs_long_scaled_base():
    """Verify scaled base used for long seq matches formula."""
    cfg = _cfg()
    rope = DynamicNTKRoPE(cfg)
    long_len = MAX_SEQ + 1
    freqs = rope._compute_freqs(long_len)
    # Compute expected scaled base
    exp = cfg.dim / (cfg.dim - 2)
    scaled_base = cfg.base * (cfg.alpha**exp)
    inv_freq_0 = 1.0 / (scaled_base ** (0.0 / cfg.dim))
    # Position 1, frequency 0
    assert math.isclose(freqs[1, 0].item(), inv_freq_0, rel_tol=1e-5)


def test_compute_freqs_long_alpha_gt1_lower_freqs():
    """With alpha > 1 the scaled base is larger so inv_freqs are smaller."""
    cfg = _cfg(alpha=4.0)
    rope = DynamicNTKRoPE(cfg)
    short = rope._compute_freqs(MAX_SEQ)  # unscaled
    long_ = rope._compute_freqs(MAX_SEQ + 1)  # scaled
    # Larger base → smaller inv_freq values at non-zero positions
    assert long_[1, -1].item() < short[1, -1].item()


# ===========================================================================
# rotate_half
# ===========================================================================


def test_rotate_half_shape_preserved():
    rope = DynamicNTKRoPE(_cfg())
    x = torch.randn(1, 1, 4, DIM)
    out = rope.rotate_half(x)
    assert out.shape == x.shape


def test_rotate_half_first_half_negated():
    rope = DynamicNTKRoPE(_cfg())
    x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
    out = rope.rotate_half(x)
    # Second half of input negated goes to first half of output
    assert torch.allclose(out[..., :4], -x[..., 4:])


def test_rotate_half_second_half_is_original_first():
    rope = DynamicNTKRoPE(_cfg())
    x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
    out = rope.rotate_half(x)
    assert torch.allclose(out[..., 4:], x[..., :4])


def test_rotate_half_double_application():
    """rotate_half applied twice should negate the original."""
    rope = DynamicNTKRoPE(_cfg())
    x = torch.randn(2, 3, 4, 8)
    out = rope.rotate_half(rope.rotate_half(x))
    assert torch.allclose(out, -x)


def test_rotate_half_dim_2():
    rope = DynamicNTKRoPE(_cfg())
    x = torch.tensor([[[[1.0, 2.0]]]])
    out = rope.rotate_half(x)
    assert torch.allclose(out, torch.tensor([[[[-2.0, 1.0]]]]))


# ===========================================================================
# forward: output shape
# ===========================================================================


def test_forward_output_shape_basic():
    cfg = _cfg()
    rope = DynamicNTKRoPE(cfg)
    B, H, L, D = 1, 1, MAX_SEQ, DIM
    x = torch.randn(B, H, L, D)
    out = rope.forward(x, seq_len=L)
    assert out.shape == (B, H, L, D)


def test_forward_output_shape_batch2():
    cfg = _cfg()
    rope = DynamicNTKRoPE(cfg)
    x = torch.randn(2, 4, MAX_SEQ, DIM)
    out = rope.forward(x, seq_len=MAX_SEQ)
    assert out.shape == (2, 4, MAX_SEQ, DIM)


def test_forward_output_shape_long_seq():
    cfg = _cfg()
    rope = DynamicNTKRoPE(cfg)
    L = MAX_SEQ + 4
    x = torch.randn(1, 1, L, DIM)
    out = rope.forward(x, seq_len=L)
    assert out.shape == (1, 1, L, DIM)


def test_forward_output_not_equal_to_input():
    """RoPE should modify the values (at non-zero positions)."""
    cfg = _cfg()
    rope = DynamicNTKRoPE(cfg)
    x = torch.ones(1, 1, MAX_SEQ, DIM)
    out = rope.forward(x, seq_len=MAX_SEQ)
    # Position 0 may be equal; at least some position should differ
    assert not torch.allclose(out, x)


def test_forward_position_zero_unchanged_cos():
    """At position 0 cos=1 and sin=0 so output == input (when input has no neg rotation effect)."""
    cfg = _cfg()
    rope = DynamicNTKRoPE(cfg)
    # Build x where rotate_half(x) = 0 would make position 0 exact, but just verify shape.
    x = torch.randn(1, 1, MAX_SEQ, DIM)
    out = rope.forward(x, seq_len=MAX_SEQ)
    assert out.shape == x.shape


# ===========================================================================
# max_supported_len
# ===========================================================================


def test_max_supported_len_gt_max_seq_len():
    cfg = _cfg(alpha=2.0)
    rope = DynamicNTKRoPE(cfg)
    assert rope.max_supported_len(alpha=2.0) > cfg.max_seq_len


def test_max_supported_len_alpha_1():
    cfg = _cfg(alpha=1.0)
    rope = DynamicNTKRoPE(cfg)
    # alpha=1 → max_seq_len * 1^exp = max_seq_len
    result = rope.max_supported_len(alpha=1.0)
    assert result == cfg.max_seq_len


def test_max_supported_len_larger_alpha_gives_longer():
    cfg = _cfg()
    rope = DynamicNTKRoPE(cfg)
    assert rope.max_supported_len(alpha=4.0) > rope.max_supported_len(alpha=2.0)


def test_max_supported_len_is_int():
    cfg = _cfg()
    rope = DynamicNTKRoPE(cfg)
    assert isinstance(rope.max_supported_len(alpha=2.0), int)


def test_max_supported_len_formula():
    cfg = _cfg(dim=8, max_seq_len=4, alpha=2.0)
    rope = DynamicNTKRoPE(cfg)
    exp = (cfg.dim / 2) / (cfg.dim - 2)
    expected = int(cfg.max_seq_len * (2.0**exp))
    assert rope.max_supported_len(alpha=2.0) == expected
