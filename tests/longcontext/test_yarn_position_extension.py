"""Unit tests for :mod:`src.longcontext.yarn_position_extension`."""

from __future__ import annotations

import time

import pytest
import torch

from src.longcontext.yarn_position_extension import (
    YarnConfig,
    apply_rotary,
    build_yarn_rotary_cache,
    yarn_inv_freq,
    yarn_linear_ramp_mask,
    yarn_mscale,
)


HEAD_DIM = 16
SCALING = 4.0


def _cfg(**overrides) -> YarnConfig:
    base = dict(
        head_dim=HEAD_DIM,
        rope_theta=10000.0,
        original_max_seq_len=128,
        scaling_factor=SCALING,
    )
    base.update(overrides)
    return YarnConfig(**base)


# ---------------------------------------------------------------------------
# inv_freq
# ---------------------------------------------------------------------------


def test_yarn_inv_freq_shape() -> None:
    cfg = _cfg()
    inv = yarn_inv_freq(cfg)
    assert inv.shape == (HEAD_DIM // 2,)


def test_yarn_inv_freq_monotonically_decreasing() -> None:
    cfg = _cfg()
    inv = yarn_inv_freq(cfg)
    diffs = inv[1:] - inv[:-1]
    # Frequencies decrease with band index (wavelengths grow).
    assert torch.all(diffs <= 0.0), diffs


def test_yarn_inv_freq_interpolation_bound() -> None:
    """Low-frequency bands must be rescaled by 1/scaling_factor."""
    cfg = _cfg()
    inv_yarn = yarn_inv_freq(cfg)
    cfg_plain = _cfg(scaling_factor=1.0)
    inv_plain = yarn_inv_freq(cfg_plain)
    # Last band is the most interpolated; ratio should be close to 1/scaling.
    ratio = (inv_yarn[-1] / inv_plain[-1]).item()
    assert ratio == pytest.approx(1.0 / SCALING, rel=1e-5)


# ---------------------------------------------------------------------------
# linear ramp mask
# ---------------------------------------------------------------------------


def test_yarn_linear_ramp_mask_in_unit_interval() -> None:
    cfg = _cfg()
    mask = yarn_linear_ramp_mask(cfg)
    assert mask.shape == (HEAD_DIM // 2,)
    assert torch.all(mask >= 0.0)
    assert torch.all(mask <= 1.0)


def test_yarn_linear_ramp_mask_monotonic() -> None:
    cfg = _cfg()
    mask = yarn_linear_ramp_mask(cfg)
    diffs = mask[1:] - mask[:-1]
    assert torch.all(diffs >= -1e-7)


# ---------------------------------------------------------------------------
# mscale
# ---------------------------------------------------------------------------


def test_yarn_mscale_at_original_is_unity() -> None:
    cfg = _cfg()
    s = yarn_mscale(cfg, cfg.original_max_seq_len).item()
    assert s == pytest.approx(1.0, abs=1e-6)


def test_yarn_mscale_larger_at_extended_positions() -> None:
    cfg = _cfg()
    inside = yarn_mscale(cfg, cfg.original_max_seq_len // 2).item()
    extended = yarn_mscale(
        cfg, int(cfg.scaling_factor * cfg.original_max_seq_len)
    ).item()
    assert extended > inside
    assert extended > 1.0


# ---------------------------------------------------------------------------
# rotary cache
# ---------------------------------------------------------------------------


def test_build_yarn_rotary_cache_shape() -> None:
    cfg = _cfg()
    seq_len = 256
    cos, sin = build_yarn_rotary_cache(cfg, seq_len)
    assert cos.shape == (seq_len, HEAD_DIM)
    assert sin.shape == (seq_len, HEAD_DIM)


def test_build_yarn_rotary_cache_value_range() -> None:
    cfg = _cfg()
    cos, sin = build_yarn_rotary_cache(cfg, 512)
    # mscale may slightly exceed 1.0 but with mscale_factor=0.1 and
    # scaling=4, mscale ~ 1.14; allow a small slack.
    assert cos.min().item() >= -1.5
    assert cos.max().item() <= 1.5
    assert sin.min().item() >= -1.5
    assert sin.max().item() <= 1.5
    # With mscale_factor=0 the range is strictly in [-1, 1].
    cfg2 = _cfg(mscale_factor=0.0)
    c2, s2 = build_yarn_rotary_cache(cfg2, 512)
    assert c2.min().item() >= -1.0 - 1e-6
    assert c2.max().item() <= 1.0 + 1e-6
    assert s2.min().item() >= -1.0 - 1e-6
    assert s2.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# apply_rotary
# ---------------------------------------------------------------------------


def test_apply_rotary_preserves_shape() -> None:
    cfg = _cfg()
    seq_len = 64
    cos, sin = build_yarn_rotary_cache(cfg, seq_len)
    x = torch.randn(2, 4, seq_len, HEAD_DIM)
    y = apply_rotary(x, cos, sin)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_apply_rotary_determinism() -> None:
    cfg = _cfg()
    seq_len = 32
    cos, sin = build_yarn_rotary_cache(cfg, seq_len)
    torch.manual_seed(0)
    x = torch.randn(1, 2, seq_len, HEAD_DIM)
    y1 = apply_rotary(x, cos, sin)
    y2 = apply_rotary(x, cos, sin)
    assert torch.equal(y1, y2)


def test_apply_rotary_at_position_zero_is_identity() -> None:
    cfg = _cfg(mscale_factor=0.0)  # remove mscale so cos[0]==1, sin[0]==0
    cos, sin = build_yarn_rotary_cache(cfg, 4)
    x = torch.randn(1, 1, 1, HEAD_DIM)
    y = apply_rotary(x, cos[:1], sin[:1])
    assert torch.allclose(y, x, atol=1e-6)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_config_rejects_odd_head_dim() -> None:
    with pytest.raises(ValueError):
        YarnConfig(head_dim=15)


def test_config_rejects_scaling_below_one() -> None:
    with pytest.raises(ValueError):
        YarnConfig(head_dim=16, scaling_factor=0.5)


# ---------------------------------------------------------------------------
# perf
# ---------------------------------------------------------------------------


def test_extrapolation_runs_fast_for_32k() -> None:
    cfg = _cfg()
    t0 = time.perf_counter()
    cos, sin = build_yarn_rotary_cache(cfg, 32768)
    elapsed = time.perf_counter() - t0
    assert cos.shape == (32768, HEAD_DIM)
    assert elapsed < 0.5, f"build_yarn_rotary_cache took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# gradient flow
# ---------------------------------------------------------------------------


def test_apply_rotary_gradient_flow() -> None:
    cfg = _cfg()
    seq_len = 16
    cos, sin = build_yarn_rotary_cache(cfg, seq_len)
    x = torch.randn(1, 2, seq_len, HEAD_DIM, requires_grad=True)
    out = apply_rotary(x, cos, sin)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0.0
