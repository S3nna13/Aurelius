"""Tests for position_interpolation.py — new PIConfig-based API.

Tiny configs: D_HEAD=8, ORIG_LEN=16, EXT_LEN=32, B=2, T=8
All tests use pure PyTorch tensors only.
"""

from __future__ import annotations

import pytest
import torch

from src.model.position_interpolation_v2 import (
    PIConfig,
    PositionInterpolator,
    _standard_freqs_cis,
    build_interpolated_freqs_cis,
    compute_scale_factor,
    interpolate_freqs_cis,
    ntk_aware_freqs_cis,
    yarn_freqs_cis,
)

# ---------------------------------------------------------------------------
# Tiny test constants
# ---------------------------------------------------------------------------

D_HEAD = 8
ORIG_LEN = 16
EXT_LEN = 32
B = 2
T = 8


# ---------------------------------------------------------------------------
# 1. PIConfig defaults
# ---------------------------------------------------------------------------


def test_piconfig_defaults():
    cfg = PIConfig()
    assert cfg.d_head == 64
    assert cfg.original_max_seq_len == 2048
    assert cfg.extended_max_seq_len == 8192
    assert cfg.base == 10000.0
    assert cfg.method == "linear"


# ---------------------------------------------------------------------------
# 2. compute_scale_factor linear = extended / original
# ---------------------------------------------------------------------------


def test_compute_scale_factor_linear():
    cfg = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=EXT_LEN,
        method="linear",
    )
    scale = compute_scale_factor(cfg)
    assert scale == pytest.approx(EXT_LEN / ORIG_LEN, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. compute_scale_factor NTK != linear
# ---------------------------------------------------------------------------


def test_compute_scale_factor_ntk_differs_from_linear():
    cfg_lin = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=EXT_LEN,
        method="linear",
    )
    cfg_ntk = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=EXT_LEN,
        method="ntk",
    )
    scale_lin = compute_scale_factor(cfg_lin)
    scale_ntk = compute_scale_factor(cfg_ntk)
    assert scale_ntk != pytest.approx(scale_lin, rel=1e-3)


# ---------------------------------------------------------------------------
# 4. build_interpolated_freqs_cis shape (EXT_LEN, D_HEAD // 2)
# ---------------------------------------------------------------------------


def test_build_interpolated_freqs_cis_shape():
    cfg = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=EXT_LEN,
        method="linear",
    )
    freqs = build_interpolated_freqs_cis(cfg)
    assert freqs.shape == (EXT_LEN, D_HEAD // 2)


# ---------------------------------------------------------------------------
# 5. interpolated freqs_cis is complex dtype
# ---------------------------------------------------------------------------


def test_build_interpolated_freqs_cis_complex_dtype():
    cfg = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=EXT_LEN,
        method="linear",
    )
    freqs = build_interpolated_freqs_cis(cfg)
    assert freqs.is_complex(), f"Expected complex tensor, got {freqs.dtype}"


# ---------------------------------------------------------------------------
# 6. freqs_cis unit magnitude (|e^{i*theta}| = 1)
# ---------------------------------------------------------------------------


def test_freqs_cis_unit_magnitude():
    cfg = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=EXT_LEN,
        method="linear",
    )
    freqs = build_interpolated_freqs_cis(cfg)
    magnitudes = freqs.abs()
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)


# ---------------------------------------------------------------------------
# 7. ntk_aware_freqs_cis shape
# ---------------------------------------------------------------------------


def test_ntk_aware_freqs_cis_shape():
    scale = 2.0
    freqs = ntk_aware_freqs_cis(D_HEAD, EXT_LEN, base=10000.0, scale=scale)
    assert freqs.shape == (EXT_LEN, D_HEAD // 2)


# ---------------------------------------------------------------------------
# 8. yarn_freqs_cis shape
# ---------------------------------------------------------------------------


def test_yarn_freqs_cis_shape():
    scale = 2.0
    freqs = yarn_freqs_cis(D_HEAD, EXT_LEN, base=10000.0, scale=scale)
    assert freqs.shape == (EXT_LEN, D_HEAD // 2)


# ---------------------------------------------------------------------------
# 9. PositionInterpolator.get_freqs_cis shape for short seq_len
# ---------------------------------------------------------------------------


def test_position_interpolator_get_freqs_cis_shape_short():
    cfg = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=EXT_LEN,
        method="linear",
    )
    interp = PositionInterpolator(cfg)
    freqs = interp.get_freqs_cis(T)
    assert freqs.shape == (T, D_HEAD // 2)


# ---------------------------------------------------------------------------
# 10. get_freqs_cis for extended length (> original_max_seq_len) works
# ---------------------------------------------------------------------------


def test_position_interpolator_get_freqs_cis_extended():
    cfg = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=EXT_LEN,
        method="linear",
    )
    interp = PositionInterpolator(cfg)
    freqs = interp.get_freqs_cis(EXT_LEN)
    assert freqs.shape == (EXT_LEN, D_HEAD // 2)
    assert freqs.is_complex()


# ---------------------------------------------------------------------------
# 11. extend_context output shape
# ---------------------------------------------------------------------------


def test_extend_context_output_shape():
    cfg = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=EXT_LEN,
        method="linear",
    )
    interp = PositionInterpolator(cfg)
    original_freqs = _standard_freqs_cis(D_HEAD, ORIG_LEN, 10000.0)
    extended = interp.extend_context(original_freqs, EXT_LEN)
    assert extended.shape == (EXT_LEN, D_HEAD // 2)


# ---------------------------------------------------------------------------
# 12. linear and NTK produce different freqs_cis for same seq_len
# ---------------------------------------------------------------------------


def test_linear_and_ntk_differ():
    seq_len = EXT_LEN

    cfg_lin = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=seq_len,
        method="linear",
    )
    cfg_ntk = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=seq_len,
        method="ntk",
    )

    freqs_lin = build_interpolated_freqs_cis(cfg_lin)
    freqs_ntk = build_interpolated_freqs_cis(cfg_ntk)

    # They should differ — confirm at least one element differs
    diff = (freqs_lin - freqs_ntk).abs().max().item()
    assert diff > 1e-4, f"Linear and NTK freqs_cis are unexpectedly identical (diff={diff})"


# ---------------------------------------------------------------------------
# 13. scale_factor > 1 when extended > original
# ---------------------------------------------------------------------------


def test_scale_factor_greater_than_one():
    for method in ("linear", "ntk", "yarn"):
        cfg = PIConfig(
            d_head=D_HEAD,
            original_max_seq_len=ORIG_LEN,
            extended_max_seq_len=EXT_LEN,
            method=method,
        )
        scale = compute_scale_factor(cfg)
        assert scale > 1.0, f"scale={scale} not > 1.0 for method={method}"


# ---------------------------------------------------------------------------
# 14. YaRN and NTK produce different freqs_cis
# ---------------------------------------------------------------------------


def test_yarn_and_ntk_differ():
    scale = EXT_LEN / ORIG_LEN

    freqs_ntk = ntk_aware_freqs_cis(D_HEAD, EXT_LEN, base=10000.0, scale=scale)
    freqs_yarn = yarn_freqs_cis(D_HEAD, EXT_LEN, base=10000.0, scale=scale)

    diff = (freqs_ntk - freqs_yarn).abs().max().item()
    assert diff > 1e-4, f"YaRN and NTK freqs_cis are unexpectedly identical (diff={diff})"


# ---------------------------------------------------------------------------
# 15. freqs_cis preserves periodicity — magnitude = 1 for all methods
# ---------------------------------------------------------------------------


def test_freqs_cis_unit_magnitude_all_methods():
    scale = EXT_LEN / ORIG_LEN

    freqs_ntk = ntk_aware_freqs_cis(D_HEAD, EXT_LEN, base=10000.0, scale=scale)
    freqs_yarn = yarn_freqs_cis(D_HEAD, EXT_LEN, base=10000.0, scale=scale)

    for name, freqs in [("ntk", freqs_ntk), ("yarn", freqs_yarn)]:
        mags = freqs.abs()
        assert torch.allclose(mags, torch.ones_like(mags), atol=1e-5), (
            f"Method {name} magnitude not 1: max_dev={(mags - 1).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# 16. interpolate_freqs_cis produces correct output shape
# ---------------------------------------------------------------------------


def test_interpolate_freqs_cis_shape():
    original = _standard_freqs_cis(D_HEAD, ORIG_LEN, 10000.0)
    scale = EXT_LEN / ORIG_LEN
    extended = interpolate_freqs_cis(original, EXT_LEN, scale)
    assert extended.shape == (EXT_LEN, D_HEAD // 2)
    assert extended.is_complex()


# ---------------------------------------------------------------------------
# 17. PositionInterpolator caches results (same object returned on second call)
# ---------------------------------------------------------------------------


def test_position_interpolator_caches():
    cfg = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=EXT_LEN,
        method="linear",
    )
    interp = PositionInterpolator(cfg)
    freqs_a = interp.get_freqs_cis(T)
    freqs_b = interp.get_freqs_cis(T)
    assert freqs_a is freqs_b, "Expected same cached tensor object on second call"


# ---------------------------------------------------------------------------
# 18. build_interpolated_freqs_cis works for NTK method
# ---------------------------------------------------------------------------


def test_build_interpolated_freqs_cis_ntk():
    cfg = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=EXT_LEN,
        method="ntk",
    )
    freqs = build_interpolated_freqs_cis(cfg)
    assert freqs.shape == (EXT_LEN, D_HEAD // 2)
    assert freqs.is_complex()
    # Unit magnitude
    mags = freqs.abs()
    assert torch.allclose(mags, torch.ones_like(mags), atol=1e-5)


# ---------------------------------------------------------------------------
# 19. build_interpolated_freqs_cis works for YaRN method
# ---------------------------------------------------------------------------


def test_build_interpolated_freqs_cis_yarn():
    cfg = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=EXT_LEN,
        method="yarn",
    )
    freqs = build_interpolated_freqs_cis(cfg)
    assert freqs.shape == (EXT_LEN, D_HEAD // 2)
    assert freqs.is_complex()


# ---------------------------------------------------------------------------
# 20. NTK scale = ratio^(d_head/(d_head-2)) matches manual computation
# ---------------------------------------------------------------------------


def test_ntk_scale_factor_formula():
    cfg = PIConfig(
        d_head=D_HEAD,
        original_max_seq_len=ORIG_LEN,
        extended_max_seq_len=EXT_LEN,
        method="ntk",
    )
    scale = compute_scale_factor(cfg)
    ratio = EXT_LEN / ORIG_LEN
    exponent = D_HEAD / (D_HEAD - 2)
    expected = ratio**exponent
    assert scale == pytest.approx(expected, rel=1e-6)
