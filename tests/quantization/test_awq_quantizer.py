"""Tests for src/quantization/awq_quantizer.py — covers legacy and new API."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.quantization.awq_quantizer import (
    ActivationStats,
    AWQConfig,
    AWQQuantizer,
    AWQScaleSearch,
    AWQ_QUANTIZER_REGISTRY,
    QUANTIZATION_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_weight() -> "torch.Tensor":
    torch.manual_seed(0)
    return torch.randn(4, 8)


@pytest.fixture()
def tiny_activations() -> "torch.Tensor":
    torch.manual_seed(1)
    return torch.randn(16, 8)


@pytest.fixture()
def config4bit() -> AWQConfig:
    return AWQConfig(bits=4, group_size=128, zero_point=True)


@pytest.fixture()
def config8bit() -> AWQConfig:
    return AWQConfig(bits=8, group_size=128, zero_point=True)


# ---------------------------------------------------------------------------
# AWQConfig
# ---------------------------------------------------------------------------

class TestAWQConfig:
    def test_defaults(self):
        cfg = AWQConfig()
        assert cfg.bits == 4
        assert cfg.group_size == 128
        assert cfg.zero_point is True
        assert cfg.version == "gemm"

    def test_clip_ratio_backcompat(self):
        cfg = AWQConfig()
        assert cfg.clip_ratio == pytest.approx(0.9)

    def test_custom(self):
        cfg = AWQConfig(bits=8, group_size=64, zero_point=False, version="gemv")
        assert cfg.bits == 8
        assert cfg.group_size == 64
        assert cfg.zero_point is False
        assert cfg.version == "gemv"


# ---------------------------------------------------------------------------
# ActivationStats
# ---------------------------------------------------------------------------

class TestActivationStats:
    def test_frozen(self, tiny_activations):
        q = AWQQuantizer()
        stats = q.collect_activation_stats(tiny_activations)
        with pytest.raises(Exception):
            stats.channel_scales = torch.zeros(1)  # type: ignore[misc]

    def test_fields(self, tiny_activations):
        q = AWQQuantizer()
        stats = q.collect_activation_stats(tiny_activations)
        assert isinstance(stats, ActivationStats)
        assert stats.channel_scales.shape == (tiny_activations.shape[-1],)
        assert stats.max_activations.shape == (tiny_activations.shape[-1],)


# ---------------------------------------------------------------------------
# collect_activation_stats
# ---------------------------------------------------------------------------

class TestCollectActivationStats:
    def test_shape_2d(self, tiny_activations):
        q = AWQQuantizer()
        stats = q.collect_activation_stats(tiny_activations)
        assert stats.channel_scales.shape == (8,)
        assert stats.max_activations.shape == (8,)

    def test_shape_3d(self):
        q = AWQQuantizer()
        acts = torch.randn(2, 5, 8)
        stats = q.collect_activation_stats(acts)
        assert stats.channel_scales.shape == (8,)
        assert stats.max_activations.shape == (8,)

    def test_non_negative(self, tiny_activations):
        stats = AWQQuantizer().collect_activation_stats(tiny_activations)
        assert (stats.channel_scales >= 0).all()
        assert (stats.max_activations >= 0).all()

    def test_max_ge_mean(self, tiny_activations):
        stats = AWQQuantizer().collect_activation_stats(tiny_activations)
        assert (stats.max_activations + 1e-6 >= stats.channel_scales).all()

    def test_scalar_activation_raises(self):
        q = AWQQuantizer()
        with pytest.raises(ValueError):
            q.collect_activation_stats(torch.tensor(0.0))


# ---------------------------------------------------------------------------
# compute_scale_factor
# ---------------------------------------------------------------------------

class TestComputeScaleFactor:
    def test_shape(self, tiny_weight, tiny_activations):
        q = AWQQuantizer()
        stats = q.collect_activation_stats(tiny_activations)
        sf = q.compute_scale_factor(tiny_weight, stats)
        assert sf.shape == (tiny_weight.shape[-1],)

    def test_positive(self, tiny_weight, tiny_activations):
        q = AWQQuantizer()
        stats = q.collect_activation_stats(tiny_activations)
        sf = q.compute_scale_factor(tiny_weight, stats)
        assert (sf > 0).all()

    def test_finite(self, tiny_weight, tiny_activations):
        q = AWQQuantizer()
        stats = q.collect_activation_stats(tiny_activations)
        sf = q.compute_scale_factor(tiny_weight, stats)
        assert torch.isfinite(sf).all()

    def test_clamp_floor(self):
        q = AWQQuantizer()
        stats = ActivationStats(
            channel_scales=torch.zeros(4),
            max_activations=torch.zeros(4),
        )
        w = torch.randn(3, 4)
        sf = q.compute_scale_factor(w, stats)
        assert (sf >= 1e-4 - 1e-8).all()


# ---------------------------------------------------------------------------
# quantize_with_scales / reconstruction_error
# ---------------------------------------------------------------------------

class TestQuantizeWithScales:
    def test_shapes(self, tiny_weight, tiny_activations, config4bit):
        q = AWQQuantizer(config4bit)
        stats = q.collect_activation_stats(tiny_activations)
        sf = q.compute_scale_factor(tiny_weight, stats)
        w_int, scale, zero = q.quantize_with_scales(tiny_weight, sf)
        assert w_int.shape == tiny_weight.shape
        assert scale.shape == (tiny_weight.shape[0],)
        assert zero.shape == (tiny_weight.shape[0],)

    def test_range_4bit(self, tiny_weight, tiny_activations, config4bit):
        q = AWQQuantizer(config4bit)
        stats = q.collect_activation_stats(tiny_activations)
        sf = q.compute_scale_factor(tiny_weight, stats)
        w_int, _, _ = q.quantize_with_scales(tiny_weight, sf)
        assert int(w_int.min().item()) >= 0
        assert int(w_int.max().item()) <= 15

    def test_range_8bit(self, tiny_weight, tiny_activations, config8bit):
        q = AWQQuantizer(config8bit)
        stats = q.collect_activation_stats(tiny_activations)
        sf = q.compute_scale_factor(tiny_weight, stats)
        w_int, _, _ = q.quantize_with_scales(tiny_weight, sf)
        assert int(w_int.min().item()) >= 0
        assert int(w_int.max().item()) <= 255

    def test_int_dtype(self, tiny_weight, tiny_activations, config4bit):
        q = AWQQuantizer(config4bit)
        stats = q.collect_activation_stats(tiny_activations)
        sf = q.compute_scale_factor(tiny_weight, stats)
        w_int, _, _ = q.quantize_with_scales(tiny_weight, sf)
        assert w_int.dtype == torch.int32

    def test_symmetric(self, tiny_weight, tiny_activations):
        cfg = AWQConfig(bits=4, zero_point=False)
        q = AWQQuantizer(cfg)
        stats = q.collect_activation_stats(tiny_activations)
        sf = q.compute_scale_factor(tiny_weight, stats)
        _, _, zero = q.quantize_with_scales(tiny_weight, sf)
        assert torch.all(zero == 0)

    def test_shape_mismatch(self, tiny_weight):
        q = AWQQuantizer()
        bad = torch.ones(3)
        with pytest.raises(ValueError):
            q.quantize_with_scales(tiny_weight, bad)


class TestReconstructionError:
    def test_finite(self, tiny_weight, tiny_activations, config4bit):
        q = AWQQuantizer(config4bit)
        stats = q.collect_activation_stats(tiny_activations)
        sf = q.compute_scale_factor(tiny_weight, stats)
        w_int, scale, zero = q.quantize_with_scales(tiny_weight, sf)
        err = q.reconstruction_error(tiny_weight, w_int, scale, zero)
        assert err >= 0
        assert err == err  # not NaN

    def test_reasonable_4bit(self, tiny_weight, tiny_activations, config4bit):
        q = AWQQuantizer(config4bit)
        stats = q.collect_activation_stats(tiny_activations)
        sf = q.compute_scale_factor(tiny_weight, stats)
        w_int, scale, zero = q.quantize_with_scales(tiny_weight, sf)
        err = q.reconstruction_error(tiny_weight, w_int, scale, zero)
        assert err < 10.0


# ---------------------------------------------------------------------------
# Legacy scale search
# ---------------------------------------------------------------------------

class TestLegacyAWQScaleSearch:
    def test_shape(self, tiny_weight, tiny_activations):
        s = AWQScaleSearch()
        scales = s.search_scales(tiny_weight, tiny_activations)
        assert scales.shape == (tiny_weight.shape[0],)

    def test_positive(self, tiny_weight, tiny_activations):
        s = AWQScaleSearch()
        scales = s.search_scales(tiny_weight, tiny_activations)
        assert (scales > 0).all()


# ---------------------------------------------------------------------------
# Legacy layer API
# ---------------------------------------------------------------------------

class TestLegacyQuantizeLayer:
    def test_roundtrip(self, tiny_weight, tiny_activations, config8bit):
        q = AWQQuantizer(config8bit)
        qw, scales, zeros = q.quantize_layer(tiny_weight, tiny_activations)
        recon = q.dequantize(qw, scales, zeros)
        assert recon.shape == tiny_weight.shape
        assert torch.isfinite(recon).all()

    def test_range_4bit(self, tiny_weight, tiny_activations, config4bit):
        q = AWQQuantizer(config4bit)
        qw, _, _ = q.quantize_layer(tiny_weight, tiny_activations)
        assert int(qw.min().item()) >= 0
        assert int(qw.max().item()) <= 15


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_awq_legacy(self):
        assert QUANTIZATION_REGISTRY["awq"] is AWQQuantizer

    def test_default(self):
        assert AWQ_QUANTIZER_REGISTRY["default"] is AWQQuantizer
