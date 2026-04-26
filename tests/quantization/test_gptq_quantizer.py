"""Tests for src/quantization/gptq_quantizer.py."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.quantization.gptq_quantizer import (  # noqa: E402
    GPTQ_QUANTIZER_REGISTRY,
    GPTQConfig,
    GPTQQuantizer,
    QuantizedLayer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_weight() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(8, 16)


@pytest.fixture()
def cfg_sym() -> GPTQConfig:
    return GPTQConfig(bits=4, group_size=128, sym=True)


@pytest.fixture()
def cfg_asym() -> GPTQConfig:
    return GPTQConfig(bits=4, group_size=128, sym=False)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestGPTQConfig:
    def test_defaults(self):
        c = GPTQConfig()
        assert c.bits == 4
        assert c.group_size == 128
        assert c.desc_act is False
        assert c.damp_percent == pytest.approx(0.01)
        assert c.sym is True

    def test_custom(self):
        c = GPTQConfig(bits=8, group_size=64, desc_act=True, damp_percent=0.05, sym=False)
        assert c.bits == 8
        assert c.group_size == 64
        assert c.desc_act is True
        assert c.damp_percent == pytest.approx(0.05)
        assert c.sym is False


# ---------------------------------------------------------------------------
# QuantizedLayer container
# ---------------------------------------------------------------------------


class TestQuantizedLayer:
    def test_frozen(self, small_weight, cfg_sym):
        q = GPTQQuantizer(cfg_sym)
        layer = q.quantize_weight(small_weight)
        with pytest.raises(Exception):
            layer.bits = 8  # type: ignore[misc]

    def test_fields_populated_sym(self, small_weight, cfg_sym):
        q = GPTQQuantizer(cfg_sym)
        layer = q.quantize_weight(small_weight)
        assert layer.weight_int is not None
        assert layer.scale is not None
        assert layer.zero_point is not None
        assert layer.bits == 4
        assert layer.group_size == 128

    def test_fields_populated_asym(self, small_weight, cfg_asym):
        q = GPTQQuantizer(cfg_asym)
        layer = q.quantize_weight(small_weight)
        assert layer.weight_int is not None
        assert layer.scale is not None
        assert layer.zero_point is not None


# ---------------------------------------------------------------------------
# Quantization semantics
# ---------------------------------------------------------------------------


class TestQuantizeWeight:
    def test_shape_preserved(self, small_weight, cfg_sym):
        layer = GPTQQuantizer(cfg_sym).quantize_weight(small_weight)
        assert layer.weight_int.shape == small_weight.shape

    def test_sym_range_4bit(self, small_weight, cfg_sym):
        layer = GPTQQuantizer(cfg_sym).quantize_weight(small_weight)
        assert int(layer.weight_int.min().item()) >= -8
        assert int(layer.weight_int.max().item()) <= 7

    def test_asym_range_4bit(self, small_weight, cfg_asym):
        layer = GPTQQuantizer(cfg_asym).quantize_weight(small_weight)
        assert int(layer.weight_int.min().item()) >= 0
        assert int(layer.weight_int.max().item()) <= 15

    def test_sym_range_8bit(self, small_weight):
        cfg = GPTQConfig(bits=8, sym=True)
        layer = GPTQQuantizer(cfg).quantize_weight(small_weight)
        assert int(layer.weight_int.min().item()) >= -128
        assert int(layer.weight_int.max().item()) <= 127

    def test_asym_range_8bit(self, small_weight):
        cfg = GPTQConfig(bits=8, sym=False)
        layer = GPTQQuantizer(cfg).quantize_weight(small_weight)
        assert int(layer.weight_int.min().item()) >= 0
        assert int(layer.weight_int.max().item()) <= 255

    def test_int_dtype(self, small_weight, cfg_sym):
        layer = GPTQQuantizer(cfg_sym).quantize_weight(small_weight)
        assert layer.weight_int.dtype == torch.int32

    def test_scale_positive(self, small_weight, cfg_sym):
        layer = GPTQQuantizer(cfg_sym).quantize_weight(small_weight)
        assert float(layer.scale.item()) > 0.0

    def test_type_error_on_non_tensor(self, cfg_sym):
        q = GPTQQuantizer(cfg_sym)
        with pytest.raises(TypeError):
            q.quantize_weight([1.0, 2.0])  # type: ignore[arg-type]

    def test_zeros_weight_handled(self, cfg_sym):
        w = torch.zeros(4, 4)
        layer = GPTQQuantizer(cfg_sym).quantize_weight(w)
        assert torch.all(layer.weight_int == 0)


# ---------------------------------------------------------------------------
# Dequantization / round-trip
# ---------------------------------------------------------------------------


class TestDequantize:
    def test_shape_preserved(self, small_weight, cfg_sym):
        q = GPTQQuantizer(cfg_sym)
        recon = q.dequantize(q.quantize_weight(small_weight))
        assert recon.shape == small_weight.shape

    def test_finite(self, small_weight, cfg_sym):
        q = GPTQQuantizer(cfg_sym)
        recon = q.dequantize(q.quantize_weight(small_weight))
        assert torch.isfinite(recon).all()

    def test_roundtrip_mse_small_8bit_sym(self, small_weight):
        cfg = GPTQConfig(bits=8, sym=True)
        q = GPTQQuantizer(cfg)
        layer = q.quantize_weight(small_weight)
        err = q.quantize_error(small_weight, layer)
        assert err < 1e-2

    def test_roundtrip_mse_small_8bit_asym(self, small_weight):
        cfg = GPTQConfig(bits=8, sym=False)
        q = GPTQQuantizer(cfg)
        layer = q.quantize_weight(small_weight)
        err = q.quantize_error(small_weight, layer)
        assert err < 1e-2

    def test_roundtrip_mse_reasonable_4bit(self, small_weight, cfg_sym):
        q = GPTQQuantizer(cfg_sym)
        err = q.quantize_error(small_weight, q.quantize_weight(small_weight))
        assert err < 1.0

    def test_dequantize_empty_raises(self, cfg_sym):
        q = GPTQQuantizer(cfg_sym)
        empty = QuantizedLayer(weight_int=None, scale=None, zero_point=None, bits=4, group_size=128)
        with pytest.raises(ValueError):
            q.dequantize(empty)


# ---------------------------------------------------------------------------
# bits_saved_ratio / misc
# ---------------------------------------------------------------------------


class TestBitsSavedRatio:
    def test_default(self):
        q = GPTQQuantizer(GPTQConfig(bits=4))
        assert q.bits_saved_ratio() == pytest.approx(4 / 32)

    def test_custom_original(self):
        q = GPTQQuantizer(GPTQConfig(bits=8))
        assert q.bits_saved_ratio(16) == pytest.approx(0.5)

    def test_invalid_original_raises(self):
        q = GPTQQuantizer(GPTQConfig(bits=4))
        with pytest.raises(ValueError):
            q.bits_saved_ratio(0)

    def test_ratio_less_than_one(self):
        q = GPTQQuantizer(GPTQConfig(bits=4))
        assert q.bits_saved_ratio(32) < 1.0


class TestRegistry:
    def test_default_key(self):
        assert "default" in GPTQ_QUANTIZER_REGISTRY

    def test_default_cls(self):
        assert GPTQ_QUANTIZER_REGISTRY["default"] is GPTQQuantizer


class TestValidation:
    def test_bits_too_small(self):
        with pytest.raises(ValueError):
            GPTQQuantizer(GPTQConfig(bits=1))

    def test_bits_too_large(self):
        with pytest.raises(ValueError):
            GPTQQuantizer(GPTQConfig(bits=32))

    def test_default_constructor(self):
        q = GPTQQuantizer()
        assert q.config.bits == 4
