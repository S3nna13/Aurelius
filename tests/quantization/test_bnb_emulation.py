"""Tests for src/quantization/bnb_emulation.py — 10+ tests, CPU-only, tiny tensors."""

from __future__ import annotations

import pytest
import torch

from src.quantization.bnb_emulation import (
    BnBConfig,
    BnBQuantizer,
    OutlierDetector,
    QUANTIZATION_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_weight() -> torch.Tensor:
    """4×8 weight matrix."""
    torch.manual_seed(0)
    return torch.randn(4, 8)


@pytest.fixture()
def weight_with_outliers() -> torch.Tensor:
    """4×8 weight with explicit large outlier values."""
    w = torch.zeros(4, 8)
    w[0, 0] = 10.0   # outlier
    w[1, 3] = -8.0   # outlier
    w[2, 5] = 0.5    # normal
    w[3, 7] = 1.2    # normal
    return w


@pytest.fixture()
def default_config() -> BnBConfig:
    return BnBConfig(bits=8, threshold=6.0, has_fp16_weights=True)


# ---------------------------------------------------------------------------
# BnBConfig tests
# ---------------------------------------------------------------------------

class TestBnBConfig:
    def test_defaults(self):
        cfg = BnBConfig()
        assert cfg.bits == 8
        assert cfg.threshold == 6.0
        assert cfg.has_fp16_weights is True

    def test_custom(self):
        cfg = BnBConfig(bits=8, threshold=3.0, has_fp16_weights=False)
        assert cfg.threshold == 3.0
        assert cfg.has_fp16_weights is False


# ---------------------------------------------------------------------------
# OutlierDetector tests
# ---------------------------------------------------------------------------

class TestOutlierDetector:
    def test_detect_returns_bool_tensor(self, tiny_weight, default_config):
        det = OutlierDetector(default_config)
        mask = det.detect(tiny_weight)
        assert mask.dtype == torch.bool

    def test_detect_shape(self, tiny_weight, default_config):
        det = OutlierDetector(default_config)
        mask = det.detect(tiny_weight)
        assert mask.shape == tiny_weight.shape

    def test_detect_finds_outliers(self, weight_with_outliers):
        cfg = BnBConfig(threshold=6.0)
        det = OutlierDetector(cfg)
        mask = det.detect(weight_with_outliers)
        # Values 10.0 and -8.0 should be detected
        assert mask[0, 0].item() is True
        assert mask[1, 3].item() is True
        # Normal values should not be detected
        assert mask[2, 5].item() is False

    def test_detect_no_outliers(self, default_config):
        w = torch.ones(4, 8) * 0.5  # all below threshold
        det = OutlierDetector(default_config)
        mask = det.detect(w)
        assert not mask.any()

    def test_extract_outliers_shapes(self, weight_with_outliers, default_config):
        det = OutlierDetector(default_config)
        sparse, dense = det.extract_outliers(weight_with_outliers)
        assert sparse.shape == weight_with_outliers.shape
        assert dense.shape == weight_with_outliers.shape

    def test_extract_outliers_partition(self, weight_with_outliers, default_config):
        """sparse + dense should reconstruct original tensor."""
        det = OutlierDetector(default_config)
        sparse, dense = det.extract_outliers(weight_with_outliers)
        recon = sparse + dense
        assert torch.allclose(recon, weight_with_outliers)

    def test_extract_outliers_sparse_zeros_normal(self, weight_with_outliers, default_config):
        """Normal positions in sparse_outliers should be zero."""
        det = OutlierDetector(default_config)
        sparse, _ = det.extract_outliers(weight_with_outliers)
        # Position (2,5) = 0.5 is normal → should be zero in sparse
        assert sparse[2, 5].item() == 0.0


# ---------------------------------------------------------------------------
# BnBQuantizer tests
# ---------------------------------------------------------------------------

class TestBnBQuantizer:
    def test_quantize_output_types(self, tiny_weight, default_config):
        q = BnBQuantizer(default_config)
        q8, absmax = q.quantize(tiny_weight)
        assert q8.dtype == torch.int8
        assert absmax.dtype == torch.float32

    def test_quantize_shapes(self, tiny_weight, default_config):
        q = BnBQuantizer(default_config)
        q8, absmax = q.quantize(tiny_weight)
        assert q8.shape == tiny_weight.shape
        assert absmax.shape == (tiny_weight.shape[0],)

    def test_quantize_range(self, tiny_weight, default_config):
        q = BnBQuantizer(default_config)
        q8, _ = q.quantize(tiny_weight)
        assert q8.min() >= -127
        assert q8.max() <= 127

    def test_dequantize_shape(self, tiny_weight, default_config):
        q = BnBQuantizer(default_config)
        q8, absmax = q.quantize(tiny_weight)
        recon = q.dequantize(q8, absmax)
        assert recon.shape == tiny_weight.shape

    def test_roundtrip_error_small(self, tiny_weight, default_config):
        q = BnBQuantizer(default_config)
        q8, absmax = q.quantize(tiny_weight)
        recon = q.dequantize(q8, absmax)
        err = (recon - tiny_weight).abs().max().item()
        # 8-bit absmax error should be ≤ absmax/127 ≈ a few percent
        max_abs = tiny_weight.abs().max().item()
        assert err <= max_abs * 0.02, f"Reconstruction error {err} too large"

    def test_absmax_positive(self, tiny_weight, default_config):
        q = BnBQuantizer(default_config)
        _, absmax = q.quantize(tiny_weight)
        assert (absmax > 0).all()

    def test_registry_entry(self):
        assert "bnb_int8" in QUANTIZATION_REGISTRY
        assert QUANTIZATION_REGISTRY["bnb_int8"] is BnBQuantizer

    def test_quantize_1d_weight(self, default_config):
        """1-D weight should be handled without error."""
        w = torch.randn(8)
        q = BnBQuantizer(default_config)
        q8, absmax = q.quantize(w)
        assert q8.shape == (1, 8)

    def test_all_zeros_weight(self, default_config):
        """All-zero weight should not produce NaN."""
        w = torch.zeros(4, 8)
        q = BnBQuantizer(default_config)
        q8, absmax = q.quantize(w)
        recon = q.dequantize(q8, absmax)
        assert torch.isfinite(recon).all()
        assert torch.allclose(recon, torch.zeros(4, 8))
