"""Tests for src/quantization/quantization_aware_training.py — 10+ tests, CPU-only."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.quantization.quantization_aware_training import (
    FakeQuantize,
    QATConfig,
    QATWrapper,
    QUANTIZATION_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cfg8() -> QATConfig:
    return QATConfig(bits=8, symmetric=True, per_channel=False, ema_decay=0.9)


@pytest.fixture()
def cfg8_asym() -> QATConfig:
    return QATConfig(bits=8, symmetric=False)


@pytest.fixture()
def linear_4x8() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(8, 4, bias=True)


@pytest.fixture()
def tiny_input() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(2, 8)


# ---------------------------------------------------------------------------
# QATConfig tests
# ---------------------------------------------------------------------------

class TestQATConfig:
    def test_defaults(self):
        cfg = QATConfig()
        assert cfg.bits == 8
        assert cfg.symmetric is True
        assert cfg.per_channel is False
        assert cfg.ema_decay == 0.9

    def test_custom(self):
        cfg = QATConfig(bits=4, symmetric=False, per_channel=True, ema_decay=0.99)
        assert cfg.bits == 4
        assert cfg.symmetric is False
        assert cfg.per_channel is True
        assert cfg.ema_decay == 0.99


# ---------------------------------------------------------------------------
# FakeQuantize tests
# ---------------------------------------------------------------------------

class TestFakeQuantize:
    def test_output_shape(self, cfg8, tiny_input):
        fq = FakeQuantize(cfg8)
        out = fq(tiny_input)
        assert out.shape == tiny_input.shape

    def test_output_finite(self, cfg8, tiny_input):
        fq = FakeQuantize(cfg8)
        out = fq(tiny_input)
        assert torch.isfinite(out).all()

    def test_symmetric_q_range(self, cfg8):
        """Symmetric 8-bit: values should be quantised to ±127 grid."""
        fq = FakeQuantize(cfg8)
        x = torch.tensor([0.5, -0.5, 1.0, -1.0])
        out = fq(x)
        # Dequantised output should be close to input (good resolution at 8-bit)
        assert torch.allclose(out, x, atol=0.05)

    def test_asymmetric_q_range(self, cfg8_asym):
        fq = FakeQuantize(cfg8_asym)
        x = torch.randn(4, 8)
        out = fq(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_ste_gradient_passes_through(self, cfg8):
        """Backward pass should not raise and gradient should be non-zero."""
        fq = FakeQuantize(cfg8)
        x = torch.randn(4, 8, requires_grad=True)
        out = fq(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_zero_input(self, cfg8):
        fq = FakeQuantize(cfg8)
        x = torch.zeros(4, 8)
        out = fq(x)
        assert torch.allclose(out, x)


# ---------------------------------------------------------------------------
# QATWrapper tests
# ---------------------------------------------------------------------------

class TestQATWrapper:
    def test_forward_shape(self, linear_4x8, tiny_input, cfg8):
        wrapper = QATWrapper(linear_4x8, cfg8)
        out = wrapper(tiny_input)
        assert out.shape == (2, 4)

    def test_forward_finite(self, linear_4x8, tiny_input, cfg8):
        wrapper = QATWrapper(linear_4x8, cfg8)
        out = wrapper(tiny_input)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self, linear_4x8, tiny_input, cfg8):
        wrapper = QATWrapper(linear_4x8, cfg8)
        out = wrapper(tiny_input)
        loss = out.sum()
        loss.backward()
        assert linear_4x8.weight.grad is not None

    def test_fold_bn_modifies_weight(self, linear_4x8, cfg8):
        wrapper = QATWrapper(linear_4x8, cfg8)
        original_w = linear_4x8.weight.data.clone()
        out_features = linear_4x8.out_features

        bn_mean = torch.zeros(out_features)
        bn_var = torch.ones(out_features)
        bn_weight = torch.ones(out_features) * 2.0  # scale by 2
        bn_bias = torch.zeros(out_features)

        wrapper.fold_bn(bn_mean, bn_var, bn_weight, bn_bias)
        # Weight should be scaled by 2
        expected = original_w * (bn_weight / (bn_var + 1e-5).sqrt()).unsqueeze(1)
        assert torch.allclose(linear_4x8.weight.data, expected, atol=1e-4)

    def test_fold_bn_creates_bias_if_none(self, cfg8):
        """Linear without bias should get a bias parameter after fold_bn."""
        lin = nn.Linear(8, 4, bias=False)
        wrapper = QATWrapper(lin, cfg8)
        out_features = 4
        wrapper.fold_bn(
            torch.zeros(out_features),
            torch.ones(out_features),
            torch.ones(out_features),
            torch.ones(out_features),  # bn_bias = 1 → bias = 1
        )
        assert lin.bias is not None
        assert lin.bias.shape == (out_features,)

    def test_registry_entry(self):
        assert "qat" in QUANTIZATION_REGISTRY
        assert QUANTIZATION_REGISTRY["qat"] is QATWrapper

    def test_wrapper_has_weight_and_act_fq(self, linear_4x8, cfg8):
        wrapper = QATWrapper(linear_4x8, cfg8)
        assert hasattr(wrapper, "weight_fq")
        assert hasattr(wrapper, "act_fq")
        assert isinstance(wrapper.weight_fq, FakeQuantize)
        assert isinstance(wrapper.act_fq, FakeQuantize)

    def test_wrapper_train_mode(self, linear_4x8, tiny_input, cfg8):
        wrapper = QATWrapper(linear_4x8, cfg8)
        wrapper.train()
        out = wrapper(tiny_input)
        assert out.shape == (2, 4)

    def test_wrapper_no_grad(self, linear_4x8, tiny_input, cfg8):
        wrapper = QATWrapper(linear_4x8, cfg8)
        wrapper.eval()
        with torch.no_grad():
            out = wrapper(tiny_input)
        assert out.shape == (2, 4)
