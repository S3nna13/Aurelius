"""Tests for Liger kernel integration."""

from __future__ import annotations

import torch

from src.model.liger_integration import apply_liger_cross_entropy, apply_liger_kernels


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.RMSNorm(128)
        self.linear = torch.nn.Linear(128, 128)

    def forward(self, x):
        return self.ln(self.linear(x))


def test_apply_liger_kernels_returns_bool():
    model = SimpleModel()
    result = apply_liger_kernels(model, enable_rope=False, enable_fused_ce=False)
    assert isinstance(result, bool)


def test_apply_liger_cross_entropy_returns_callable_or_none():
    result = apply_liger_cross_entropy(vocab_size=1000)
    assert result is None or callable(result)


def test_model_forward_without_liger():
    model = SimpleModel()
    x = torch.randn(2, 4, 128)
    out = model(x)
    assert out.shape == (2, 4, 128)
    assert not torch.isnan(out).any()
