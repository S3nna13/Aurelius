"""Tests for RMSNorm."""

import torch

from src.model.rms_norm import RMSNorm


def test_output_shape():
    norm = RMSNorm(2048)
    x = torch.randn(2, 16, 2048)
    assert norm(x).shape == (2, 16, 2048)


def test_no_bias():
    norm = RMSNorm(2048)
    param_names = [n for n, _ in norm.named_parameters()]
    assert not any("bias" in n for n in param_names)


def test_weight_initialized_to_ones():
    norm = RMSNorm(64)
    assert torch.allclose(norm.weight, torch.ones(64))


def test_normalizes_large_input():
    norm = RMSNorm(256)
    # Large input — output should still be bounded
    x = torch.randn(2, 8, 256) * 1000
    out = norm(x)
    # RMS of output should be ~1 (weight is 1 by default)
    rms = out.pow(2).mean(-1).sqrt()
    assert rms.max().item() < 10.0  # sanity: not exploding


def test_dtype_preserved():
    norm = RMSNorm(128)
    x = torch.randn(1, 4, 128).half()
    out = norm(x)
    assert out.dtype == torch.float16
