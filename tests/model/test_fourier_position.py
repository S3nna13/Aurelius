"""Tests for FourierPositionEncoding."""

import torch
import pytest
from src.model.fourier_position import FourierPEConfig, FourierPositionEncoding


@pytest.fixture
def cfg():
    return FourierPEConfig(d_model=64, max_seq_len=512, learnable=True)


@pytest.fixture
def enc(cfg):
    return FourierPositionEncoding(cfg)


def test_fourier_pe_config_instantiates():
    cfg = FourierPEConfig(d_model=128)
    assert cfg.d_model == 128
    assert cfg.n_freqs == 64
    assert cfg.learnable is True


def test_instantiates_learnable(cfg):
    enc = FourierPositionEncoding(cfg)
    assert isinstance(enc, FourierPositionEncoding)


def test_instantiates_fixed():
    cfg = FourierPEConfig(d_model=64, learnable=False)
    enc = FourierPositionEncoding(cfg)
    assert isinstance(enc, FourierPositionEncoding)


def test_compute_encodings_shape(enc, cfg):
    T = 16
    out = enc._compute_encodings(T, torch.device("cpu"))
    assert out.shape == (T, cfg.d_model)


def test_encodings_are_finite(enc, cfg):
    out = enc._compute_encodings(32, torch.device("cpu"))
    assert torch.isfinite(out).all()


def test_different_positions_get_different_encodings(enc, cfg):
    out = enc._compute_encodings(8, torch.device("cpu"))
    # All position vectors should differ from at least some other positions
    # Compare first and last row
    assert not torch.allclose(out[0], out[-1])


def test_forward_returns_same_shape(enc, cfg):
    B, T = 3, 10
    x = torch.randn(B, T, cfg.d_model)
    out = enc(x)
    assert out.shape == x.shape


def test_frequencies_are_parameters_when_learnable(cfg):
    enc = FourierPositionEncoding(cfg)
    param_names = [name for name, _ in enc.named_parameters()]
    assert "frequencies" in param_names


def test_gradient_flows_through_frequencies(cfg):
    enc = FourierPositionEncoding(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    out = enc(x)
    out.sum().backward()
    assert enc.frequencies.grad is not None
    assert enc.frequencies.grad.abs().sum() > 0


def test_works_with_t1(cfg):
    enc = FourierPositionEncoding(cfg)
    x = torch.randn(1, 1, cfg.d_model)
    out = enc(x)
    assert out.shape == (1, 1, cfg.d_model)
    assert torch.isfinite(out).all()
