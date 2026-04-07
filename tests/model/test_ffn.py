"""Tests for SwiGLU FFN."""
import torch
import pytest
from src.model.config import AureliusConfig
from src.model.ffn import SwiGLUFFN


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
        d_ff=512, vocab_size=1000,
    )


def test_output_shape(small_cfg):
    ffn = SwiGLUFFN(small_cfg)
    x = torch.randn(2, 16, small_cfg.d_model)
    assert ffn(x).shape == (2, 16, small_cfg.d_model)


def test_three_weight_matrices(small_cfg):
    ffn = SwiGLUFFN(small_cfg)
    param_names = [n for n, _ in ffn.named_parameters()]
    # Should have gate_proj (W1), down_proj (W2), up_proj (W3)
    assert len(param_names) == 3


def test_no_bias(small_cfg):
    ffn = SwiGLUFFN(small_cfg)
    for name, _ in ffn.named_parameters():
        assert "bias" not in name


def test_swiglu_activation():
    """SwiGLU = SiLU(W1(x)) * W3(x) — output should differ from simple MLP."""
    cfg = AureliusConfig(n_layers=2, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64, d_ff=512, vocab_size=1000)
    ffn = SwiGLUFFN(cfg)
    x = torch.randn(1, 1, 256)
    with torch.no_grad():
        out = ffn(x)
    # Just verify it's not all zeros and has right shape
    assert out.shape == (1, 1, 256)
    assert not torch.allclose(out, torch.zeros_like(out))
