"""Tests for Manifold-Constrained Hyper-Connections (mHC)."""

import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.mhc import ManifoldConstrainedHyperConnection, MHCLayer, _sinkhorn_knopp


def test_sinkhorn_knopp():
    M = torch.randn(3, 3)
    result = _sinkhorn_knopp(M, steps=20)
    assert result.shape == (3, 3)
    assert torch.allclose(result.sum(dim=-1), torch.ones(3), atol=1e-4)
    assert torch.allclose(result.sum(dim=-2), torch.ones(3), atol=1e-4)
    assert (result >= 0).all()


def test_mhc_wraps_sublayer():
    d_model = 64
    sublayer = nn.Linear(d_model, d_model, bias=False)
    mhc = ManifoldConstrainedHyperConnection(d_model, n_hc=4, sinkhorn_steps=20)
    x = torch.randn(2, 8, d_model)
    out = mhc(x, sublayer)
    assert out.shape == (2, 8, d_model)
    assert not torch.isnan(out).any()


def test_mhc_gradient_flow():
    d_model = 64
    sublayer = nn.Linear(d_model, d_model, bias=False)
    mhc = ManifoldConstrainedHyperConnection(d_model, n_hc=4, sinkhorn_steps=20)
    x = torch.randn(1, 4, d_model, requires_grad=True)
    out = mhc(x, sublayer)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_mhc_output_differs_from_input():
    d_model = 64
    sublayer = nn.Linear(d_model, d_model, bias=False)
    mhc = ManifoldConstrainedHyperConnection(d_model, n_hc=4)
    x = torch.randn(1, 4, d_model)
    out = mhc(x, sublayer)
    assert not torch.allclose(out, x, atol=1e-6)


def test_mhc_layer_integration():
    class DummyAttn(nn.Module):
        def forward(self, x, freqs_cis=None, mask=None, past_kv=None):
            return x, (x, x)

    class DummyFFN(nn.Module):
        def forward(self, x):
            return x, torch.tensor(0.0)

    config = AureliusConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        mhc_enabled=True,
        mhc_expansion_factor=4,
        mhc_sinkhorn_iterations=20,
    )
    attn = DummyAttn()
    ffn = DummyFFN()
    layer = MHCLayer(attn, ffn, config.d_model)
    x = torch.randn(2, 8, config.d_model)
    out, kv, aux_loss = layer(x)
    assert out.shape == (2, 8, config.d_model)
    assert isinstance(aux_loss, torch.Tensor)
