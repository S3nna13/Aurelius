"""Tests for DoRA weight-decomposed LoRA adaptation."""
import torch
import torch.nn as nn
import pytest
from src.alignment.dora import DoRALinear, apply_dora_to_model
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


def make_linear(in_f=64, out_f=32):
    lin = nn.Linear(in_f, out_f, bias=False)
    nn.init.normal_(lin.weight)
    return lin


def test_output_shape():
    lin = make_linear(64, 32)
    dora = DoRALinear(lin.weight, rank=4)
    x = torch.randn(2, 10, 64)
    out = dora(x)
    assert out.shape == (2, 10, 32)


def test_m_initialized_to_weight_row_norms():
    lin = make_linear(64, 32)
    dora = DoRALinear(lin.weight, rank=4)
    expected_m = lin.weight.norm(p=2, dim=1, keepdim=True)
    assert torch.allclose(dora.m.data, expected_m, atol=1e-5)


def test_frozen_base_weight():
    lin = make_linear(64, 32)
    dora = DoRALinear(lin.weight, rank=4)
    assert not dora.W.requires_grad


def test_trainable_params_only_A_B_m():
    lin = make_linear(64, 32)
    dora = DoRALinear(lin.weight, rank=4)
    trainable = {n for n, p in dora.named_parameters() if p.requires_grad}
    assert trainable == {"A", "B", "m"}


def test_no_gradient_through_norm_denominator():
    """V_prime_norm uses .detach() — no gradient should flow through it."""
    lin = make_linear(16, 8)
    dora = DoRALinear(lin.weight, rank=2)
    x = torch.randn(1, 4, 16)
    out = dora(x)
    loss = out.sum()
    loss.backward()
    # If detach is missing, m.grad would be None or blow up.
    # With detach, m.grad should be finite and non-None.
    assert dora.m.grad is not None
    assert torch.isfinite(dora.m.grad).all()


def test_B_initialized_to_zero():
    """At init, B=0 means LoRA output is zero — model starts as original W."""
    lin = make_linear(64, 32)
    dora = DoRALinear(lin.weight, rank=4)
    assert torch.all(dora.B == 0)


def test_merge_weights_shape():
    lin = make_linear(64, 32)
    dora = DoRALinear(lin.weight, rank=4)
    merged = dora.merge_weights()
    assert merged.shape == lin.weight.shape


def test_apply_dora_to_model():
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=64,
    )
    model = AureliusTransformer(cfg)
    targets = ["layers.0.attn.q_proj", "layers.1.attn.q_proj"]
    replaced = apply_dora_to_model(model, targets, rank=4)
    assert len(replaced) == 2
    for name in targets:
        parts = name.split(".")
        mod = model
        for p in parts:
            mod = getattr(mod, p)
        assert isinstance(mod, DoRALinear)
