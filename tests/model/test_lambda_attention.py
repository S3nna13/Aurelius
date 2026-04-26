"""Unit tests for LambdaAttention (Bello 2021)."""

from __future__ import annotations

import time

import pytest
import torch

from src.model.lambda_attention import LambdaAttention

D_MODEL = 32
N_HEADS = 4
DK = 4
DV = 8  # D_MODEL == N_HEADS * DV
N_POS = 32
B = 2
S = 16


def _make(n_pos: int = N_POS) -> LambdaAttention:
    return LambdaAttention(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim_key=DK,
        head_dim_value=DV,
        n_positions=n_pos,
    )


def test_output_shape():
    mod = _make()
    x = torch.randn(B, S, D_MODEL)
    y = mod(x)
    assert y.shape == (B, S, D_MODEL)


def test_dtype_preserved():
    mod = _make().to(torch.float32)
    x = torch.randn(B, S, D_MODEL, dtype=torch.float32)
    y = mod(x)
    assert y.dtype == torch.float32


def test_gradient_flow():
    mod = _make()
    x = torch.randn(B, S, D_MODEL, requires_grad=True)
    y = mod(x)
    y.pow(2).mean().backward()
    assert x.grad is not None
    for name, p in mod.named_parameters():
        assert p.grad is not None, f"no grad on {name}"


def test_determinism():
    torch.manual_seed(42)
    m1 = _make()
    torch.manual_seed(42)
    m2 = _make()
    x = torch.randn(B, S, D_MODEL)
    m1.eval()
    m2.eval()
    y1 = m1(x)
    y2 = m2(x)
    assert torch.allclose(y1, y2)


def test_no_nan_inf():
    mod = _make()
    x = torch.randn(B, S, D_MODEL)
    y = mod(x)
    assert torch.isfinite(y).all()


def test_s_le_n_positions():
    mod = _make(n_pos=64)
    for s in (2, 8, 32, 64):
        x = torch.randn(B, s, D_MODEL)
        y = mod(x)
        assert y.shape == (B, s, D_MODEL)


def test_s_gt_n_positions_raises():
    mod = _make(n_pos=16)
    x = torch.randn(B, 17, D_MODEL)
    with pytest.raises(ValueError, match="exceeds n_positions"):
        mod(x)


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        LambdaAttention(d_model=33, n_heads=4, head_dim_key=4, head_dim_value=8, n_positions=16)
    with pytest.raises(ValueError):
        LambdaAttention(d_model=32, n_heads=0, head_dim_key=4, head_dim_value=8, n_positions=16)
    with pytest.raises(ValueError):
        LambdaAttention(d_model=32, n_heads=4, head_dim_key=4, head_dim_value=8, n_positions=0)


def test_edge_batch1_s1():
    mod = _make(n_pos=4).eval()
    x = torch.randn(1, 1, D_MODEL)
    y = mod(x)
    assert y.shape == (1, 1, D_MODEL)
    assert torch.isfinite(y).all()


def test_long_sequence_runs_fast():
    n = 1024
    mod = LambdaAttention(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim_key=DK,
        head_dim_value=DV,
        n_positions=n,
    ).eval()
    x = torch.randn(1, n, D_MODEL)
    with torch.no_grad():
        mod(x)
        t0 = time.perf_counter()
        mod(x)
        t1 = time.perf_counter()
    assert (t1 - t0) < 1.0, f"forward took {t1 - t0:.3f}s for S=1024"


def test_sub_quadratic_scaling():
    mod_small = LambdaAttention(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        head_dim_key=DK,
        head_dim_value=DV,
        n_positions=512,
    ).eval()
    x128 = torch.randn(1, 128, D_MODEL)
    x512 = torch.randn(1, 512, D_MODEL)
    with torch.no_grad():
        mod_small(x128)
        mod_small(x512)
        iters = 5
        t0 = time.perf_counter()
        for _ in range(iters):
            mod_small(x128)
        t_small = (time.perf_counter() - t0) / iters
        t0 = time.perf_counter()
        for _ in range(iters):
            mod_small(x512)
        t_large = (time.perf_counter() - t0) / iters
    ratio = t_large / max(t_small, 1e-6)
    assert ratio < 12.0, f"S=512 / S=128 ratio was {ratio:.2f}; expected sub-quadratic."


def test_train_vs_eval_differs_due_to_bn():
    torch.manual_seed(0)
    mod = _make()
    # Break the zero-init on the output projection so BN effects are visible.
    with torch.no_grad():
        mod.to_out.weight.copy_(torch.randn_like(mod.to_out.weight) * 0.1)
    x = torch.randn(B, S, D_MODEL)
    # Warm up BN running stats on *different* data so running stats diverge
    # from the current batch statistics.
    mod.train()
    for _ in range(5):
        _ = mod(torch.randn(B, S, D_MODEL))
    y_train = mod(x)
    mod.eval()
    y_eval = mod(x)
    assert not torch.allclose(y_train, y_eval, atol=1e-5)


def test_positional_lambda_trainable():
    mod = _make()
    assert mod.pos_embed.requires_grad
    x = torch.randn(B, S, D_MODEL)
    y = mod(x)
    y.sum().backward()
    assert mod.pos_embed.grad is not None
    assert mod.pos_embed.grad.shape == mod.pos_embed.shape


def test_zero_init_output_is_zero():
    mod = _make().eval()
    x = torch.randn(B, S, D_MODEL)
    y = mod(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=0.0)
