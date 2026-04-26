"""Tests for src/model/ssm_layer.py — Mamba-inspired SSM layer."""

from __future__ import annotations

import torch

from src.model.ssm_layer import (
    SSMBlock,
    SSMConfig,
    SSMLayer,
    count_ssm_parameters,
    init_A_matrix,
    selective_scan,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

B = 2  # batch size
T = 8  # sequence length
D = 32  # d_model (smaller for speed)

SMALL_CFG = SSMConfig(d_model=D, d_state=8, dt_rank=2, expand=2)


# ---------------------------------------------------------------------------
# 1. SSMConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = SSMConfig()
    assert cfg.d_model == 64
    assert cfg.d_state == 16
    assert cfg.expand == 2


# ---------------------------------------------------------------------------
# 2 and 3. init_A_matrix
# ---------------------------------------------------------------------------


def test_init_A_matrix_shape():
    d_model, d_state = 16, 8
    log_A = init_A_matrix(d_model, d_state)
    assert log_A.shape == (d_model, d_state)


def test_init_A_matrix_range():
    """log_A values should be in (0, 1) (U(0,1) initialisation before negation)."""
    log_A = init_A_matrix(64, 16)
    assert log_A.min().item() >= 0.0
    assert log_A.max().item() <= 1.0


# ---------------------------------------------------------------------------
# 4 to 6. selective_scan
# ---------------------------------------------------------------------------


def test_selective_scan_shape():
    N = 8
    u = torch.randn(B, T, D)
    A = -torch.exp(torch.rand(D, N))
    Bm = torch.randn(B, T, N)
    C = torch.randn(B, T, N)
    delta = torch.ones(B, T, D) * 0.01

    out = selective_scan(u, A, Bm, C, delta)
    assert out.shape == (B, T, D)


def test_selective_scan_causal():
    """Output at position t must not depend on inputs at positions > t."""
    torch.manual_seed(0)
    N = 8
    A = -torch.exp(torch.rand(D, N))

    x1 = torch.randn(B, T, D)
    x2 = x1.clone()
    # Perturb the second half of x2
    x2[:, T // 2 :, :] = torch.randn(B, T - T // 2, D)

    Bm = torch.randn(B, T, N)
    C = torch.randn(B, T, N)
    delta = torch.ones(B, T, D) * 0.01

    out1 = selective_scan(x1, A, Bm, C, delta)
    out2 = selective_scan(x2, A, Bm, C, delta)

    # First T//2 outputs must be identical
    assert torch.allclose(out1[:, : T // 2, :], out2[:, : T // 2, :], atol=1e-6), (
        "selective_scan is not causal"
    )


def test_selective_scan_seq_len_1():
    N = 8
    u = torch.randn(B, 1, D)
    A = -torch.exp(torch.rand(D, N))
    Bm = torch.randn(B, 1, N)
    C = torch.randn(B, 1, N)
    delta = torch.ones(B, 1, D) * 0.01

    out = selective_scan(u, A, Bm, C, delta)
    assert out.shape == (B, 1, D)


# ---------------------------------------------------------------------------
# 7 to 10. SSMLayer
# ---------------------------------------------------------------------------


def test_ssm_layer_output_shape():
    layer = SSMLayer(SMALL_CFG)
    x = torch.randn(B, T, D)
    out = layer(x)
    assert out.shape == (B, T, D)


def test_ssm_layer_residual():
    """Output should differ from the raw input (the layer does something)."""
    torch.manual_seed(42)
    layer = SSMLayer(SMALL_CFG)
    x = torch.randn(B, T, D)
    out = layer(x)
    assert not torch.allclose(out, x, atol=1e-4), "SSMLayer output is identical to input"


def test_ssm_layer_causal():
    """SSMLayer forward pass should be causal."""
    torch.manual_seed(7)
    layer = SSMLayer(SMALL_CFG)
    layer.eval()

    x1 = torch.randn(B, T, D)
    x2 = x1.clone()
    x2[:, T // 2 :, :] = torch.randn(B, T - T // 2, D)

    with torch.no_grad():
        out1 = layer(x1)
        out2 = layer(x2)

    assert torch.allclose(out1[:, : T // 2, :], out2[:, : T // 2, :], atol=1e-5), (
        "SSMLayer forward is not causal"
    )


def test_ssm_layer_different_seq_lens():
    """SSMLayer should handle T=1, T=8, T=16 without errors."""
    layer = SSMLayer(SMALL_CFG)
    for t in (1, 8, 16):
        x = torch.randn(B, t, D)
        out = layer(x)
        assert out.shape == (B, t, D), f"Shape mismatch for T={t}"


# ---------------------------------------------------------------------------
# 11 and 12. SSMBlock
# ---------------------------------------------------------------------------


def test_ssm_block_shape():
    block = SSMBlock(SMALL_CFG)
    x = torch.randn(B, T, D)
    out = block(x)
    assert out.shape == (B, T, D)


def test_ssm_block_residual():
    """SSMBlock output should differ from input."""
    torch.manual_seed(99)
    block = SSMBlock(SMALL_CFG)
    x = torch.randn(B, T, D)
    out = block(x)
    assert not torch.allclose(out, x, atol=1e-4), "SSMBlock output is identical to input"


# ---------------------------------------------------------------------------
# 13 and 14. count_ssm_parameters
# ---------------------------------------------------------------------------


def test_count_ssm_parameters_positive():
    n = count_ssm_parameters(SMALL_CFG)
    assert n > 0, "Parameter count should be positive"


def test_count_ssm_parameters_consistent():
    """count_ssm_parameters must match actual SSMLayer parameter count."""
    layer = SSMLayer(SMALL_CFG)
    actual = sum(p.numel() for p in layer.parameters())
    reported = count_ssm_parameters(SMALL_CFG)
    assert reported == actual, (
        f"count_ssm_parameters returned {reported} but actual count is {actual}"
    )


# ---------------------------------------------------------------------------
# 15. Gradient flow
# ---------------------------------------------------------------------------


def test_ssm_layer_gradient_flows():
    """loss.backward() should complete without errors."""
    layer = SSMLayer(SMALL_CFG)
    x = torch.randn(B, T, D, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()  # must not raise
    assert x.grad is not None, "No gradient flowed back to input"
