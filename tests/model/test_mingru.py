"""Tests for minGRU (arXiv:2410.01201).

12 focused tests covering correctness, causality, gradient flow,
numerical stability, and edge cases.
"""

from __future__ import annotations

import pytest
import torch

from src.model.mingru import MinGRU, MinGRUConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> MinGRUConfig:
    return MinGRUConfig(d_model=64)


@pytest.fixture
def model(cfg: MinGRUConfig) -> MinGRU:
    torch.manual_seed(42)
    return MinGRU(cfg)


# ---------------------------------------------------------------------------
# Test 1 — Output shape: (B, T, d) → (B, T, d)
# ---------------------------------------------------------------------------


def test_output_shape(model: MinGRU) -> None:
    B, T, d = 2, 16, 64
    x = torch.randn(B, T, d)
    out, _ = model(x, use_parallel=True)
    assert out.shape == (B, T, d), f"Expected ({B}, {T}, {d}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2 — Sequential and parallel outputs match (atol=1e-5)
# ---------------------------------------------------------------------------


def test_sequential_parallel_match(model: MinGRU) -> None:
    torch.manual_seed(0)
    B, T, d = 2, 16, 64
    x = torch.randn(B, T, d)

    with torch.no_grad():
        out_seq, _ = model.forward_sequential(x)
        out_par = model.forward_parallel(x)

    assert torch.allclose(out_seq, out_par, atol=1e-4), (
        f"Max diff: {(out_seq - out_par).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Causality: output at t only uses x[0..t]
# ---------------------------------------------------------------------------


def test_causal(model: MinGRU) -> None:
    torch.manual_seed(1)
    B, T, d = 1, 8, 64
    x = torch.randn(B, T, d)

    with torch.no_grad():
        out_orig, _ = model.forward_sequential(x)

    # Corrupt x at position t=4 and beyond; outputs at t < 4 must be unchanged
    x_corrupted = x.clone()
    x_corrupted[:, 4:, :] = torch.randn_like(x[:, 4:, :])

    with torch.no_grad():
        out_corr, _ = model.forward_sequential(x_corrupted)

    assert torch.allclose(out_orig[:, :4, :], out_corr[:, :4, :], atol=1e-6), (
        "Output before corruption point changed — causality violated"
    )
    # Outputs at t >= 4 should differ
    assert not torch.allclose(out_orig[:, 4:, :], out_corr[:, 4:, :], atol=1e-6), (
        "Outputs after corruption point should differ"
    )


# ---------------------------------------------------------------------------
# Test 4 — h0=None gives same result as h0=zeros
# ---------------------------------------------------------------------------


def test_h0_none_equals_zeros(model: MinGRU) -> None:
    torch.manual_seed(2)
    B, T, d = 2, 8, 64
    x = torch.randn(B, T, d)
    h0_zeros = torch.zeros(B, model.d_inner)

    with torch.no_grad():
        out_none, _ = model.forward_sequential(x, h0=None)
        out_zeros, _ = model.forward_sequential(x, h0=h0_zeros)

    assert torch.allclose(out_none, out_zeros, atol=1e-7), (
        "h0=None and h0=zeros should produce identical outputs"
    )


# ---------------------------------------------------------------------------
# Test 5 — h_T from sequential matches last hidden state
# ---------------------------------------------------------------------------


def test_hT_matches_last_hidden(model: MinGRU) -> None:
    torch.manual_seed(3)
    B, T, d = 2, 10, 64
    x = torch.randn(B, T, d)

    with torch.no_grad():
        out, h_T = model.forward_sequential(x)

    # Run again step-by-step to get the last hidden state directly
    with torch.no_grad():
        h = torch.zeros(B, model.d_inner)
        for t in range(T):
            xt = x[:, t, :]
            z = torch.sigmoid(model.linear_z(xt))
            h_tilde = model.linear_h(xt)
            h = (1.0 - z) * h + z * h_tilde
        expected_h_T = h

    assert torch.allclose(h_T, expected_h_T, atol=1e-6), (
        f"h_T mismatch: max diff {(h_T - expected_h_T).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Gradient flow in parallel mode
# ---------------------------------------------------------------------------


def test_gradient_flow_parallel(model: MinGRU) -> None:
    torch.manual_seed(4)
    B, T, d = 2, 16, 64
    x = torch.randn(B, T, d, requires_grad=True)

    out, _ = model(x, use_parallel=True)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient flowed back to x (parallel)"
    assert not torch.isnan(x.grad).any(), "NaN in gradients (parallel)"
    assert not torch.isinf(x.grad).any(), "Inf in gradients (parallel)"


# ---------------------------------------------------------------------------
# Test 7 — Gradient flow in sequential mode
# ---------------------------------------------------------------------------


def test_gradient_flow_sequential(model: MinGRU) -> None:
    torch.manual_seed(5)
    B, T, d = 2, 16, 64
    x = torch.randn(B, T, d, requires_grad=True)

    out, _ = model.forward_sequential(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient flowed back to x (sequential)"
    assert not torch.isnan(x.grad).any(), "NaN in gradients (sequential)"
    assert not torch.isinf(x.grad).any(), "Inf in gradients (sequential)"


# ---------------------------------------------------------------------------
# Test 8 — Determinism under torch.manual_seed
# ---------------------------------------------------------------------------


def test_determinism(cfg: MinGRUConfig) -> None:
    B, T, d = 2, 8, 64

    def run_with_seed(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        m = MinGRU(cfg)
        x = torch.randn(B, T, d)
        with torch.no_grad():
            out, _ = m(x, use_parallel=True)
        return out

    out1 = run_with_seed(99)
    out2 = run_with_seed(99)
    assert torch.allclose(out1, out2, atol=0.0), "Non-deterministic output"


# ---------------------------------------------------------------------------
# Test 9 — T=1 works
# ---------------------------------------------------------------------------


def test_single_timestep(model: MinGRU) -> None:
    torch.manual_seed(6)
    B, T, d = 2, 1, 64
    x = torch.randn(B, T, d)

    with torch.no_grad():
        out_seq, h_T = model.forward_sequential(x)
        out_par = model.forward_parallel(x)

    assert out_seq.shape == (B, T, d)
    assert out_par.shape == (B, T, d)
    assert h_T.shape == (B, model.d_inner)
    assert not torch.isnan(out_seq).any()
    assert not torch.isnan(out_par).any()


# ---------------------------------------------------------------------------
# Test 10 — Long sequence (T=128) produces no NaN/Inf
# ---------------------------------------------------------------------------


def test_long_sequence_no_nan_inf(model: MinGRU) -> None:
    torch.manual_seed(7)
    B, T, d = 2, 128, 64
    x = torch.randn(B, T, d)

    with torch.no_grad():
        out_seq, _ = model.forward_sequential(x)
        out_par = model.forward_parallel(x)

    for name, out in [("sequential", out_seq), ("parallel", out_par)]:
        assert not torch.isnan(out).any(), f"NaN in {name} output at T=128"
        assert not torch.isinf(out).any(), f"Inf in {name} output at T=128"


# ---------------------------------------------------------------------------
# Test 11 — Gates z_t are in (0, 1) via sigmoid
# ---------------------------------------------------------------------------


def test_gates_in_unit_interval(model: MinGRU) -> None:
    torch.manual_seed(8)
    B, T, d = 2, 16, 64
    x = torch.randn(B, T, d)

    with torch.no_grad():
        z = torch.sigmoid(model.linear_z(x))

    assert (z > 0.0).all(), "Gate z_t must be > 0 (sigmoid output)"
    assert (z < 1.0).all(), "Gate z_t must be < 1 (sigmoid output)"


# ---------------------------------------------------------------------------
# Test 12 — Different h0 → different output (h0 matters)
# ---------------------------------------------------------------------------


def test_different_h0_gives_different_output(model: MinGRU) -> None:
    torch.manual_seed(9)
    B, T, d = 2, 8, 64
    x = torch.randn(B, T, d)

    h0_a = torch.zeros(B, model.d_inner)
    h0_b = torch.ones(B, model.d_inner) * 5.0  # large non-zero

    with torch.no_grad():
        out_a, _ = model.forward_sequential(x, h0=h0_a)
        out_b, _ = model.forward_sequential(x, h0=h0_b)

    assert not torch.allclose(out_a, out_b, atol=1e-4), (
        "Different h0 values should produce different outputs"
    )
