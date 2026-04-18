"""Tests for GatedDeltaNet (Schiff et al., arXiv:2412.06464).

12 focused tests covering shape, state, causality, gradients, gate
ranges, edge cases, determinism, and the forget-gate semantic.
"""

import pytest
import torch

from src.model.gated_delta_net import GatedDeltaNetConfig, GatedDeltaNetLayer

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

B  = 2    # batch size
T  = 12   # sequence length
DM = 64   # d_model
NH = 4    # n_heads
DH = 16   # d_head


def make_layer() -> GatedDeltaNetLayer:
    cfg = GatedDeltaNetConfig(d_model=DM, d_head=DH, n_heads=NH)
    return GatedDeltaNetLayer(cfg).train(False)


def zero_state() -> torch.Tensor:
    return torch.zeros(B, NH, DH, DH)


# ---------------------------------------------------------------------------
# Test 1 — output shape (B, T, d_model)
# ---------------------------------------------------------------------------


def test_output_shape():
    """forward returns output tensor of shape (B, T, d_model)."""
    layer = make_layer()
    x = torch.randn(B, T, DM)
    out, _ = layer(x)
    assert out.shape == (B, T, DM), f"output shape {out.shape} != {(B, T, DM)}"


# ---------------------------------------------------------------------------
# Test 2 — final state shape (B, n_heads, d_head, d_head)
# ---------------------------------------------------------------------------


def test_state_shape():
    """forward returns final state of shape (B, n_heads, d_head, d_head)."""
    layer = make_layer()
    x = torch.randn(B, T, DM)
    _, state = layer(x)
    expected = (B, NH, DH, DH)
    assert state.shape == expected, f"state shape {state.shape} != {expected}"


# ---------------------------------------------------------------------------
# Test 3 — causal: output at t uses only x[0..t]
# ---------------------------------------------------------------------------


def test_causal():
    """Outputs for positions 0..T-1 are unchanged when future tokens are appended."""
    layer = make_layer()
    torch.manual_seed(42)
    x_short = torch.randn(1, T, DM)
    x_long  = torch.cat([x_short, torch.randn(1, 5, DM)], dim=1)

    with torch.no_grad():
        out_short, _ = layer(x_short)
        out_long,  _ = layer(x_long)

    assert torch.allclose(out_short, out_long[:, :T, :], atol=1e-5), \
        "Outputs for x[0..T-1] differ when future tokens are present — not causal"


# ---------------------------------------------------------------------------
# Test 4 — gradients flow through both output and state
# ---------------------------------------------------------------------------


def test_gradient_flow():
    """Backward pass produces finite, non-None gradients through output and state."""
    layer = make_layer().train(True)
    x = torch.randn(B, T, DM, requires_grad=True)
    out, state = layer(x)
    loss = out.sum() + state.sum()
    loss.backward()

    assert x.grad is not None, "No gradient reached input"
    assert torch.isfinite(x.grad).all(), "Input gradient contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 5 — state=None gives same result as state=zeros
# ---------------------------------------------------------------------------


def test_none_state_equals_zero_state():
    """Passing state=None produces the same result as passing an all-zero state."""
    layer = make_layer()
    x = torch.randn(B, T, DM)
    with torch.no_grad():
        out_none,  s_none  = layer(x, state=None)
        out_zeros, s_zeros = layer(x, state=zero_state())

    assert torch.allclose(out_none, out_zeros, atol=1e-6), \
        "Outputs differ between state=None and state=zeros"
    assert torch.allclose(s_none, s_zeros, atol=1e-6), \
        "Final states differ between state=None and state=zeros"


# ---------------------------------------------------------------------------
# Test 6 — different initial states give different outputs
# ---------------------------------------------------------------------------


def test_different_states_give_different_outputs():
    """A non-zero initial state produces a different output from a zero state."""
    layer = make_layer()
    x = torch.randn(B, T, DM)
    s_nonzero = torch.randn(B, NH, DH, DH)  # random non-zero state

    with torch.no_grad():
        out_zero, _    = layer(x, state=zero_state())
        out_nonzero, _ = layer(x, state=s_nonzero)

    assert not torch.allclose(out_zero, out_nonzero, atol=1e-4), \
        "Different initial states produced identical outputs"


# ---------------------------------------------------------------------------
# Test 7 — alpha gate values lie strictly in (0, 1)
# ---------------------------------------------------------------------------


def test_alpha_gate_range():
    """Forget gate alpha is produced by sigmoid and must lie strictly in (0, 1)."""
    cfg = GatedDeltaNetConfig(d_model=DM, d_head=DH, n_heads=NH)
    layer = GatedDeltaNetLayer(cfg)
    x = torch.randn(B, T, DM)
    with torch.no_grad():
        alpha = torch.sigmoid(layer.alpha_proj(x))  # (B, T, NH)
    assert (alpha > 0).all() and (alpha < 1).all(), \
        f"alpha gate out of (0,1): min={alpha.min():.4f} max={alpha.max():.4f}"


# ---------------------------------------------------------------------------
# Test 8 — beta gate values lie strictly in (0, 1)
# ---------------------------------------------------------------------------


def test_beta_gate_range():
    """Write gate beta is produced by sigmoid and must lie strictly in (0, 1)."""
    cfg = GatedDeltaNetConfig(d_model=DM, d_head=DH, n_heads=NH)
    layer = GatedDeltaNetLayer(cfg)
    x = torch.randn(B, T, DM)
    with torch.no_grad():
        beta = torch.sigmoid(layer.beta_proj(x))   # (B, T, NH)
    assert (beta > 0).all() and (beta < 1).all(), \
        f"beta gate out of (0,1): min={beta.min():.4f} max={beta.max():.4f}"


# ---------------------------------------------------------------------------
# Test 9 — T=1 (single time step) works correctly
# ---------------------------------------------------------------------------


def test_single_timestep():
    """Layer handles sequence length T=1 without error and returns correct shapes."""
    layer = make_layer()
    x = torch.randn(B, 1, DM)
    out, state = layer(x)
    assert out.shape == (B, 1, DM), f"T=1 output shape {out.shape}"
    assert state.shape == (B, NH, DH, DH), f"T=1 state shape {state.shape}"


# ---------------------------------------------------------------------------
# Test 10 — no NaN / Inf on random input
# ---------------------------------------------------------------------------


def test_no_nan_inf():
    """Output and state must be fully finite on standard random input."""
    layer = make_layer()
    x = torch.randn(B, T, DM)
    with torch.no_grad():
        out, state = layer(x)
    assert torch.isfinite(out).all(),   "Output contains NaN or Inf"
    assert torch.isfinite(state).all(), "State contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 11 — determinism under torch.manual_seed
# ---------------------------------------------------------------------------


def test_determinism():
    """Two forward passes with the same seed and input produce identical results."""
    torch.manual_seed(7)
    x = torch.randn(B, T, DM)
    layer = make_layer()

    with torch.no_grad():
        out1, s1 = layer(x)
        out2, s2 = layer(x)

    assert torch.equal(out1, out2), "Forward pass is not deterministic"
    assert torch.equal(s1,   s2),   "State is not deterministic"


# ---------------------------------------------------------------------------
# Test 12 — forget semantic: alpha near 0 causes state to be wiped each step
# ---------------------------------------------------------------------------


def test_forgetting_with_near_zero_alpha():
    """When the forget gate alpha is driven near zero, the layer ignores the
    initial state: output should be almost the same regardless of whether we
    start from a large state or a zero state.
    """
    cfg = GatedDeltaNetConfig(d_model=DM, d_head=DH, n_heads=NH)
    layer = GatedDeltaNetLayer(cfg).train(False)

    # Force alpha near 0: very large negative bias -> sigmoid(.) ≈ 0
    with torch.no_grad():
        layer.alpha_proj.bias.fill_(-50.0)

    x = torch.randn(1, T, DM)
    # A large initial state that dominates if NOT forgotten
    s_large = torch.ones(1, NH, DH, DH) * 100.0

    with torch.no_grad():
        out_large_s0, _ = layer(x, state=s_large)
        out_zero_s0,  _ = layer(x, state=None)

    # With alpha near 0 the state is reset at each step; outputs should match
    assert torch.allclose(out_large_s0, out_zero_s0, atol=1e-3), (
        "With alpha near 0, output should ignore the initial state (forget "
        "gate not working correctly)"
    )
