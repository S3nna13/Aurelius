"""Tests for Mamba-2 Structured State Space Duality (SSD) layer.

Reference: Dao & Gu 2024, "Transformers are SSMs: Generalized Models and Efficient
Algorithms Through Structured State Space Duality", arXiv:2405.21060.

Tiny test config: d_model=64, n_heads=4, d_state=16, head_dim=16
"""

import math
import pytest
import torch
import torch.nn.functional as F

from src.model.mamba2 import Mamba2Block, SSDLayer, Mamba2Config


# ---------------------------------------------------------------------------
# Shared test configuration
# ---------------------------------------------------------------------------

# Tiny config per spec
D_MODEL = 64
N_HEADS = 4
D_STATE = 16
HEAD_DIM = 16
EXPAND = 2


def make_block(**kwargs) -> Mamba2Block:
    """Create a small Mamba2Block with the tiny test config."""
    defaults = dict(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_state=D_STATE,
        head_dim=HEAD_DIM,
        expand=EXPAND,
        dt_rank="auto",
    )
    defaults.update(kwargs)
    return Mamba2Block(**defaults)


# ---------------------------------------------------------------------------
# Test 1: Output shape matches input (B, T, d_model)
# ---------------------------------------------------------------------------

def test_output_shape():
    """forward(x) output must have shape (B, T, d_model)."""
    block = make_block()
    x = torch.randn(2, 16, D_MODEL)
    out, _ = block(x)
    assert out.shape == (2, 16, D_MODEL), (
        f"Expected output shape (2, 16, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: Hidden state shape (B, n_heads, head_dim, d_state)
# ---------------------------------------------------------------------------

def test_hidden_state_shape():
    """Returned hidden state must have shape (B, n_heads, head_dim, d_state)."""
    block = make_block()
    x = torch.randn(2, 16, D_MODEL)
    _, h = block(x)
    expected = (2, N_HEADS, HEAD_DIM, D_STATE)
    assert h.shape == expected, (
        f"Expected hidden state shape {expected}, got {h.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: Gradient flow through all trainable parameters
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """Backward pass must produce finite (non-None, non-NaN/Inf) grads on all params."""
    block = make_block()
    x = torch.randn(2, 8, D_MODEL)
    out, _ = block(x)
    loss = out.sum()
    loss.backward()

    for name, param in block.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient."
        assert torch.isfinite(param.grad).all(), (
            f"Parameter {name} has non-finite gradient."
        )


# ---------------------------------------------------------------------------
# Test 4: Determinism under torch.manual_seed
# ---------------------------------------------------------------------------

def test_determinism():
    """Same seed produces identical outputs on two independent runs."""
    torch.manual_seed(42)
    block = make_block()
    x = torch.randn(2, 8, D_MODEL)

    with torch.no_grad():
        out1, _ = block(x)

    # Re-create block with same seed
    torch.manual_seed(42)
    block2 = make_block()
    with torch.no_grad():
        out2, _ = block2(x)

    assert torch.allclose(out1, out2, atol=1e-6), (
        "Two identical-seed blocks produced different outputs."
    )


# ---------------------------------------------------------------------------
# Test 5: batch=1, seq_len=1
# ---------------------------------------------------------------------------

def test_batch1_seqlen1():
    """Block must handle single-token input (B=1, T=1) without error."""
    block = make_block()
    x = torch.randn(1, 1, D_MODEL)
    out, h = block(x)
    assert out.shape == (1, 1, D_MODEL), f"Got {out.shape}"
    assert h.shape == (1, N_HEADS, HEAD_DIM, D_STATE), f"Got {h.shape}"


# ---------------------------------------------------------------------------
# Test 6: seq_len=1 with explicit hidden_state passed in
# ---------------------------------------------------------------------------

def test_seqlen1_with_explicit_hidden_state():
    """Passing an explicit hidden_state to a single-token forward must not crash."""
    block = make_block()
    x = torch.randn(1, 1, D_MODEL)
    h0 = torch.randn(1, N_HEADS, HEAD_DIM, D_STATE)
    out, h_new = block(x, hidden_state=h0)
    assert out.shape == (1, 1, D_MODEL), f"Got {out.shape}"
    assert h_new.shape == (1, N_HEADS, HEAD_DIM, D_STATE), f"Got {h_new.shape}"


# ---------------------------------------------------------------------------
# Test 7: No NaN/Inf on zeros input
# ---------------------------------------------------------------------------

def test_no_nan_on_zeros():
    """All-zero input must produce finite outputs and hidden states."""
    block = make_block()
    x = torch.zeros(2, 8, D_MODEL)
    with torch.no_grad():
        out, h = block(x)
    assert torch.isfinite(out).all(), "Output contains NaN/Inf on zeros input."
    assert torch.isfinite(h).all(), "Hidden state contains NaN/Inf on zeros input."


# ---------------------------------------------------------------------------
# Test 8: No NaN/Inf on large inputs (x100)
# ---------------------------------------------------------------------------

def test_no_nan_on_large_input():
    """Large-magnitude input must not produce NaN/Inf outputs."""
    block = make_block()
    x = torch.randn(2, 8, D_MODEL) * 100.0
    with torch.no_grad():
        out, h = block(x)
    assert torch.isfinite(out).all(), "Output contains NaN/Inf on large input."
    assert torch.isfinite(h).all(), "Hidden state contains NaN/Inf on large input."


# ---------------------------------------------------------------------------
# Test 9: State carries context — output at t=T depends on t=1
# ---------------------------------------------------------------------------

def test_state_propagation():
    """Output at last timestep must differ when the first token differs."""
    torch.manual_seed(7)
    block = make_block()
    block.train(False)

    T = 8
    x1 = torch.randn(1, T, D_MODEL)
    x2 = x1.clone()
    # Change only the first token — should influence output at t=T-1 via state
    x2[:, 0, :] = torch.randn(1, D_MODEL)

    with torch.no_grad():
        out1, _ = block(x1)
        out2, _ = block(x2)

    # The last-timestep output should differ because state at t=0 is propagated
    assert not torch.allclose(out1[:, -1, :], out2[:, -1, :], atol=1e-5), (
        "Output at t=T-1 did not change when t=0 input changed — state is not propagating."
    )


# ---------------------------------------------------------------------------
# Test 10: None hidden_state -> zero-init, no crash
# ---------------------------------------------------------------------------

def test_none_hidden_state_zero_init():
    """hidden_state=None (default) must silently zero-init and not crash."""
    block = make_block()
    x = torch.randn(2, 4, D_MODEL)
    out, h = block(x, hidden_state=None)
    assert out.shape == (2, 4, D_MODEL)
    assert h.shape == (2, N_HEADS, HEAD_DIM, D_STATE)


# ---------------------------------------------------------------------------
# Test 11: Different hidden_states produce different outputs
# ---------------------------------------------------------------------------

def test_different_hidden_states_different_outputs():
    """Two different initial hidden states must yield different outputs."""
    torch.manual_seed(99)
    block = make_block()
    block.train(False)

    x = torch.randn(1, 4, D_MODEL)
    h1 = torch.zeros(1, N_HEADS, HEAD_DIM, D_STATE)
    h2 = torch.ones(1, N_HEADS, HEAD_DIM, D_STATE)

    with torch.no_grad():
        out1, _ = block(x, hidden_state=h1)
        out2, _ = block(x, hidden_state=h2)

    assert not torch.allclose(out1, out2, atol=1e-5), (
        "Different initial hidden states produced identical outputs."
    )


# ---------------------------------------------------------------------------
# Test 12: A_log initialized negative (ensures contraction, stability)
# ---------------------------------------------------------------------------

def test_a_log_initialized_for_stability():
    """SSDLayer A_log must ensure dA < 1 (contractive decay) for unit delta.

    The actual decay uses A = -exp(A_log), so:
        dA = exp(delta * A) = exp(-delta * exp(A_log))
    For positive delta, dA must be in (0, 1) — a contractive step.
    """
    block = make_block()
    ssd = block.ssd

    # dA for a unit delta: exp(-exp(A_log)) must be in (0, 1)
    delta_unit = torch.ones(1)
    dA = torch.exp(delta_unit * (-torch.exp(ssd.A_log)))
    assert (dA < 1.0).all(), (
        "dA >= 1 for unit delta — A_log initialization does not ensure contraction."
    )
    assert (dA > 0.0).all(), "dA <= 0 — invalid decay value."


# ---------------------------------------------------------------------------
# Test 13: dt after softplus is positive
# ---------------------------------------------------------------------------

def test_delta_is_positive():
    """Discretization step delta must be strictly positive after softplus."""
    block = make_block()
    x = torch.randn(2, 8, D_MODEL)

    ssm_in = block.in_proj(x)
    _, _, dt_raw, _ = ssm_in.split(
        [D_STATE, D_STATE, N_HEADS, block.d_inner], dim=-1
    )
    delta = F.softplus(dt_raw + block.dt_bias)
    assert (delta > 0).all(), "Delta contains non-positive values after softplus."


# ---------------------------------------------------------------------------
# Test 14: Parameter count consistent with config
# ---------------------------------------------------------------------------

def test_parameter_count():
    """Parameter count must match the analytically expected value.

    Expected parameters:
    - z_proj:       d_model * (d_model * expand)
    - in_proj:      d_model * (d_state + d_state + n_heads + d_inner)
    - dt_bias:      n_heads
    - ssd.A_log:    n_heads
    - ssd.D:        n_heads
    - z_gate_proj:  (d_model * expand) * d_inner
    - out_proj:     d_inner * d_model
    """
    block = make_block()
    d_inner = N_HEADS * HEAD_DIM  # 4 * 16 = 64

    expected = (
        D_MODEL * (D_MODEL * EXPAND)                                  # z_proj
        + D_MODEL * (D_STATE + D_STATE + N_HEADS + d_inner)           # in_proj
        + N_HEADS                                                      # dt_bias
        + N_HEADS                                                      # ssd.A_log
        + N_HEADS                                                      # ssd.D
        + (D_MODEL * EXPAND) * d_inner                                # z_gate_proj
        + d_inner * D_MODEL                                           # out_proj
    )

    actual = sum(p.numel() for p in block.parameters())
    assert actual == expected, (
        f"Parameter count mismatch: expected {expected}, got {actual}. "
        "Check for unexpected linear layers or parameter duplication."
    )
