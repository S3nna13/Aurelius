"""Tests for src/model/xlstm.py (Beck et al., 2024, arXiv:2405.04517).

Covers all 14 required test scenarios.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.xlstm import mLSTMCell, sLSTMCell, xLSTMBlock, xLSTMModel

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------

D = 32       # d_model
B = 2        # batch size
T = 8        # sequence length


# ===========================================================================
# 1. sLSTMCell output shape
# ===========================================================================

def test_slstm_cell_output_shape():
    cell = sLSTMCell(D)
    x_t = torch.randn(B, D)
    h_t, state = cell(x_t)
    assert h_t.shape == (B, D), f"Expected ({B}, {D}), got {h_t.shape}"
    c, n, m = state
    assert c.shape == (B, D)
    assert n.shape == (B, D)
    assert m.shape == (B, D)


# ===========================================================================
# 2. mLSTMCell output shape
# ===========================================================================

def test_mlstm_cell_output_shape():
    cell = mLSTMCell(D)
    x_t = torch.randn(B, D)
    h_t, state = cell(x_t)
    assert h_t.shape == (B, D), f"Expected ({B}, {D}), got {h_t.shape}"
    C, n, m = state
    assert C.shape == (B, D, D)
    assert n.shape == (B, D)
    assert m.shape == (B,)


# ===========================================================================
# 3. xLSTMModel output shape (B, T, d_model)
# ===========================================================================

def test_xlstm_model_output_shape():
    model = xLSTMModel(d_model=D, n_layers=2)
    x = torch.randn(B, T, D)
    out, _ = model(x)
    assert out.shape == (B, T, D), f"Expected ({B}, {T}, {D}), got {out.shape}"


# ===========================================================================
# 4. Hidden states returned and have correct shapes
# ===========================================================================

def test_hidden_states_shapes():
    model = xLSTMModel(d_model=D, n_layers=2, block_types=["mlstm", "slstm"])
    x = torch.randn(B, T, D)
    _, hs = model(x)
    assert len(hs) == 2, "Should return one state per layer"

    # Layer 0: mLSTM -> (C, n, m)
    C, n, m = hs[0]
    assert C.shape == (B, D, D)
    assert n.shape == (B, D)
    assert m.shape == (B,)

    # Layer 1: sLSTM -> (c, n, m)
    c, n2, m2 = hs[1]
    assert c.shape == (B, D)
    assert n2.shape == (B, D)
    assert m2.shape == (B, D)


# ===========================================================================
# 5. Gradient flow: finite grads on all params
# ===========================================================================

def test_gradient_flow():
    model = xLSTMModel(d_model=D, n_layers=2)
    x = torch.randn(B, T, D)
    out, _ = model(x)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


# ===========================================================================
# 6. Determinism under fixed seed
# ===========================================================================

def test_determinism():
    def run():
        torch.manual_seed(42)
        model = xLSTMModel(d_model=D, n_layers=2)
        model.eval()
        x = torch.randn(B, T, D)
        with torch.no_grad():
            out, _ = model(x)
        return out

    assert torch.allclose(run(), run()), "Results differ across identical seeds"


# ===========================================================================
# 7. Edge case: batch=1, T=1
# ===========================================================================

def test_edge_case_batch1_T1():
    model = xLSTMModel(d_model=D, n_layers=2)
    x = torch.randn(1, 1, D)
    out, hs = model(x)
    assert out.shape == (1, 1, D)
    assert len(hs) == 2


# ===========================================================================
# 8. hidden_states=None -> zero init, no crash
# ===========================================================================

def test_hidden_states_none_no_crash():
    model = xLSTMModel(d_model=D, n_layers=2)
    x = torch.randn(B, T, D)
    out, hs = model(x, hidden_states=None)
    assert out.shape == (B, T, D)


# ===========================================================================
# 9. Passed state affects output
# ===========================================================================

def test_passed_state_affects_output():
    model = xLSTMModel(d_model=D, n_layers=2)
    model.eval()
    x = torch.randn(B, T, D)

    with torch.no_grad():
        _, hs = model(x, hidden_states=None)
        x2 = torch.randn(B, T, D)
        out_with_state, _ = model(x2, hidden_states=hs)
        out_no_state, _ = model(x2, hidden_states=None)

    assert not torch.allclose(out_with_state, out_no_state), \
        "Output should differ when non-zero state is passed"


# ===========================================================================
# 10. Stabiliser m_t == max(z_f + m_prev, z_i) at each step (sLSTM)
# ===========================================================================

def test_slstm_stabiliser_is_max():
    """m_t in log-space must equal max(z_f + m_prev, z_i) exactly."""
    cell = sLSTMCell(D)
    cell.eval()
    B_test = 4
    x_t = torch.randn(B_test, D)

    with torch.no_grad():
        state = cell.init_state(B_test, x_t.device, x_t.dtype)
        _, (_, _, m_prev) = cell(x_t, state)

        x_t2 = torch.randn(B_test, D)
        z_i = cell.W_i(x_t2)
        z_f = cell.W_f(x_t2)
        expected_m = torch.maximum(z_f + m_prev, z_i)

        c_dummy = torch.zeros(B_test, D)
        n_dummy = torch.zeros(B_test, D)
        _, (_, _, m_t) = cell(x_t2, (c_dummy, n_dummy, m_prev))

    assert torch.allclose(m_t, expected_m, atol=1e-5), \
        "Stabiliser m_t does not match max(z_f + m_prev, z_i)"


# ===========================================================================
# 11. No NaN/Inf on zeros input
# ===========================================================================

def test_no_nan_inf_zeros_input():
    model = xLSTMModel(d_model=D, n_layers=2)
    x = torch.zeros(B, T, D)
    out, _ = model(x)
    assert torch.isfinite(out).all(), "NaN or Inf in output for zeros input"


# ===========================================================================
# 12. No NaN/Inf on large inputs
# ===========================================================================

def test_no_nan_inf_large_input():
    model = xLSTMModel(d_model=D, n_layers=2)
    x = torch.randn(B, T, D) * 100.0
    out, _ = model(x)
    assert torch.isfinite(out).all(), "NaN or Inf in output for large input"


# ===========================================================================
# 13. Normaliser prevents overflow (output finite for extreme gates)
# ===========================================================================

def test_normaliser_prevents_overflow():
    """Force extreme gate values; output must remain finite."""
    cell = sLSTMCell(D)
    cell.eval()

    with torch.no_grad():
        nn.init.constant_(cell.W_i.weight, 10.0)
        nn.init.constant_(cell.W_f.weight, 10.0)
        nn.init.constant_(cell.W_i.bias, 10.0)
        nn.init.constant_(cell.W_f.bias, 10.0)

    x_t = torch.ones(B, D)
    h_t, _ = cell(x_t)
    assert torch.isfinite(h_t).all(), "Output not finite under extreme gate values"


# ===========================================================================
# 14. block_types list of length != n_layers raises ValueError
# ===========================================================================

def test_mismatched_block_types_raises():
    with pytest.raises(ValueError, match="block_types"):
        xLSTMModel(d_model=D, n_layers=3, block_types=["mlstm", "slstm"])
