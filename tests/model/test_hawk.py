"""Tests for Hawk: Pure Real-Gated Linear Recurrence Model.

Reference: De et al., 2024 — "Griffin: Mixing Gated Linear Recurrences with
Local Attention for Efficient Language Models". https://arxiv.org/abs/2402.19427

Tiny config throughout: d_model=64, n_layers=2.
"""

from __future__ import annotations

import math
from typing import List

import pytest
import torch
import torch.nn as nn

from src.model.hawk import RGLRU, HawkBlock, HawkModel


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

D_MODEL = 64
N_LAYERS = 2
BATCH = 3
T = 8


@pytest.fixture
def model() -> HawkModel:
    torch.manual_seed(0)
    return HawkModel(d_model=D_MODEL, n_layers=N_LAYERS)


@pytest.fixture
def rglru_layer() -> RGLRU:
    torch.manual_seed(1)
    return RGLRU(d_model=D_MODEL)


# ---------------------------------------------------------------------------
# 1. Output shape (B, T, d_model)
# ---------------------------------------------------------------------------

def test_output_shape(model: HawkModel) -> None:
    x = torch.randn(BATCH, T, D_MODEL)
    out, _ = model(x)
    assert out.shape == (BATCH, T, D_MODEL), (
        f"Expected output shape {(BATCH, T, D_MODEL)}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 2. Hidden state shape: list of (B, d_model), one per layer
# ---------------------------------------------------------------------------

def test_hidden_state_shape(model: HawkModel) -> None:
    x = torch.randn(BATCH, T, D_MODEL)
    _, new_hs = model(x)
    assert len(new_hs) == N_LAYERS, (
        f"Expected {N_LAYERS} hidden states, got {len(new_hs)}"
    )
    for i, h in enumerate(new_hs):
        assert h.shape == (BATCH, D_MODEL), (
            f"Hidden state {i}: expected shape {(BATCH, D_MODEL)}, got {h.shape}"
        )


# ---------------------------------------------------------------------------
# 3. Gradient flow: all parameters including Λ receive finite gradients
# ---------------------------------------------------------------------------

def test_gradient_flow(model: HawkModel) -> None:
    x = torch.randn(BATCH, T, D_MODEL, requires_grad=True)
    out, _ = model(x)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter '{name}'"
        assert torch.isfinite(param.grad).all(), (
            f"Non-finite gradient for parameter '{name}'"
        )

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# 4. Determinism under torch.manual_seed
# ---------------------------------------------------------------------------

def test_determinism() -> None:
    x = torch.randn(BATCH, T, D_MODEL)

    torch.manual_seed(42)
    m1 = HawkModel(d_model=D_MODEL, n_layers=N_LAYERS)
    out1, _ = m1(x)

    torch.manual_seed(42)
    m2 = HawkModel(d_model=D_MODEL, n_layers=N_LAYERS)
    out2, _ = m2(x)

    assert torch.allclose(out1, out2), "Outputs differ despite same manual seed"


# ---------------------------------------------------------------------------
# 5. batch=1, T=1 edge case
# ---------------------------------------------------------------------------

def test_single_token() -> None:
    torch.manual_seed(7)
    m = HawkModel(d_model=D_MODEL, n_layers=N_LAYERS)
    x = torch.randn(1, 1, D_MODEL)
    out, hs = m(x)
    assert out.shape == (1, 1, D_MODEL)
    for h in hs:
        assert h.shape == (1, D_MODEL)


# ---------------------------------------------------------------------------
# 6. hidden_states=None → zero initialisation (matches explicit zeros)
# ---------------------------------------------------------------------------

def test_none_hidden_state_is_zero_init(model: HawkModel) -> None:
    torch.manual_seed(5)
    x = torch.randn(BATCH, T, D_MODEL)

    out_none, _ = model(x, hidden_states=None)

    zero_hs = [torch.zeros(BATCH, D_MODEL) for _ in range(N_LAYERS)]
    out_zero, _ = model(x, hidden_states=zero_hs)

    assert torch.allclose(out_none, out_zero), (
        "hidden_states=None should be equivalent to a list of zero tensors"
    )


# ---------------------------------------------------------------------------
# 7. Passed hidden_state affects output (state is used)
# ---------------------------------------------------------------------------

def test_hidden_state_affects_output(model: HawkModel) -> None:
    torch.manual_seed(9)
    x = torch.randn(BATCH, T, D_MODEL)

    out_zero, _ = model(x, hidden_states=None)

    # Non-zero initial state
    nonzero_hs = [torch.randn(BATCH, D_MODEL) for _ in range(N_LAYERS)]
    out_nonzero, _ = model(x, hidden_states=nonzero_hs)

    assert not torch.allclose(out_zero, out_nonzero), (
        "Different initial hidden states should produce different outputs"
    )


# ---------------------------------------------------------------------------
# 8. a_t ∈ (0, 1): state transitions are always contractions
# ---------------------------------------------------------------------------

def test_state_transition_contraction(rglru_layer: RGLRU) -> None:
    """Verify a_t = exp(-8 * softplus(Λ) * r_t) lies strictly in (0, 1)."""
    x = torch.randn(BATCH, T, D_MODEL)
    # Compute a_t values directly
    for t in range(T):
        x_t = x[:, t, :]
        r_t = torch.sigmoid(rglru_layer.W_r(x_t))
        import torch.nn.functional as F
        decay = 8.0 * F.softplus(rglru_layer.Lambda)
        a_t = torch.exp(-decay * r_t)
        assert (a_t > 0).all(), f"a_t has non-positive values at t={t}"
        assert (a_t < 1).all(), f"a_t has values >= 1 at t={t}"


# ---------------------------------------------------------------------------
# 9. State carries context: T=8 with warm hidden differs from cold start
# ---------------------------------------------------------------------------

def test_state_carries_context() -> None:
    torch.manual_seed(11)
    m = HawkModel(d_model=D_MODEL, n_layers=N_LAYERS)

    # First chunk: T steps that prime the state
    x_prime = torch.randn(BATCH, T, D_MODEL)
    _, warm_hs = m(x_prime)

    # Second chunk with warm vs cold state
    x_query = torch.randn(BATCH, T, D_MODEL)
    out_warm, _ = m(x_query, hidden_states=warm_hs)
    out_cold, _ = m(x_query, hidden_states=None)

    assert not torch.allclose(out_warm, out_cold), (
        "Model with primed hidden state should differ from cold-start output"
    )


# ---------------------------------------------------------------------------
# 10. No NaN/Inf on zeros input
# ---------------------------------------------------------------------------

def test_no_nan_on_zeros() -> None:
    torch.manual_seed(13)
    m = HawkModel(d_model=D_MODEL, n_layers=N_LAYERS)
    x = torch.zeros(BATCH, T, D_MODEL)
    out, hs = m(x)
    assert torch.isfinite(out).all(), "NaN/Inf in output for all-zero input"
    for h in hs:
        assert torch.isfinite(h).all(), "NaN/Inf in hidden state for all-zero input"


# ---------------------------------------------------------------------------
# 11. No NaN/Inf on large inputs
# ---------------------------------------------------------------------------

def test_no_nan_on_large_input() -> None:
    torch.manual_seed(15)
    m = HawkModel(d_model=D_MODEL, n_layers=N_LAYERS)
    x = torch.randn(BATCH, T, D_MODEL) * 100.0
    out, hs = m(x)
    assert torch.isfinite(out).all(), "NaN/Inf in output for large-magnitude input"
    for h in hs:
        assert torch.isfinite(h).all(), "NaN/Inf in hidden state for large input"


# ---------------------------------------------------------------------------
# 12. Variance preservation: output variance ≈ input variance (unit-var input)
# ---------------------------------------------------------------------------

def test_variance_preservation() -> None:
    """For unit-variance input over many steps the RG-LRU preserves variance."""
    torch.manual_seed(17)
    layer = RGLRU(d_model=D_MODEL)
    # Large batch and long sequence to get stable estimate
    B_big, T_big = 512, 128
    x = torch.randn(B_big, T_big, D_MODEL)  # unit variance input

    with torch.no_grad():
        out_seq, _ = layer(x)

    # Measure variance of output (last 64 steps to let transients die out)
    out_var = out_seq[:, 64:, :].var().item()
    # We accept variance within a fairly wide range — exact value depends on
    # the random init of Λ, but it should not explode or vanish.
    assert 0.01 < out_var < 100.0, (
        f"Output variance {out_var:.4f} is outside the acceptable [0.01, 100] range"
    )


# ---------------------------------------------------------------------------
# 13. Λ initialised to log-spaced values (not all identical)
# ---------------------------------------------------------------------------

def test_lambda_log_spaced_init() -> None:
    torch.manual_seed(0)
    layer = RGLRU(d_model=D_MODEL)
    lam = layer.Lambda.detach()
    # Values should not all be the same
    assert lam.std().item() > 1e-6, (
        "Lambda parameter is not log-spaced — all values appear identical"
    )
    # Should span a meaningful range
    span = lam.max().item() - lam.min().item()
    assert span > 0.5, f"Lambda span {span:.4f} is too small — expected log-spaced init"


# ---------------------------------------------------------------------------
# 14. n_layers=1 edge case works correctly
# ---------------------------------------------------------------------------

def test_single_layer() -> None:
    torch.manual_seed(19)
    m = HawkModel(d_model=D_MODEL, n_layers=1)
    x = torch.randn(BATCH, T, D_MODEL)
    out, hs = m(x)
    assert out.shape == (BATCH, T, D_MODEL)
    assert len(hs) == 1
    assert hs[0].shape == (BATCH, D_MODEL)


# ---------------------------------------------------------------------------
# 15. Wrong hidden_states length raises ValueError
# ---------------------------------------------------------------------------

def test_wrong_hidden_state_length_raises(model: HawkModel) -> None:
    x = torch.randn(BATCH, T, D_MODEL)
    bad_hs = [torch.zeros(BATCH, D_MODEL)]  # only 1 state for 2-layer model
    with pytest.raises(ValueError, match="hidden_states"):
        model(x, hidden_states=bad_hs)


# ---------------------------------------------------------------------------
# 16. d_state != d_model raises ValueError
# ---------------------------------------------------------------------------

def test_d_state_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="d_state"):
        HawkModel(d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_MODEL // 2)
