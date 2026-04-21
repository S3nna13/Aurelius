"""Tests for src/model/linear_recurrent_unit.py -- LRULayer.

Reference: Orvieto et al., 2023, "Resurrecting Recurrent Neural Networks for
Long Sequences". Tests cover config defaults, complex-diagonal A properties,
stability guarantees, forward shape/correctness, gradients, and edge cases.

Tiny test config: d_model=64, d_state=32, r_min=0.0, r_max=0.9
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.model.linear_recurrent_unit import LRUConfig, LRULayer
from src.model import MODEL_COMPONENT_REGISTRY

# ---------------------------------------------------------------------------
# Shared tiny test fixtures
# ---------------------------------------------------------------------------

D_MODEL = 64
D_STATE = 32
R_MIN = 0.0
R_MAX = 0.9

B = 2   # batch size
T = 16  # sequence length


def make_cfg(**kwargs) -> LRUConfig:
    defaults = dict(d_model=D_MODEL, d_state=D_STATE, r_min=R_MIN, r_max=R_MAX)
    defaults.update(kwargs)
    return LRUConfig(**defaults)


def make_layer(**kwargs) -> LRULayer:
    return LRULayer(make_cfg(**kwargs))


# ===========================================================================
# 1. test_config_defaults
# ===========================================================================


def test_config_defaults():
    """Default LRUConfig values match documented defaults."""
    cfg = LRUConfig()
    assert cfg.d_model == 2048
    assert cfg.d_state == 256
    assert cfg.r_min == 0.0
    assert cfg.r_max == 0.9
    assert math.isclose(cfg.max_phase, 2 * math.pi, rel_tol=1e-6)


# ===========================================================================
# 2. test_get_a_magnitude -- |A| in (r_min, r_max) for all states
# ===========================================================================


def test_get_a_magnitude():
    """Eigenvalue magnitudes lie in [r_min, r_max] with |A| < 1 guaranteed.

    When r_min=0.0 the lowest-frequency channel may initialize to mag=0
    (exp(-exp(inf))=0 is numerically valid per the LRU spec). The critical
    stability invariant is mag < 1 for all channels.
    """
    layer = make_layer()
    A_re, A_im = layer.get_A()
    mag = torch.sqrt(A_re ** 2 + A_im ** 2)

    # Non-negative (complex modulus is always >= 0)
    assert (mag >= 0).all(), "Magnitudes must be non-negative"
    # Stability: strictly less than 1
    assert (mag < 1).all(), "Magnitudes must be strictly less than 1 (stable)"
    # Upper bound at init: should not exceed r_max
    assert (mag <= R_MAX + 1e-5).all(), (
        f"All initial magnitudes should be <= r_max={R_MAX}"
    )


# ===========================================================================
# 3. test_get_a_complex -- A_re and A_im both have shape [d_state]
# ===========================================================================


def test_get_a_complex():
    """get_A() returns two real tensors each of shape [d_state]."""
    layer = make_layer()
    A_re, A_im = layer.get_A()

    assert A_re.shape == (D_STATE,), f"Expected ({D_STATE},), got {A_re.shape}"
    assert A_im.shape == (D_STATE,), f"Expected ({D_STATE},), got {A_im.shape}"
    assert A_re.dtype == torch.float32
    assert A_im.dtype == torch.float32

    assert not torch.allclose(A_im, torch.zeros_like(A_im)), (
        "A_im should not be uniformly zero -- phases should be non-trivial"
    )


# ===========================================================================
# 4. test_get_gamma_positive -- gamma > 0 for all states
# ===========================================================================


def test_get_gamma_positive():
    """get_gamma() must return strictly positive values for all state dims."""
    layer = make_layer()
    gamma = layer.get_gamma()

    assert gamma.shape == (D_STATE,), f"Expected ({D_STATE},), got {gamma.shape}"
    assert (gamma > 0).all(), "gamma must be strictly positive for all states"
    assert (gamma <= 1.0).all(), "gamma = sqrt(1-mag^2) <= 1"


# ===========================================================================
# 5. test_stability -- |A|^2 + gamma^2 = 1 by construction
# ===========================================================================


def test_stability():
    """Normalization identity: |A|^2 + gamma^2 = 1 exactly by construction."""
    layer = make_layer()
    A_re, A_im = layer.get_A()
    gamma = layer.get_gamma()

    mag_sq = A_re ** 2 + A_im ** 2
    gamma_sq = gamma ** 2

    identity = mag_sq + gamma_sq
    assert torch.allclose(identity, torch.ones_like(identity), atol=1e-5), (
        f"|A|^2 + gamma^2 should equal 1; max deviation: {(identity - 1).abs().max():.2e}"
    )


# ===========================================================================
# 6. test_forward_shape -- output [B, T, d_model]
# ===========================================================================


def test_forward_shape():
    """LRULayer.forward() must return shape (B, T, d_model)."""
    layer = make_layer()
    x = torch.randn(B, T, D_MODEL)
    y = layer(x)

    assert y.shape == (B, T, D_MODEL), (
        f"Expected ({B}, {T}, {D_MODEL}), got {y.shape}"
    )


# ===========================================================================
# 7. test_forward_deterministic -- same input -> same output
# ===========================================================================


def test_forward_deterministic():
    """Calling forward twice with the same input must produce identical output."""
    layer = make_layer()
    layer.train(False)
    x = torch.randn(B, T, D_MODEL)

    with torch.no_grad():
        y1 = layer(x)
        y2 = layer(x)

    assert torch.allclose(y1, y2), "forward() must be deterministic"


# ===========================================================================
# 8. test_skip_connection -- D != 0 contributes to output
# ===========================================================================


def test_skip_connection():
    """Setting D to zero must change the output, confirming D contributes."""
    layer = make_layer()
    layer.train(False)
    x = torch.randn(B, T, D_MODEL)

    with torch.no_grad():
        y_with_D = layer(x)
        original_D = layer.D.data.clone()
        layer.D.data.zero_()
        y_without_D = layer(x)
        layer.D.data.copy_(original_D)

    assert not torch.allclose(y_with_D, y_without_D), (
        "Zeroing D should change the output -- D contributes via skip connection"
    )


# ===========================================================================
# 9. test_state_size -- returns 2 * d_state
# ===========================================================================


def test_state_size():
    """state_size() must return 2 * d_state."""
    layer = make_layer()
    assert layer.state_size() == 2 * D_STATE, (
        f"Expected {2 * D_STATE}, got {layer.state_size()}"
    )


# ===========================================================================
# 10. test_gradient_a_params -- gradients flow through nu_log and theta_log
# ===========================================================================


def test_gradient_a_params():
    """A scalar loss backward must produce gradients for nu_log and theta_log."""
    layer = make_layer()
    x = torch.randn(B, T, D_MODEL)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert layer.nu_log.grad is not None, "nu_log should have gradients"
    assert layer.theta_log.grad is not None, "theta_log should have gradients"
    assert not torch.isnan(layer.nu_log.grad).any(), "nu_log grad has NaN"
    assert not torch.isnan(layer.theta_log.grad).any(), "theta_log grad has NaN"
    assert (layer.nu_log.grad.abs() > 0).any(), "nu_log grad is entirely zero"
    assert (layer.theta_log.grad.abs() > 0).any(), "theta_log grad is entirely zero"


# ===========================================================================
# 11. test_gradient_bc_params -- gradients flow through B_re, C_re
# ===========================================================================


def test_gradient_bc_params():
    """A scalar loss backward must produce gradients for B_re and C_re."""
    layer = make_layer()
    x = torch.randn(B, T, D_MODEL)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    for name in ("B_re", "B_im", "C_re", "C_im"):
        param = getattr(layer, name)
        assert param.grad is not None, f"{name} should have gradients"
        assert not torch.isnan(param.grad).any(), f"{name} grad has NaN"
        assert (param.grad.abs() > 0).any(), f"{name} grad is entirely zero"


# ===========================================================================
# 12. test_long_sequence_stable -- T=256 doesn't produce nan/inf
# ===========================================================================


def test_long_sequence_stable():
    """Forward pass over a long sequence (T=256) must not produce NaN or Inf."""
    layer = make_layer()
    layer.train(False)
    x = torch.randn(2, 256, D_MODEL)

    with torch.no_grad():
        y = layer(x)

    assert not torch.isnan(y).any(), "NaN detected in long-sequence output"
    assert not torch.isinf(y).any(), "Inf detected in long-sequence output"


# ===========================================================================
# 13. test_single_step -- T=1 works
# ===========================================================================


def test_single_step():
    """Forward pass with a single-token sequence (T=1) must work correctly."""
    layer = make_layer()
    layer.train(False)
    x = torch.randn(B, 1, D_MODEL)

    with torch.no_grad():
        y = layer(x)

    assert y.shape == (B, 1, D_MODEL), f"Expected ({B}, 1, {D_MODEL}), got {y.shape}"
    assert not torch.isnan(y).any(), "NaN in single-step output"


# ===========================================================================
# 14. test_batch_independence -- outputs for different batch items are independent
# ===========================================================================


def test_batch_independence():
    """Each batch item's output must depend only on its own input."""
    layer = make_layer()
    layer.train(False)

    x = torch.randn(B, T, D_MODEL)
    with torch.no_grad():
        y_batch = layer(x)

    for b in range(B):
        x_single = x[b : b + 1]
        with torch.no_grad():
            y_single = layer(x_single)
        assert torch.allclose(y_batch[b : b + 1], y_single, atol=1e-5), (
            f"Batch item {b} output differs from independent forward pass"
        )


# ===========================================================================
# 15. test_registry -- LRULayer registered in MODEL_COMPONENT_REGISTRY
# ===========================================================================


def test_registry():
    """LRULayer must be registered under key 'lru' in MODEL_COMPONENT_REGISTRY."""
    assert "lru" in MODEL_COMPONENT_REGISTRY, (
        "'lru' key not found in MODEL_COMPONENT_REGISTRY"
    )
    assert MODEL_COMPONENT_REGISTRY["lru"] is LRULayer, (
        "MODEL_COMPONENT_REGISTRY['lru'] should point to LRULayer"
    )


# ===========================================================================
# 16. Integration test -- full forward + backward, no NaN, correct shapes
# ===========================================================================


def test_integration_forward_backward():
    """Integration: d_model=64, d_state=32, B=2, T=16.

    Runs a full forward and backward pass and verifies:
      - Output shape is (B, T, d_model)
      - No NaN or Inf in output
      - All parameter gradients are finite and non-None
    """
    cfg = LRUConfig(d_model=64, d_state=32, r_min=0.0, r_max=0.9)
    layer = LRULayer(cfg)

    x = torch.randn(2, 16, 64, requires_grad=True)
    y = layer(x)

    assert y.shape == (2, 16, 64), f"Expected (2, 16, 64), got {y.shape}"
    assert not torch.isnan(y).any(), "NaN in integration output"
    assert not torch.isinf(y).any(), "Inf in integration output"

    loss = y.mean()
    loss.backward()

    assert x.grad is not None, "Input x should have gradients"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"

    param_names = ["nu_log", "theta_log", "B_re", "B_im", "C_re", "C_im", "D"]
    for name in param_names:
        param = getattr(layer, name)
        assert param.grad is not None, f"Parameter '{name}' has no gradient"
        assert not torch.isnan(param.grad).any(), f"NaN in gradient of '{name}'"
        assert not torch.isinf(param.grad).any(), f"Inf in gradient of '{name}'"
