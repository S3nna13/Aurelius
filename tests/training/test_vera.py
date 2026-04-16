"""Tests for VeRA (Vector-based Random Matrix Adaptation).

Reference: Kopiczko et al., arXiv:2310.11454.

Test config: in_features=64, out_features=128, rank=8.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.vera import VeRALinear, VeRAModel, _make_shared_matrix

# ---------------------------------------------------------------------------
# Tiny test dimensions (kept small for speed)
# ---------------------------------------------------------------------------
IN_F = 64
OUT_F = 128
RANK = 8
BATCH = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_vera_linear(freeze_base: bool = True) -> VeRALinear:
    """Return a VeRALinear with no shared A/B supplied (internally generated)."""
    layer = VeRALinear(IN_F, OUT_F, RANK)
    if freeze_base:
        layer.base_linear.weight.requires_grad_(False)
        if layer.base_linear.bias is not None:
            layer.base_linear.bias.requires_grad_(False)
    return layer


def _tiny_model() -> nn.Module:
    """Two-layer MLP with named linear layers for VeRAModel testing."""
    return nn.Sequential(
        nn.Linear(IN_F, OUT_F),
        nn.ReLU(),
        nn.Linear(OUT_F, IN_F),
    )


# ---------------------------------------------------------------------------
# Test 1: Output shape
# ---------------------------------------------------------------------------

def test_output_shape():
    """VeRALinear output shape matches nn.Linear for same inputs."""
    vera = _fresh_vera_linear()
    x = torch.randn(BATCH, IN_F)
    out = vera(x)
    assert out.shape == (BATCH, OUT_F), f"Expected ({BATCH}, {OUT_F}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: Zero-delta initialisation (d=0 → delta term vanishes)
# ---------------------------------------------------------------------------

def test_initial_output_equals_base():
    """With d=0, VeRALinear output should equal the base linear output."""
    vera = _fresh_vera_linear()
    # Confirm d is zeros
    assert torch.all(vera.d == 0.0), "d should be initialised to zeros"
    x = torch.randn(BATCH, IN_F)
    base_out = vera.base_linear(x)
    vera_out = vera(x)
    torch.testing.assert_close(
        vera_out, base_out,
        msg="Initial VeRA output should match base linear output when d=0",
    )


# ---------------------------------------------------------------------------
# Test 3: A, B are frozen (requires_grad=False)
# ---------------------------------------------------------------------------

def test_frozen_shared_matrices():
    """A and B must not require gradients."""
    vera = _fresh_vera_linear()
    assert not vera.A.requires_grad, "A must be frozen (requires_grad=False)"
    assert not vera.B.requires_grad, "B must be frozen (requires_grad=False)"


# ---------------------------------------------------------------------------
# Test 4: d, b are trainable (requires_grad=True)
# ---------------------------------------------------------------------------

def test_trainable_scaling_vectors():
    """d and b must be trainable parameters."""
    vera = _fresh_vera_linear()
    assert vera.d.requires_grad, "d must require grad"
    assert vera.b.requires_grad, "b must require grad"


# ---------------------------------------------------------------------------
# Test 5: Gradients flow through d and b
# ---------------------------------------------------------------------------

def test_gradient_flows_through_d_and_b():
    """Backward pass must produce non-None, non-zero gradients for d and b."""
    vera = _fresh_vera_linear()
    # Set d to non-zero so delta is non-zero
    with torch.no_grad():
        vera.d.fill_(1.0)
    x = torch.randn(BATCH, IN_F, requires_grad=False)
    loss = vera(x).sum()
    loss.backward()
    assert vera.d.grad is not None, "d.grad is None after backward"
    assert vera.b.grad is not None, "b.grad is None after backward"
    assert vera.d.grad.abs().sum() > 0, "d.grad is all zeros"
    assert vera.b.grad.abs().sum() > 0, "b.grad is all zeros"


# ---------------------------------------------------------------------------
# Test 6: d/b gradients are finite
# ---------------------------------------------------------------------------

def test_gradients_are_finite():
    """d and b gradients must be finite (no NaN/Inf)."""
    vera = _fresh_vera_linear()
    with torch.no_grad():
        vera.d.fill_(1.0)
    x = torch.randn(BATCH, IN_F)
    loss = vera(x).sum()
    loss.backward()
    assert torch.isfinite(vera.d.grad).all(), "d.grad contains non-finite values"
    assert torch.isfinite(vera.b.grad).all(), "b.grad contains non-finite values"


# ---------------------------------------------------------------------------
# Test 7: Determinism — same seed → same A, B, same output
# ---------------------------------------------------------------------------

def test_determinism_same_seed():
    """Two VeRALinear instances built with the same seed should produce identical outputs."""
    torch.manual_seed(42)
    vera1 = _fresh_vera_linear()
    torch.manual_seed(42)
    vera2 = _fresh_vera_linear()

    x = torch.randn(BATCH, IN_F)
    torch.manual_seed(99)
    out1 = vera1(x)
    torch.manual_seed(99)
    out2 = vera2(x)

    torch.testing.assert_close(vera1.A, vera2.A, msg="A matrices differ across seeds")
    torch.testing.assert_close(vera1.B, vera2.B, msg="B matrices differ across seeds")
    torch.testing.assert_close(out1, out2, msg="Outputs differ for same seed")


# ---------------------------------------------------------------------------
# Test 8: Shared A/B — two layers sharing the same tensors → same object id
# ---------------------------------------------------------------------------

def test_shared_matrices_same_object():
    """When two VeRALinear layers are given the same A/B tensors, they must
    reference the same objects (not copies)."""
    A = _make_shared_matrix(RANK, IN_F)
    B = _make_shared_matrix(OUT_F, RANK)

    layer1 = VeRALinear(IN_F, OUT_F, RANK, shared_A=A, shared_B=B)
    layer2 = VeRALinear(IN_F, OUT_F, RANK, shared_A=A, shared_B=B)

    # Buffers are registered as the same tensor
    assert layer1.A.data_ptr() == A.data_ptr(), "layer1.A is not the same object as A"
    assert layer2.A.data_ptr() == A.data_ptr(), "layer2.A is not the same object as A"
    assert layer1.B.data_ptr() == B.data_ptr(), "layer1.B is not the same object as B"
    assert layer2.B.data_ptr() == B.data_ptr(), "layer2.B is not the same object as B"


# ---------------------------------------------------------------------------
# Test 9: VeRAModel — only d and b appear in optimizer params
# ---------------------------------------------------------------------------

def test_vera_model_only_d_b_trainable():
    """After VeRAModel wraps a model, only d and b vectors should be trainable."""
    base = _tiny_model()
    # Target both linear layers by matching "0" and "2" (nn.Sequential indices)
    vera_model = VeRAModel(base, target_modules=["0", "2"], rank=RANK)

    trainable_params = [(n, p) for n, p in vera_model.named_parameters() if p.requires_grad]
    trainable_names = [n for n, _ in trainable_params]

    # Every trainable parameter name must end with '.d' or '.b'
    for name in trainable_names:
        assert name.endswith(".d") or name.endswith(".b"), (
            f"Unexpected trainable parameter: {name}"
        )

    # There must be at least one d and one b
    assert any(n.endswith(".d") for n in trainable_names), "No d parameters found"
    assert any(n.endswith(".b") for n in trainable_names), "No b parameters found"


# ---------------------------------------------------------------------------
# Test 10: VeRAModel — trainable param count << LoRA param count (same rank)
# ---------------------------------------------------------------------------

def test_vera_fewer_params_than_lora():
    """VeRA trainable parameters must be fewer than LoRA for the same rank.

    For a single linear layer (k=IN_F, m=OUT_F, r=RANK):
        LoRA  trainable = r*k + m*r = RANK*(IN_F + OUT_F)
        VeRA  trainable = r   + m   = RANK + OUT_F
    """
    lora_params = RANK * IN_F + OUT_F * RANK  # per layer × 2 layers
    lora_total = 2 * lora_params

    base = _tiny_model()
    vera_model = VeRAModel(base, target_modules=["0", "2"], rank=RANK)
    vera_total = vera_model.trainable_parameter_count()

    assert vera_total < lora_total, (
        f"VeRA trainable params ({vera_total}) should be < LoRA ({lora_total})"
    )


# ---------------------------------------------------------------------------
# Test 11: VeRAModel forward — output shape matches original model
# ---------------------------------------------------------------------------

def test_vera_model_forward_shape():
    """VeRAModel forward pass must produce the same output shape as the original."""
    x = torch.randn(BATCH, IN_F)
    base = _tiny_model()
    with torch.no_grad():
        expected_shape = base(x).shape

    vera_model = VeRAModel(base, target_modules=["0", "2"], rank=RANK)
    with torch.no_grad():
        out = vera_model(x)

    assert out.shape == expected_shape, (
        f"Output shape {out.shape} != expected {expected_shape}"
    )


# ---------------------------------------------------------------------------
# Test 12: Numerical stability — zero input
# ---------------------------------------------------------------------------

def test_numerical_stability_zero_input():
    """Output must be finite (no NaN/Inf) on an all-zeros input."""
    vera = _fresh_vera_linear()
    x = torch.zeros(BATCH, IN_F)
    out = vera(x)
    assert torch.isfinite(out).all(), "Output contains non-finite values for zero input"


# ---------------------------------------------------------------------------
# Test 13: Numerical stability — large input
# ---------------------------------------------------------------------------

def test_numerical_stability_large_input():
    """Output must be finite (no NaN/Inf) on large-valued inputs."""
    vera = _fresh_vera_linear()
    with torch.no_grad():
        vera.d.fill_(1.0)
    x = torch.ones(BATCH, IN_F) * 1e3
    out = vera(x)
    assert torch.isfinite(out).all(), "Output contains non-finite values for large input"


# ---------------------------------------------------------------------------
# Test 14: Optimizer step changes d and b from initial values
# ---------------------------------------------------------------------------

def test_optimizer_step_updates_d_and_b():
    """After one SGD step on a simple regression task, d and b must change."""
    torch.manual_seed(0)
    vera = _fresh_vera_linear()

    # Set d to a non-trivial value so the VeRA delta term is active;
    # this ensures gradients flow to both d and b.
    with torch.no_grad():
        vera.d.fill_(1.0)

    # Copy initial values AFTER setting d
    d_init = vera.d.data.clone()
    b_init = vera.b.data.clone()

    # Use a large lr so that even small gradients produce a visible update
    optimizer = torch.optim.SGD([vera.d, vera.b], lr=1.0)
    x = torch.randn(BATCH, IN_F)
    target = torch.zeros(BATCH, OUT_F)  # constant target → non-trivial residual

    loss = F.mse_loss(vera(x), target)
    loss.backward()
    optimizer.step()

    assert not torch.allclose(vera.d.data, d_init, atol=1e-6), (
        "d did not change after optimizer step"
    )
    assert not torch.allclose(vera.b.data, b_init, atol=1e-6), (
        "b did not change after optimizer step"
    )


# ---------------------------------------------------------------------------
# Test 15: VeRALinear init — b is all-ones
# ---------------------------------------------------------------------------

def test_b_initialized_to_ones():
    """b must be initialised to all-ones (paper Section 3.2)."""
    vera = _fresh_vera_linear()
    assert torch.all(vera.b == 1.0), "b should be initialised to all-ones"


# ---------------------------------------------------------------------------
# Test 16: Non-square / asymmetric shapes work correctly
# ---------------------------------------------------------------------------

def test_non_square_shapes():
    """VeRALinear should handle arbitrary in_features != out_features."""
    in_f, out_f, r = 32, 256, 4
    vera = VeRALinear(in_f, out_f, r)
    x = torch.randn(2, in_f)
    out = vera(x)
    assert out.shape == (2, out_f), f"Expected (2, {out_f}), got {out.shape}"
