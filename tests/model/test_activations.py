"""Tests for src/model/activations.py.

Uses tiny dimensions throughout so the suite runs quickly on CPU.
"""

import torch
import pytest

from src.model.activations import (
    swiglu,
    geglu,
    reglu,
    silu,
    quick_gelu,
    squared_relu,
    SwiGLUFFN,
    GeGLUFFN,
    ActivationFactory,
    benchmark_activation,
)

# ---------------------------------------------------------------------------
# Shared tiny dimensions
# ---------------------------------------------------------------------------
B = 2
T = 4
D_MODEL = 16
D_FF = 32


# ---------------------------------------------------------------------------
# Functional: swiglu
# ---------------------------------------------------------------------------

def test_swiglu_output_shape():
    x = torch.randn(B, T, D_MODEL)
    gate = torch.randn(B, T, D_MODEL)
    out = swiglu(x, gate)
    assert out.shape == (B, T, D_MODEL)


def test_swiglu_zero_gate_gives_zero():
    """When gate == 0, silu(0) == 0 so output must be zero everywhere."""
    x = torch.randn(B, T, D_MODEL)
    gate = torch.zeros(B, T, D_MODEL)
    out = swiglu(x, gate)
    assert torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# Functional: geglu
# ---------------------------------------------------------------------------

def test_geglu_output_shape():
    x = torch.randn(B, T, D_MODEL)
    gate = torch.randn(B, T, D_MODEL)
    out = geglu(x, gate)
    assert out.shape == (B, T, D_MODEL)


def test_geglu_zero_gate_gives_zero():
    """gelu(0) == 0 so output must be zero when gate == 0."""
    x = torch.randn(B, T, D_MODEL)
    gate = torch.zeros(B, T, D_MODEL)
    out = geglu(x, gate)
    assert torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# Functional: reglu
# ---------------------------------------------------------------------------

def test_reglu_output_shape():
    x = torch.randn(B, T, D_MODEL)
    gate = torch.randn(B, T, D_MODEL)
    out = reglu(x, gate)
    assert out.shape == (B, T, D_MODEL)


def test_reglu_negative_gate_gives_zero():
    """relu(gate) == 0 for all negative gate values."""
    x = torch.ones(B, T, D_MODEL)
    gate = -torch.abs(torch.randn(B, T, D_MODEL)) - 1e-6  # strictly negative
    out = reglu(x, gate)
    assert torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# Functional: silu
# ---------------------------------------------------------------------------

def test_silu_output_shape():
    x = torch.randn(B, T, D_MODEL)
    assert silu(x).shape == (B, T, D_MODEL)


def test_silu_differentiable():
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out = silu(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# Functional: quick_gelu
# ---------------------------------------------------------------------------

def test_quick_gelu_output_shape():
    x = torch.randn(B, T, D_MODEL)
    assert quick_gelu(x).shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# Functional: squared_relu
# ---------------------------------------------------------------------------

def test_squared_relu_non_negative():
    """squared_relu must produce non-negative values for any input."""
    x = torch.randn(B, T, D_MODEL) * 5  # mix of positive and negative
    out = squared_relu(x)
    assert (out >= 0).all()


def test_squared_relu_output_shape():
    x = torch.randn(B, T, D_MODEL)
    assert squared_relu(x).shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# Module: SwiGLUFFN
# ---------------------------------------------------------------------------

def test_swiglu_ffn_output_shape():
    ffn = SwiGLUFFN(D_MODEL, D_FF)
    x = torch.randn(B, T, D_MODEL)
    assert ffn(x).shape == (B, T, D_MODEL)


def test_swiglu_ffn_gradient_flows():
    """Gradients should reach input tensor through SwiGLUFFN."""
    ffn = SwiGLUFFN(D_MODEL, D_FF)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out = ffn(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    # Gradient should not be all-zero for a random init
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


# ---------------------------------------------------------------------------
# Module: GeGLUFFN
# ---------------------------------------------------------------------------

def test_geglu_ffn_output_shape():
    ffn = GeGLUFFN(D_MODEL, D_FF)
    x = torch.randn(B, T, D_MODEL)
    assert ffn(x).shape == (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# ActivationFactory
# ---------------------------------------------------------------------------

def test_activation_factory_get_silu_callable():
    fn = ActivationFactory.get("silu")
    assert callable(fn)
    x = torch.randn(3, 3)
    out = fn(x)
    assert out.shape == x.shape


def test_activation_factory_get_swish_alias():
    """'swish' should be the same function as 'silu'."""
    fn_silu = ActivationFactory.get("silu")
    fn_swish = ActivationFactory.get("swish")
    x = torch.randn(4, 4)
    assert torch.allclose(fn_silu(x), fn_swish(x))


def test_activation_factory_list_available_returns_list():
    names = ActivationFactory.list_available()
    assert isinstance(names, list)
    assert len(names) >= 5
    for expected in ("silu", "swish", "gelu", "relu", "quick_gelu", "squared_relu"):
        assert expected in names


def test_activation_factory_unknown_name_raises():
    with pytest.raises(KeyError):
        ActivationFactory.get("nonexistent_activation_xyz")


# ---------------------------------------------------------------------------
# benchmark_activation
# ---------------------------------------------------------------------------

def test_benchmark_activation_returns_correct_keys():
    x = torch.randn(B, T, D_MODEL)
    result = benchmark_activation(silu, x, n_runs=5)
    assert "mean_ms" in result
    assert "std_ms" in result
    assert "output_shape" in result


def test_benchmark_activation_output_shape_string():
    x = torch.randn(B, T, D_MODEL)
    result = benchmark_activation(quick_gelu, x, n_runs=3)
    # output_shape should encode all three dimensions
    shape_str = result["output_shape"]
    assert str(B) in shape_str
    assert str(T) in shape_str
    assert str(D_MODEL) in shape_str


def test_benchmark_activation_mean_positive():
    x = torch.randn(16, 16)
    result = benchmark_activation(squared_relu, x, n_runs=5)
    assert result["mean_ms"] > 0


# ---------------------------------------------------------------------------
# Cross-activation: swiglu vs geglu differ on same inputs
# ---------------------------------------------------------------------------

def test_swiglu_and_geglu_differ():
    """For the same (x, gate) pair, SwiGLU and GeGLU should produce different
    outputs because silu != gelu in general."""
    torch.manual_seed(0)
    x = torch.randn(B, T, D_MODEL)
    gate = torch.randn(B, T, D_MODEL)
    out_swiglu = swiglu(x, gate)
    out_geglu = geglu(x, gate)
    # They should not be identical (extremely unlikely with random data)
    assert not torch.allclose(out_swiglu, out_geglu)
