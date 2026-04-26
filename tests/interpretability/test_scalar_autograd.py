"""Tests for src/interpretability/scalar_autograd.py — micrograd-style autograd."""

import math

from src.interpretability.scalar_autograd import (
    ScalarLayer,
    ScalarMLP,
    ScalarNeuron,
    ScalarValue,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wrap(x) -> ScalarValue:
    return ScalarValue(x)


def allclose(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Basic operations — forward pass
# ---------------------------------------------------------------------------


def test_add_forward():
    a, b = wrap(2.0), wrap(3.0)
    c = a + b
    assert allclose(c.data, 5.0)


def test_mul_forward():
    a, b = wrap(4.0), wrap(-2.0)
    c = a * b
    assert allclose(c.data, -8.0)


def test_pow_forward():
    a = wrap(3.0)
    c = a**2
    assert allclose(c.data, 9.0)


def test_relu_forward_positive():
    a = wrap(2.5)
    assert allclose(a.relu().data, 2.5)


def test_relu_forward_negative():
    a = wrap(-1.0)
    assert allclose(a.relu().data, 0.0)


def test_tanh_forward():
    a = wrap(0.0)
    assert allclose(a.tanh().data, 0.0)


def test_exp_forward():
    a = wrap(1.0)
    assert allclose(a.exp().data, math.e)


# ---------------------------------------------------------------------------
# Backward — gradient checks
# ---------------------------------------------------------------------------


def test_add_backward():
    a, b = wrap(2.0), wrap(3.0)
    c = a + b
    c.backward()
    assert allclose(a.grad, 1.0)
    assert allclose(b.grad, 1.0)


def test_mul_backward():
    a, b = wrap(4.0), wrap(-2.0)
    c = a * b
    c.backward()
    assert allclose(a.grad, -2.0)  # dc/da = b
    assert allclose(b.grad, 4.0)  # dc/db = a


def test_pow_backward():
    a = wrap(3.0)
    c = a**2
    c.backward()
    assert allclose(a.grad, 6.0)  # d/da a^2 = 2a = 6


def test_relu_backward_positive():
    a = wrap(1.5)
    out = a.relu()
    out.backward()
    assert allclose(a.grad, 1.0)


def test_relu_backward_negative():
    a = wrap(-0.5)
    out = a.relu()
    out.backward()
    assert allclose(a.grad, 0.0)


def test_tanh_backward():
    a = wrap(0.0)
    out = a.tanh()
    out.backward()
    # d/dx tanh(x)|_{x=0} = 1 - tanh(0)^2 = 1
    assert allclose(a.grad, 1.0)


def test_exp_backward():
    a = wrap(0.0)
    out = a.exp()
    out.backward()
    # d/dx e^x|_{x=0} = e^0 = 1
    assert allclose(a.grad, 1.0)


# ---------------------------------------------------------------------------
# Chain rule
# ---------------------------------------------------------------------------


def test_chain_rule_add_mul():
    # f(a, b, c) = (a + b) * c
    a, b, c = wrap(2.0), wrap(3.0), wrap(4.0)
    out = (a + b) * c
    out.backward()
    # df/da = c = 4, df/db = c = 4, df/dc = a+b = 5
    assert allclose(a.grad, 4.0)
    assert allclose(b.grad, 4.0)
    assert allclose(c.grad, 5.0)


def test_chain_rule_pow_mul():
    # f(a) = (a^2) * 3 → df/da = 6a
    a = wrap(2.0)
    out = (a**2) * wrap(3.0)
    out.backward()
    assert allclose(a.grad, 12.0)


# ---------------------------------------------------------------------------
# Operator overloads
# ---------------------------------------------------------------------------


def test_radd():
    a = wrap(3.0)
    c = 2.0 + a
    assert allclose(c.data, 5.0)


def test_sub():
    a, b = wrap(5.0), wrap(2.0)
    assert allclose((a - b).data, 3.0)


def test_rsub():
    a = wrap(2.0)
    assert allclose((5.0 - a).data, 3.0)


def test_rmul():
    a = wrap(4.0)
    assert allclose((3.0 * a).data, 12.0)


def test_truediv():
    a, b = wrap(10.0), wrap(2.0)
    assert allclose((a / b).data, 5.0)


def test_rtruediv():
    a = wrap(2.0)
    assert allclose((10.0 / a).data, 5.0)


def test_neg():
    a = wrap(3.0)
    assert allclose((-a).data, -3.0)


def test_repr():
    a = wrap(1.5)
    r = repr(a)
    assert "ScalarValue" in r
    assert "data=" in r
    assert "grad=" in r


# ---------------------------------------------------------------------------
# zero_grad pattern
# ---------------------------------------------------------------------------


def test_zero_grad_pattern():
    a = wrap(2.0)
    out = a**2
    out.backward()
    assert a.grad != 0.0
    # Simulate zero_grad by resetting.
    a.grad = 0.0
    out2 = a**2
    out2.backward()
    assert allclose(a.grad, 4.0)  # fresh gradient, not accumulated twice


# ---------------------------------------------------------------------------
# ScalarNeuron
# ---------------------------------------------------------------------------


def test_scalar_neuron_forward_shape():
    n = ScalarNeuron(nin=3, nonlin=True)
    x = [wrap(1.0), wrap(0.5), wrap(-1.0)]
    out = n(x)
    assert isinstance(out, ScalarValue)
    assert out.data >= 0.0  # relu output


def test_scalar_neuron_parameters_count():
    n = ScalarNeuron(nin=4)
    params = n.parameters()
    assert len(params) == 5  # 4 weights + 1 bias


def test_scalar_neuron_linear_no_relu():
    n = ScalarNeuron(nin=2, nonlin=False)
    x = [wrap(1.0), wrap(0.0)]
    out = n(x)
    # Output = w[0]*1 + w[1]*0 + b = w[0] + 0.0
    assert isinstance(out, ScalarValue)


# ---------------------------------------------------------------------------
# ScalarLayer
# ---------------------------------------------------------------------------


def test_scalar_layer_forward_shape():
    layer = ScalarLayer(nin=2, nout=3)
    x = [wrap(1.0), wrap(-1.0)]
    out = layer(x)
    assert len(out) == 3
    assert all(isinstance(v, ScalarValue) for v in out)


def test_scalar_layer_parameters_count():
    layer = ScalarLayer(nin=3, nout=2)
    # Each neuron has nin+1 params, nout neurons
    assert len(layer.parameters()) == 2 * (3 + 1)


# ---------------------------------------------------------------------------
# ScalarMLP — forward and backward
# ---------------------------------------------------------------------------


def test_mlp_forward_single_output():
    mlp = ScalarMLP(nin=2, nouts=[4, 1])
    x = [wrap(0.5), wrap(-0.5)]
    out = mlp(x)
    assert isinstance(out, ScalarValue)


def test_mlp_forward_multi_output():
    mlp = ScalarMLP(nin=2, nouts=[4, 3])
    x = [wrap(0.5), wrap(-0.5)]
    out = mlp(x)
    assert isinstance(out, list)
    assert len(out) == 3


def test_mlp_backward_populates_grads():
    mlp = ScalarMLP(nin=2, nouts=[3, 1])
    x = [wrap(1.0), wrap(-1.0)]
    out = mlp(x)
    out.backward()
    # All parameters should have non-None grads (float).
    for p in mlp.parameters():
        assert isinstance(p.grad, float)


def test_mlp_parameters_count():
    # nin=2, hidden=3, out=1
    # Layer 0: 3 neurons × (2+1) = 9
    # Layer 1: 1 neuron  × (3+1) = 4
    mlp = ScalarMLP(nin=2, nouts=[3, 1])
    assert len(mlp.parameters()) == 13


def test_mlp_gradient_descent_step():
    """Verify a single gradient-descent step reduces loss."""
    mlp = ScalarMLP(nin=1, nouts=[4, 1])
    x = [wrap(1.0)]
    target = 1.0
    lr = 0.01

    out = mlp(x)
    loss = (out - target) ** 2
    loss.backward()
    loss_before = loss.data

    # Apply SGD update.
    for p in mlp.parameters():
        p.data -= lr * p.grad
        p.grad = 0.0

    out2 = mlp(x)
    loss2 = (out2 - target) ** 2
    # Loss should have decreased (or stayed near zero).
    assert loss2.data <= loss_before + 1e-6
