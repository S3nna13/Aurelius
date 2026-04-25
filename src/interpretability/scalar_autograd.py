"""Micrograd-style scalar reverse-mode autograd for gradient debugging.

Provides ScalarValue (single float with autograd), ScalarNeuron, ScalarLayer,
and ScalarMLP — a minimal neural network stack backed entirely by scalar ops.
Useful for verifying gradient correctness and tracing computation graphs.
"""

import math
from typing import Callable


class ScalarValue:
    """A scalar value with automatic differentiation support."""

    def __init__(self, data: float, _prev: set = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable = lambda: None
        self._prev = set(_prev)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, ScalarValue) else ScalarValue(other)
        out = ScalarValue(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, ScalarValue) else ScalarValue(other)
        out = ScalarValue(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exp: float):
        out = ScalarValue(self.data ** exp, (self,), f"**{exp}")

        def _backward():
            self.grad += exp * (self.data ** (exp - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = ScalarValue(max(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = ScalarValue(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = ScalarValue(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __repr__(self):
        return f"ScalarValue(data={self.data:.4f}, grad={self.grad:.4f})"


class ScalarNeuron:
    """Single neuron with scalar autograd."""

    def __init__(self, nin: int, nonlin: bool = True):
        import random
        self.w = [ScalarValue(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = ScalarValue(0.0)
        self.nonlin = nonlin

    def __call__(self, x: list) -> ScalarValue:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> list[ScalarValue]:
        return self.w + [self.b]


class ScalarLayer:
    """A layer of ScalarNeurons."""

    def __init__(self, nin: int, nout: int, nonlin: bool = True):
        self.neurons = [ScalarNeuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x: list) -> list[ScalarValue]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[ScalarValue]:
        return [p for n in self.neurons for p in n.parameters()]


class ScalarMLP:
    """Multi-layer perceptron built from ScalarLayers."""

    def __init__(self, nin: int, nouts: list[int]):
        sz = [nin] + nouts
        self.layers = [
            ScalarLayer(sz[i], sz[i + 1], nonlin=(i < len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x: list) -> "ScalarValue | list[ScalarValue]":
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self) -> list[ScalarValue]:
        return [p for layer in self.layers for p in layer.parameters()]
