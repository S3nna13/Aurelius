"""Modern activation functions used in large language models.

Implements functional activations (swiglu, geglu, reglu, silu, quick_gelu,
squared_relu) as well as GLU-based FFN modules, an ActivationFactory, and a
simple benchmarking helper.

References:
  - Shazeer 2020: "GLU Variants Improve Transformer"
  - Ramachandran et al. 2017: "Searching for Activation Functions" (Swish/SiLU)
"""

from __future__ import annotations

import time
import statistics
from typing import Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Functional activations
# ---------------------------------------------------------------------------

def silu(x: Tensor) -> Tensor:
    """SiLU / Swish activation: x * sigmoid(x).

    Equivalent to ``torch.nn.functional.silu``.
    """
    return x * torch.sigmoid(x)


def quick_gelu(x: Tensor) -> Tensor:
    """Quick-GELU approximation: x * sigmoid(1.702 * x).

    Used by CLIP and some GPT-style models as a faster GELU substitute.
    """
    return x * torch.sigmoid(1.702 * x)


def squared_relu(x: Tensor) -> Tensor:
    """Squared ReLU: relu(x)^2.

    Used in Primer and some efficient transformer variants.
    """
    return F.relu(x) ** 2


# ---------------------------------------------------------------------------
# Gated Linear Unit variants (functional)
# ---------------------------------------------------------------------------

def swiglu(x: Tensor, gate: Tensor) -> Tensor:
    """SwiGLU gating: silu(gate) * x.

    Args:
        x:    Value tensor of shape (B, T, D).
        gate: Gate tensor of shape (B, T, D).

    Returns:
        Tensor of shape (B, T, D).
    """
    return silu(gate) * x


def geglu(x: Tensor, gate: Tensor) -> Tensor:
    """GeGLU gating: gelu(gate) * x.

    Args:
        x:    Value tensor of shape (B, T, D).
        gate: Gate tensor of shape (B, T, D).

    Returns:
        Tensor of shape (B, T, D).
    """
    return F.gelu(gate) * x


def reglu(x: Tensor, gate: Tensor) -> Tensor:
    """ReGLU gating: relu(gate) * x.

    Args:
        x:    Value tensor of shape (B, T, D).
        gate: Gate tensor of shape (B, T, D).

    Returns:
        Tensor of shape (B, T, D).
    """
    return F.relu(gate) * x


# ---------------------------------------------------------------------------
# GLU Feed-Forward Network modules
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """Feed-forward network using SwiGLU gating.

    Architecture:
        hidden = swiglu(w1(x), w2(x))   # (B, T, d_ff)
        output = w3(hidden)              # (B, T, d_model)

    All linear layers are bias-free.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # value projection
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # gate projection
        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # output projection

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(swiglu(self.w1(x), self.w2(x)))


class GeGLUFFN(nn.Module):
    """Feed-forward network using GeGLU gating.

    Architecture:
        hidden = geglu(w1(x), w2(x))    # (B, T, d_ff)
        output = w3(hidden)             # (B, T, d_model)

    All linear layers are bias-free.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(geglu(self.w1(x), self.w2(x)))


# ---------------------------------------------------------------------------
# Activation factory
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[[Tensor], Tensor]] = {
    "silu": silu,
    "swish": silu,
    "gelu": F.gelu,
    "relu": F.relu,
    "quick_gelu": quick_gelu,
    "squared_relu": squared_relu,
}


class ActivationFactory:
    """Simple registry for single-input activation functions."""

    @staticmethod
    def get(name: str) -> Callable[[Tensor], Tensor]:
        """Return the activation function registered under *name*.

        Args:
            name: One of ``"silu"``, ``"swish"``, ``"gelu"``, ``"relu"``,
                  ``"quick_gelu"``, ``"squared_relu"``.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in _REGISTRY:
            raise KeyError(
                f"Unknown activation '{name}'. "
                f"Available: {ActivationFactory.list_available()}"
            )
        return _REGISTRY[name]

    @staticmethod
    def list_available() -> List[str]:
        """Return sorted list of all registered activation names."""
        return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Benchmarking helper
# ---------------------------------------------------------------------------

def benchmark_activation(
    fn: Callable[[Tensor], Tensor],
    x: Tensor,
    n_runs: int = 10,
) -> Dict[str, float | str]:
    """Time *fn* applied to *x* over *n_runs* forward passes.

    Args:
        fn:     Single-argument callable that takes a Tensor and returns a Tensor.
        x:      Input tensor (any shape).
        n_runs: Number of timed repetitions.

    Returns:
        Dictionary with keys:
            ``"mean_ms"``      – mean wall-clock time in milliseconds,
            ``"std_ms"``       – std deviation of times in milliseconds,
            ``"output_shape"`` – string representation of the output shape.
    """
    # Warm-up pass (not timed)
    with torch.no_grad():
        out = fn(x)

    times_ms: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            fn(x)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1_000.0)

    return {
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.pstdev(times_ms),
        "output_shape": str(tuple(out.shape)),
    }
