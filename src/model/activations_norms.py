"""Alternative activation functions and normalization layers for modern LLMs.

Implements:
  - swish / SiLU functional activation
  - gelu_tanh approximation
  - SwiGLU and GeGLU gated FFN modules
  - RMSNorm, ScaleNorm, CRMSNorm normalization layers
  - compute_activation_stats utility

References:
  - Shazeer 2020: "GLU Variants Improve Transformer"
  - Zhang & Sennrich 2019: "Root Mean Square Layer Normalization"
  - Nguyen & Salazar 2019: "Transformers without Tears" (ScaleNorm)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Functional activations
# ---------------------------------------------------------------------------


def swish(x: Tensor) -> Tensor:
    """Swish / SiLU activation: x * sigmoid(x).

    Args:
        x: Input tensor of any shape.

    Returns:
        Tensor of the same shape as ``x``.
    """
    return x * torch.sigmoid(x)


def gelu_tanh(x: Tensor) -> Tensor:
    """Gaussian GELU approximation using tanh.

    Formula:
        0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Args:
        x: Input tensor of any shape.

    Returns:
        Tensor of the same shape as ``x``.
    """
    coeff = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(coeff * (x + 0.044715 * x.pow(3))))


# ---------------------------------------------------------------------------
# Gated FFN modules
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    Computes: W2( swish(W(x)) * V(x) )

    All linear projections have no bias, following common LLM practice.

    Args:
        d_model: Input/output dimension.
        d_ffn:   Hidden (gate) dimension.
    """

    def __init__(self, d_model: int, d_ffn: int) -> None:
        super().__init__()
        self.W = nn.Linear(d_model, d_ffn, bias=False)
        self.V = nn.Linear(d_model, d_ffn, bias=False)
        self.W2 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, T, d_model)

        Returns:
            Tensor of shape (B, T, d_model).
        """
        gate = swish(self.W(x))  # (B, T, d_ffn)
        value = self.V(x)  # (B, T, d_ffn)
        return self.W2(gate * value)


class GeGLU(nn.Module):
    """GeGLU feed-forward block.

    Computes: W2( gelu(W(x)) * V(x) )

    All linear projections have no bias.

    Args:
        d_model: Input/output dimension.
        d_ffn:   Hidden (gate) dimension.
    """

    def __init__(self, d_model: int, d_ffn: int) -> None:
        super().__init__()
        self.W = nn.Linear(d_model, d_ffn, bias=False)
        self.V = nn.Linear(d_model, d_ffn, bias=False)
        self.W2 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, T, d_model)

        Returns:
            Tensor of shape (B, T, d_model).
        """
        gate = F.gelu(self.W(x))  # (B, T, d_ffn)
        value = self.V(x)  # (B, T, d_ffn)
        return self.W2(gate * value)


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes each feature vector by its RMS and scales by a learnable
    per-dimension parameter gamma (initialized to ones).

    Args:
        d_model: Feature dimension (last dim of input).
        eps:     Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Tensor of any shape where the last dim is d_model.

        Returns:
            Normalized tensor of the same shape as ``x``.
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.gamma


class ScaleNorm(nn.Module):
    """ScaleNorm normalization (Nguyen & Salazar 2019).

    Normalizes each (d_model,) vector to have L2-norm equal to a learnable
    scalar g (initialized to sqrt(d_model)).

    Args:
        d_model: Feature dimension (last dim of input).
        eps:     Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.tensor(float(d_model) ** 0.5))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Tensor of any shape where the last dim is d_model.

        Returns:
            Tensor of the same shape as ``x`` with per-vector L2 norm == g.
        """
        norm = x.norm(dim=-1, keepdim=True)  # L2 norm per vector
        return x / (norm / self.g + self.eps)


class CRMSNorm(nn.Module):
    """Conditional RMSNorm.

    Applies RMSNorm without learnable parameters; scale and shift are
    provided externally (e.g. from a conditioning signal).

    Formula: RMSNorm(x) * (1 + scale) + shift

    Args:
        d_model: Feature dimension.
        eps:     Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x:     (B, T, d_model)
            scale: (B, d_model) or (d_model,) — multiplicative conditioning.
            shift: (B, d_model) or (d_model,) — additive conditioning.

        Returns:
            Tensor of shape (B, T, d_model).
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms  # (B, T, d_model)

        # Broadcast scale/shift from (B, d_model) → (B, 1, d_model) if needed
        if scale.dim() == 2:
            scale = scale.unsqueeze(1)
        if shift.dim() == 2:
            shift = shift.unsqueeze(1)

        return x_norm * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def compute_activation_stats(x: Tensor) -> dict[str, float]:
    """Compute basic statistics of an activation tensor.

    Args:
        x: Tensor of any shape.

    Returns:
        Dictionary with keys:
          - "mean": scalar mean of all elements.
          - "std": scalar std of all elements.
          - "fraction_positive": fraction of elements > 0.
          - "max_abs": maximum absolute value.
    """
    x_flat = x.detach().float().reshape(-1)
    return {
        "mean": x_flat.mean().item(),
        "std": x_flat.std().item(),
        "fraction_positive": (x_flat > 0).float().mean().item(),
        "max_abs": x_flat.abs().max().item(),
    }
