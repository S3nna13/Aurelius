"""Structured State Space Model (S4 / S4D) layer.

Gu et al. 2021 — https://arxiv.org/abs/2111.00396 (S4)
Gu et al. 2022 — https://arxiv.org/abs/2206.11893 (S4D — diagonal variant)

Implements a diagonal SSM with ZOH discretization and sequential scan.
The state space: x_t = A_bar * x_{t-1} + B_bar * u_t
                 y_t = Re(C * x_t) + D * u_t
where A_bar = exp(delta * A), B_bar = (A_bar - I) * A^{-1} * B (ZOH).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class S4DKernel(nn.Module):
    """Diagonal SSM convolution kernel (S4D).

    Parametrizes a diagonal state matrix A = A_real + i * A_imag with
    A_real forced negative via softplus for stability.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # log_A_real: real part of diagonal A; A_real = -softplus(log_A_real) < 0
        self.log_A_real = nn.Parameter(
            torch.log(0.5 * torch.ones(d_model, d_state))
        )

        # A_imag: imaginary part of diagonal A
        A_imag_init = torch.linspace(0, math.pi, d_state)  # (d_state,)
        self.A_imag = nn.Parameter(
            A_imag_init.unsqueeze(0).expand(d_model, d_state).clone()
        )

        # B: complex input projection, stored as (real, imag)
        self.B = nn.Parameter(torch.randn(d_model, d_state, 2))

        # C: complex output projection, stored as (real, imag)
        self.C = nn.Parameter(torch.randn(d_model, d_state, 2))

        # log_dt: log timescale per channel
        log_dt_init = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt_init)

        # D: skip connection
        self.D = nn.Parameter(torch.ones(d_model))

    def get_kernel(self, L: int) -> Tensor:
        """Compute the SSM convolution kernel of length L.

        Returns K of shape (d_model, L) where
          K[d, l] = Re(sum_n C[d,n] * A_bar[d,n]^l * B_bar[d,n])
        and B_bar is derived from the ZOH discretization.

        Args:
            L: sequence length.

        Returns:
            Tensor of shape (d_model, L).
        """
        # dt: (d_model,)
        dt = F.softplus(self.log_dt)

        # A_real: (d_model, d_state) — forced negative
        A_real = -F.softplus(self.log_A_real)

        # A_complex: (d_model, d_state)
        A_complex = torch.complex(A_real, self.A_imag)

        # A_bar = exp(dt * A): (d_model, d_state)
        A_bar = torch.exp(dt[:, None] * A_complex)

        # B_complex: (d_model, d_state)
        B_complex = torch.complex(self.B[..., 0], self.B[..., 1])

        # C_complex: (d_model, d_state)
        C_complex = torch.complex(self.C[..., 0], self.C[..., 1])

        # Compute A_bar^l for l = 0 .. L-1 via broadcasting
        # l_indices: (L,)
        l_indices = torch.arange(L, device=A_bar.device, dtype=torch.float32)
        # A_pow: (d_model, d_state, L)
        A_pow = A_bar[:, :, None] ** l_indices[None, None, :]

        # K = sum over d_state of Re(C * B * A^l): (d_model, L)
        K = (C_complex[:, :, None] * B_complex[:, :, None] * A_pow).sum(dim=1).real

        return K  # (d_model, L)


class S4DLayer(nn.Module):
    """Single S4D layer: depthwise causal convolution via the SSM kernel."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.kernel = S4DKernel(d_model, d_state)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, u: Tensor) -> Tensor:
        """Apply the S4D layer.

        Args:
            u: (B, L, d_model)

        Returns:
            Tensor of shape (B, L, d_model).
        """
        B, L, d_model = u.shape

        # Compute SSM convolution kernel: (d_model, L)
        K = self.kernel.get_kernel(L)

        # u: (B, d_model, L) for conv1d
        u_perm = u.permute(0, 2, 1)

        # Causal convolution: pad left by L-1
        u_padded = F.pad(u_perm, (L - 1, 0))

        # Depthwise conv1d: weight shape (d_model, 1, L), groups=d_model
        y_conv = F.conv1d(u_padded, K.unsqueeze(1), groups=d_model)[:, :, :L]
        # y_conv: (B, d_model, L)

        # Skip connection
        y = y_conv + self.kernel.D[None, :, None] * u_perm

        # Back to (B, L, d_model)
        y = y.permute(0, 2, 1)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.output_linear(y)

        return y


class S4DBlock(nn.Module):
    """S4D block with pre-norm residuals and feed-forward sublayer."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.norm = nn.LayerNorm(d_model)
        self.ssm = S4DLayer(d_model, d_state, dropout)

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pre-norm residual forward pass.

        Args:
            x: (B, L, d_model)

        Returns:
            Tensor of shape (B, L, d_model).
        """
        # SSM sublayer
        x = x + self.ssm(self.norm(x))
        # Feed-forward sublayer
        x = x + self.ff(self.ff_norm(x))
        return x
