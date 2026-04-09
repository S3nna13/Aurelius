"""Simplified Mamba-inspired Selective State Space Model (SSM) layer.

A linear-complexity alternative to attention using selective state spaces.

Reference: Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
https://arxiv.org/abs/2312.00752

Key equations (ZOH discretization, simplified):
    A_bar_t = exp(delta_t * A)
    B_bar_t = delta_t * B_t
    h_t = A_bar_t * h_{t-1} + B_bar_t * u_t
    y_t = C_t @ h_t
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SSMConfig:
    """Hyperparameters for the simplified Mamba-inspired SSM layer."""

    d_model: int = 64
    d_state: int = 16      # state dimension (N in S4/Mamba)
    d_conv: int = 4        # depthwise conv kernel size
    expand: int = 2        # expansion factor (d_inner = expand * d_model)
    dt_rank: int = 4       # rank of delta-t projection
    dt_min: float = 0.001  # min timescale
    dt_max: float = 0.1    # max timescale


# ---------------------------------------------------------------------------
# A-matrix initialization
# ---------------------------------------------------------------------------

def init_A_matrix(d_model: int, d_state: int) -> Tensor:
    """Initialize log_A ~ U(0, 1).

    Returns a (d_model, d_state) tensor of log_A values (before negation).
    The actual A matrix used in the scan is -exp(log_A), which is always negative
    (stable).
    """
    return torch.rand(d_model, d_state)


# ---------------------------------------------------------------------------
# Selective scan (sequential Python loop — correctness over speed)
# ---------------------------------------------------------------------------

def selective_scan(
    u: Tensor,      # (B, T, D) input
    A: Tensor,      # (D, N) state matrix (log-parameterized, already negated)
    B: Tensor,      # (B, T, N) input projection (selective)
    C: Tensor,      # (B, T, N) output projection (selective)
    delta: Tensor,  # (B, T, D) timescales (positive after softplus)
) -> Tensor:
    """Discretized SSM scan (simplified sequential scan, not parallel).

    Discretize:
        A_bar = exp(delta * A)          [ZOH for A]
        B_bar = delta * B               [simplified ZOH for B]
    State update:
        h_t = A_bar_t * h_{t-1} + B_bar_t * u_t   (element-wise)
    Output:
        y_t = sum_N( C_t * h_t )       → (B, D)

    Returns (B, T, D) output.
    """
    B_size, T, D = u.shape
    N = A.shape[1]

    h = torch.zeros(B_size, D, N, device=u.device, dtype=u.dtype)
    ys = []

    for t in range(T):
        delta_t = delta[:, t, :]          # (B, D)
        u_t = u[:, t, :]                  # (B, D)
        B_t = B[:, t, :]                  # (B, N)
        C_t = C[:, t, :]                  # (B, N)

        # Discretize
        # A_bar_t: (B, D, N)
        A_bar_t = torch.exp(delta_t[:, :, None] * A[None, :, :])
        # B_bar_t: (B, D, N)  — broadcast B_t from (B, N) to (B, D, N)
        B_bar_t = delta_t[:, :, None] * B_t[:, None, :]

        # State update: (B, D, N)
        h = A_bar_t * h + B_bar_t * u_t[:, :, None]

        # Output: sum over N dimension: C_t[:, None, :] * h → (B, D, N) → sum → (B, D)
        y_t = (C_t[:, None, :] * h).sum(dim=-1)  # (B, D)
        ys.append(y_t)

    # Stack along time dimension → (B, T, D)
    return torch.stack(ys, dim=1)


# ---------------------------------------------------------------------------
# SSM Layer
# ---------------------------------------------------------------------------

class SSMLayer(nn.Module):
    """Mamba-inspired SSM layer with selective state spaces."""

    def __init__(self, cfg: SSMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_inner = cfg.expand * cfg.d_model

        # Input projection: split into x and z (gating)
        self.in_proj = nn.Linear(cfg.d_model, 2 * d_inner, bias=False)

        # Depthwise conv over the inner sequence
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, cfg.d_conv,
            padding=cfg.d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        # Project x to (dt_rank + 2 * d_state)
        self.x_proj = nn.Linear(d_inner, cfg.dt_rank + 2 * cfg.d_state, bias=False)

        # Low-rank delta projection
        self.dt_proj = nn.Linear(cfg.dt_rank, d_inner, bias=True)

        # State matrix (log-parameterized)
        self.log_A = nn.Parameter(init_A_matrix(d_inner, cfg.d_state))

        # Skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)

        # Pre-norm
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, D) → (B, T, D)."""
        B, T, _ = x.shape

        # 1. Residual connection
        residual = x

        # 2. Layer norm
        x = self.norm(x)

        # 3. Input projection + gating split
        xz = self.in_proj(x)                               # (B, T, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)                   # each (B, T, d_inner)

        # 4. Depthwise conv on x
        #    Conv1d expects (B, C, T); padding adds d_conv-1 extra steps at end
        x_conv = self.conv1d(x_inner.transpose(1, 2))       # (B, d_inner, T + d_conv - 1)
        x_conv = x_conv[:, :, :T].transpose(1, 2)           # (B, T, d_inner)

        # 5. SiLU activation
        x_inner = F.silu(x_conv)

        # 6. Project to (dt_rank + 2*d_state)
        xbc = self.x_proj(x_inner)                          # (B, T, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = xbc.split(
            [self.cfg.dt_rank, self.cfg.d_state, self.cfg.d_state], dim=-1
        )
        # dt: (B, T, dt_rank), B_ssm: (B, T, d_state), C_ssm: (B, T, d_state)

        # 7. Delta: project, apply softplus, clamp
        dt = F.softplus(self.dt_proj(dt)) + self.cfg.dt_min  # (B, T, d_inner)
        dt = dt.clamp(self.cfg.dt_min, self.cfg.dt_max)

        # 8. Actual A (negative for stability)
        A = -torch.exp(self.log_A)                           # (d_inner, d_state)

        # 9. Selective scan
        y = selective_scan(x_inner, A, B_ssm, C_ssm, dt)    # (B, T, d_inner)

        # 10. Skip connection with D
        y = y + x_inner * self.D                             # (B, T, d_inner)

        # 11. Gate with z
        y = y * F.silu(z)                                    # (B, T, d_inner)

        # 12. Output projection + residual
        output = self.out_proj(y) + residual                 # (B, T, d_model)
        return output


# ---------------------------------------------------------------------------
# SSM Block (transformer-like block using SSM instead of attention)
# ---------------------------------------------------------------------------

class SSMBlock(nn.Module):
    """Transformer-like block using SSM instead of attention."""

    def __init__(self, cfg: SSMConfig) -> None:
        super().__init__()
        self.ssm = SSMLayer(cfg)
        self.ffn_norm = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x + ssm(x), then x + ffn(norm(x))."""
        x = self.ssm(x)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Parameter counter
# ---------------------------------------------------------------------------

def count_ssm_parameters(cfg: SSMConfig) -> int:
    """Count total trainable parameters in one SSMLayer."""
    layer = SSMLayer(cfg)
    return sum(p.numel() for p in layer.parameters())
