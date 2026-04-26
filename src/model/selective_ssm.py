"""Simplified Selective State Space Model (Mamba-style) layer.

Pure PyTorch implementation of the core SSM scan and Mamba block.
Uses sequential scan rather than CUDA-parallel scan for portability.

Reference:
    Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    https://arxiv.org/abs/2312.00752
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SSMConfig:
    """Configuration for the Selective SSM block."""

    d_model: int = 64
    d_state: int = 16  # N — SSM state dimension
    d_conv: int = 4  # causal depthwise conv kernel size
    expand: int = 2  # inner_dim = d_model * expand
    dt_rank: int = 8  # rank of delta (Δ) projection
    dt_min: float = 0.001  # minimum Δ after softplus
    dt_max: float = 0.1  # maximum Δ clamp

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand


# ---------------------------------------------------------------------------
# Sequential SSM scan (pure PyTorch)
# ---------------------------------------------------------------------------


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Sequential selective SSM scan.

    Discretizes and applies the SSM recurrence:
        h_t = A_bar_t * h_{t-1} + B_bar_t * u_t
        y_t = (h_t * C_t).sum(-1)

    Args:
        u:     (B, L, D) — input sequences.
        delta: (B, L, D) — softplus-activated step sizes.
        A:     (D, N)    — state matrix (stored as negative values for stability).
        B:     (B, L, N) — input projection.
        C:     (B, L, N) — output projection.

    Returns:
        (B, L, D) output tensor.
    """
    B_size, L, D = u.shape
    N = A.shape[1]

    # Discretize:
    # A_bar_t = exp(delta_t * A)  → (B, L, D, N)
    # B_bar_t = delta_t * B_t     → (B, L, D, N)  (outer product over D and N)
    delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
    delta_B_u = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)  — broadcast D over N
    # Multiply by u to get the input contribution
    delta_B_u = delta_B_u * u.unsqueeze(-1)  # (B, L, D, N)

    # Sequential scan
    h = torch.zeros(B_size, D, N, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = delta_A[:, t, :, :] * h + delta_B_u[:, t, :, :]  # (B, D, N)
        # y_t = (h * C_t).sum(-1) → (B, D)
        y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1)  # (B, D)
        ys.append(y_t)

    return torch.stack(ys, dim=1)  # (B, L, D)


# ---------------------------------------------------------------------------
# SelectiveSSM — inner SSM module (no gating)
# ---------------------------------------------------------------------------


class SelectiveSSM(nn.Module):
    """Input-dependent SSM: projects input to Δ, B, C parameters."""

    def __init__(self, config: SSMConfig) -> None:
        super().__init__()
        D = config.d_inner
        N = config.d_state

        # A is stored as log for numerical stability; init with Hippo-style values
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(0).expand(D, N))
        )
        # Skip connection scale
        self.D_skip = nn.Parameter(torch.ones(D))

        # Project x → (dt_rank + 2*N)
        self.x_proj = nn.Linear(D, config.dt_rank + 2 * N, bias=False)
        # Project delta rank → D
        self.dt_proj = nn.Linear(config.dt_rank, D, bias=True)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D_inner)

        Returns:
            (B, L, D_inner)
        """
        projected = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        dt_rank = self.config.dt_rank
        N = self.config.d_state

        delta_raw = projected[..., :dt_rank]  # (B, L, dt_rank)
        B = projected[..., dt_rank : dt_rank + N]  # (B, L, N)
        C = projected[..., dt_rank + N :]  # (B, L, N)

        # Δ: project and activate
        delta = F.softplus(self.dt_proj(delta_raw))  # (B, L, D_inner)

        # A: -exp(A_log) keeps A negative (stable decay)
        A = -torch.exp(self.A_log.float())  # (D_inner, N)

        y = selective_scan(x.float(), delta.float(), A, B.float(), C.float())
        y = y.to(x.dtype)

        # Skip connection (D vector broadcast over B, L)
        y = y + x * self.D_skip

        return y


# ---------------------------------------------------------------------------
# MambaBlock — full Mamba block with gating
# ---------------------------------------------------------------------------


class MambaBlock(nn.Module):
    """Full Mamba block: conv1d + SelectiveSSM + SiLU gating."""

    def __init__(self, config: SSMConfig) -> None:
        super().__init__()
        self.config = config
        D = config.d_model
        D_inner = config.d_inner

        # Project to 2 * D_inner (split into x and z gate)
        self.in_proj = nn.Linear(D, 2 * D_inner, bias=False)

        # Causal depthwise conv
        self.conv1d = nn.Conv1d(
            in_channels=D_inner,
            out_channels=D_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=D_inner,
            bias=True,
        )

        self.ssm = SelectiveSSM(config)

        self.out_proj = nn.Linear(D_inner, D, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (B, L, d_model)

        Returns:
            (B, L, d_model)
        """
        B, L, D = hidden.shape

        # Project and split
        xz = self.in_proj(hidden)  # (B, L, 2*D_inner)
        x, z = xz.chunk(2, dim=-1)  # each (B, L, D_inner)

        # Causal conv (conv over time, channel-last → channel-first)
        x_conv = self.conv1d(x.transpose(1, 2))  # (B, D_inner, L + pad)
        x_conv = x_conv[:, :, :L].transpose(1, 2)  # (B, L, D_inner)

        x_conv = F.silu(x_conv)

        # SSM
        y = self.ssm(x_conv)  # (B, L, D_inner)

        # SiLU gate
        y = y * F.silu(z)

        return self.out_proj(y)  # (B, L, d_model)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / norm * self.weight).to(x.dtype)


# ---------------------------------------------------------------------------
# MambaLayer — block + residual + pre-norm
# ---------------------------------------------------------------------------


class MambaLayer(nn.Module):
    """MambaBlock with RMSNorm pre-norm and residual connection."""

    def __init__(self, config: SSMConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.block = MambaBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)

        Returns:
            (B, L, d_model) — x + MambaBlock(RMSNorm(x))
        """
        return x + self.block(self.norm(x))
