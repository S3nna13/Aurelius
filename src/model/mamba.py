"""Mamba Selective State Space Model (SSM) layer.

Reference: Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
https://arxiv.org/abs/2312.00752

Pure PyTorch implementation — no custom CUDA kernels required.

Key idea: unlike linear SSMs where A, B, C are fixed, Mamba makes B, C, and Δ
INPUT-DEPENDENT (selective), enabling the model to selectively remember or forget
information at each timestep.

Discretized state equations (Zero-Order Hold):
    A_bar_t = exp(Δ_t * A)              [continuous A is negative; ZOH discretization]
    B_bar_t = Δ_t * B_t                 [simplified Euler discretization for B]
    h_t = A_bar_t * h_{t-1} + B_bar_t * x_t   [state update]
    y_t = C_t * h_t + D * x_t                  [output with skip connection]

where A is log-parameterized as A_log = log(|A|), actual A = -exp(A_log) (negative for stability).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rms_norm import RMSNorm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MambaConfig:
    """Hyperparameters for the Mamba selective SSM layer."""

    d_state: int = 16           # SSM state dimension (N in the paper)
    d_conv: int = 4             # depthwise conv kernel width
    expand: int = 2             # inner dim expansion factor (d_inner = expand * d_model)
    dt_rank: str | int = "auto" # if "auto" → ceil(d_model / 16)


# ---------------------------------------------------------------------------
# Selective SSM core
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """Selective state-space model core (S6).

    Given input x of shape (B, L, d_inner):
    1. Compute input-dependent params: Δ, B, C from x via linear projections.
    2. Discretize: A_bar = exp(Δ * A), B_bar = Δ * B  (ZOH approximation for B).
    3. Run SSM via sequential scan loop over sequence length.
    4. Return output y of shape (B, L, d_inner).

    State matrix A: (d_inner, d_state) — log-parameterized for numerical stability.
    A_log = log(arange(1, d_state+1)), actual A = -exp(A_log) (always negative → stable).
    """

    def __init__(self, d_model: int, d_inner: int, mamba_cfg: MambaConfig) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = mamba_cfg.d_state

        # Resolve dt_rank
        if isinstance(mamba_cfg.dt_rank, str) and mamba_cfg.dt_rank == "auto":
            self.dt_rank: int = math.ceil(d_model / 16)
        else:
            self.dt_rank = int(mamba_cfg.dt_rank)

        # Combined projection: x → (Δ_raw, B_in, C_in)
        # Output dim: dt_rank + 2 * d_state
        self.x_proj = nn.Linear(
            d_inner,
            self.dt_rank + 2 * mamba_cfg.d_state,
            bias=False,
        )

        # Δ projection: dt_rank → d_inner (with bias, per Mamba paper)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # A_log: log-parameterized state matrix
        # Shape (d_inner, d_state) — stores log(|A|), actual A = -exp(A_log)
        # Initialized as log-spaced: A_log[i, n] = log(n+1) for n in [0, d_state)
        A_log = (
            torch.log(torch.arange(1, mamba_cfg.d_state + 1).float())
            .unsqueeze(0)
            .expand(d_inner, -1)
        )  # (d_inner, d_state)
        self.A_log = nn.Parameter(A_log.clone())
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # D: skip connection — one scalar per d_inner channel, initialized to 1
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, d_inner)

        Returns:
            (B, L, d_inner)
        """
        B, L, D = x.shape  # D == d_inner

        # Step 1: compute input-dependent Δ, B_in, C_in from x
        xz = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta_raw, B_in, C_in = xz.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # delta_raw: (B, L, dt_rank)
        # B_in, C_in: (B, L, d_state)

        # Expand Δ from dt_rank → d_inner via dt_proj, then apply softplus
        delta = F.softplus(self.dt_proj(delta_raw))  # (B, L, d_inner)

        # Step 2: compute A = -exp(A_log) and discretize
        # A_log: (d_inner, d_state) → A: (d_inner, d_state) — always negative
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # A_bar: (B, L, d_inner, d_state) = exp(delta * A)
        # delta: (B, L, d_inner) → (B, L, d_inner, 1)
        # A: (d_inner, d_state) → (1, 1, d_inner, d_state)
        A_bar = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, L, d_inner, d_state)

        # B_bar: (B, L, d_inner, d_state) = delta * B_in
        # delta: (B, L, d_inner) → (B, L, d_inner, 1)
        # B_in: (B, L, d_state) → (B, L, 1, d_state)
        B_bar = delta.unsqueeze(-1) * B_in.unsqueeze(2)  # (B, L, d_inner, d_state)

        # Step 3: sequential scan
        # h: (B, d_inner, d_state) — hidden state initialized to zeros
        h = torch.zeros(B, self.d_inner, self.d_state, dtype=x.dtype, device=x.device)
        ys = []

        for t in range(L):
            # A_bar[:, t]: (B, d_inner, d_state)
            # B_bar[:, t]: (B, d_inner, d_state)
            # x[:, t]:     (B, d_inner) → (B, d_inner, 1)
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            # h: (B, d_inner, d_state)

            # y_t = sum(C_in[:, t] * h, dim=-1) + D * x[:, t]
            # C_in[:, t]: (B, d_state) → (B, 1, d_state)
            y_t = (h * C_in[:, t].unsqueeze(1)).sum(dim=-1) + self.D * x[:, t]
            # y_t: (B, d_inner)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, L, d_inner)


# ---------------------------------------------------------------------------
# MambaBlock — full Mamba block
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Full Mamba block: expand → conv → SSM → gate → contract.

    Architecture:
        x_in → in_proj → [x, z]  (d_model → 2*d_inner, split along last dim)
        x → depthwise causal conv1d (kernel=d_conv) → SiLU
        x → SelectiveSSM → y
        y = y * SiLU(z)          (multiplicative gating)
        out = out_proj(y)         (d_inner → d_model)

    Can be used as a drop-in replacement for attention in TransformerBlock.
    """

    def __init__(self, config: Any, mamba_cfg: MambaConfig | None = None) -> None:
        """
        Args:
            config: AureliusConfig (must have config.d_model).
            mamba_cfg: MambaConfig; uses defaults if None.
        """
        super().__init__()

        if mamba_cfg is None:
            mamba_cfg = MambaConfig()

        self.mamba_cfg = mamba_cfg
        d_model: int = config.d_model
        d_inner: int = mamba_cfg.expand * d_model
        self.d_inner = d_inner
        d_conv: int = mamba_cfg.d_conv

        # in_proj: d_model → 2 * d_inner (produces x branch and z gate)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise conv1d over x branch
        # groups=d_inner makes it depthwise; padding=d_conv-1 on left, then trim right
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            bias=True,
            padding=d_conv - 1,  # left-pad so output length >= input length
        )

        # Selective SSM core
        self.ssm = SelectiveSSM(d_model, d_inner, mamba_cfg)

        # out_proj: d_inner → d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, d_model)

        Returns:
            (B, L, d_model)
        """
        B, L, _ = x.shape

        # Project and split into x branch and z gate
        xz = self.in_proj(x)         # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal depthwise conv1d
        # Conv1d expects (B, C, L); padding=d_conv-1 on left → trim d_conv-1 from right
        x_conv = x_branch.transpose(1, 2)          # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)               # (B, d_inner, L + d_conv - 1)
        x_conv = x_conv[:, :, :L]                  # (B, d_inner, L) — trim right
        x_conv = x_conv.transpose(1, 2)            # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # Selective SSM
        y = self.ssm(x_conv)   # (B, L, d_inner)

        # Multiplicative gating
        y = y * F.silu(z)      # (B, L, d_inner)

        # Project back to d_model
        return self.out_proj(y)  # (B, L, d_model)


# ---------------------------------------------------------------------------
# MambaLayer — pre-norm residual wrapper
# ---------------------------------------------------------------------------

class MambaLayer(nn.Module):
    """Pre-norm residual wrapper around MambaBlock.

    out = x + MambaBlock(RMSNorm(x))

    Drop-in replacement for attention layers; accepts and ignores **kwargs
    for compatibility with the rest of the codebase (e.g., past_key_values).
    """

    def __init__(self, config: Any, mamba_cfg: MambaConfig | None = None) -> None:
        super().__init__()
        norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.norm = RMSNorm(config.d_model, eps=norm_eps)
        self.mamba = MambaBlock(config, mamba_cfg)

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """
        Args:
            x: (B, L, d_model)
            **kwargs: ignored — for API compatibility with attention layers
                      (e.g., past_key_values, freqs_cis, mask).

        Returns:
            (B, L, d_model)
        """
        return x + self.mamba(self.norm(x))
