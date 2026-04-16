"""Mamba-2 Structured State Space Duality (SSD) layer.

Reference: Dao & Gu 2024, "Transformers are SSMs: Generalized Models and Efficient
Algorithms Through Structured State Space Duality", arXiv:2405.21060.

Pure PyTorch implementation — no custom CUDA kernels, no external SSM libraries.

Key differences from Mamba-1:
- A_t is a *scalar* per head (not a full matrix per channel). This is the structural
  constraint that enables the SSD duality with linear attention.
- Multi-head design: input X is split into (n_heads, head_dim) analogous to attention.
- B and C are shared across heads (shape T×d_state), not per-channel.
- hidden state shape: (B, n_heads, head_dim, d_state) — outer product structure.

State equations (Section 4, Algorithm 1):
    h_t = A_t * h_{t-1} + B_t^T x_t        [scalar A_t × identity]
    y_t = C_t * h_t                          [output projection]

where:
    A_t ∈ R (scalar per head, via log parameterization)
    B_t ∈ R^{d_state}
    C_t ∈ R^{d_state}
    x_t ∈ R^{head_dim} (per head)
    h_t ∈ R^{head_dim × d_state} (outer product state)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Mamba2Config:
    """Hyperparameters for the Mamba-2 SSD layer."""

    n_heads: int = 8            # number of SSM heads (analogous to attention heads)
    d_state: int = 64           # SSM state dimension N (per B/C vector)
    head_dim: int = 64          # dimension per head (d_inner = n_heads * head_dim)
    expand: int = 2             # expansion factor for z gate projection
    dt_rank: str | int = "auto" # if "auto" → n_heads (one dt per head)


# ---------------------------------------------------------------------------
# SSD Core (Algorithm 1 from paper)
# ---------------------------------------------------------------------------

class SSDLayer(nn.Module):
    """Structured State Space Duality (SSD) core scan.

    Implements the recurrent formulation of Algorithm 1 from the paper.
    State: h ∈ R^{B × n_heads × head_dim × d_state}

    The key structural simplification vs Mamba-1:
        A_t = scalar × I  (scalar per head, shared across head_dim)
    This gives h_t = A_t * h_{t-1} + x_t ⊗ B_t (outer product update).

    Variable naming follows paper notation:
        Δ (delta): discretization step size
        A_log: log-parameterized decay (stored negative, so A = -exp(A_log))
        B: state input projection (T × d_state)
        C: state output projection (T × d_state)
        X: main SSM input (T × n_heads × head_dim)
        h: SSM hidden state (B × n_heads × head_dim × d_state)
    """

    def __init__(self, n_heads: int, head_dim: int, d_state: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_state = d_state

        # A_log: shape (n_heads,) — one scalar decay per head
        # Initialized negative so that A = -exp(A_log) < 0 → stable contraction
        # log-spaced initialization in [log(1), log(n_heads+1)] → negative after negation
        A_log_init = torch.log(
            torch.arange(1, n_heads + 1, dtype=torch.float32)
        )  # positive log values; stored as-is, applied as A = -exp(A_log)
        self.A_log = nn.Parameter(A_log_init)
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # D: skip connection scalar per head (one per n_heads, broadcast over head_dim)
        self.D = nn.Parameter(torch.ones(n_heads))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

    def forward(
        self,
        X: Tensor,           # (B, T, n_heads, head_dim)
        delta: Tensor,       # (B, T, n_heads) — after softplus + dt_bias
        B: Tensor,           # (B, T, d_state)
        C: Tensor,           # (B, T, d_state)
        h0: Optional[Tensor] = None,  # (B, n_heads, head_dim, d_state) or None
    ) -> Tuple[Tensor, Tensor]:
        """Run SSD recurrent scan.

        Returns:
            y: (B, T, n_heads, head_dim) — SSM outputs before skip
            h_last: (B, n_heads, head_dim, d_state) — final hidden state
        """
        batch_size, T, n_heads, head_dim = X.shape
        d_state = self.d_state
        device = X.device
        dtype = X.dtype

        if h0 is None:
            h = torch.zeros(
                batch_size, n_heads, head_dim, d_state,
                device=device, dtype=dtype
            )
        else:
            h = h0.to(dtype=dtype)

        ys = []
        for t in range(T):
            # Discretized A: scalar per head per batch step
            # dA: (B, n_heads) = exp(delta_t * A_log) — note: A = -exp(A_log)
            # We compute: dA = exp(delta * (-exp(A_log))) = exp(-delta * exp(A_log))
            # Equivalent to: dA = (-exp(A_log)).unsqueeze broadcast
            dA = torch.exp(
                delta[:, t, :] * (-torch.exp(self.A_log))
            )  # (B, n_heads)

            # dB: (B, n_heads, d_state) = delta_t * B_t, broadcast across heads
            # B[:, t, :]: (B, d_state) → (B, 1, d_state) broadcast to (B, n_heads, d_state)
            dB = delta[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1)
            # dB: (B, n_heads, d_state)

            # State update: h = dA * h + x_t ⊗ dB
            # h: (B, n_heads, head_dim, d_state)
            # dA: (B, n_heads) → (B, n_heads, 1, 1) for broadcast
            # X[:, t]: (B, n_heads, head_dim) → (B, n_heads, head_dim, 1) for outer product
            # dB: (B, n_heads, d_state) → (B, n_heads, 1, d_state) for outer product
            h = (
                dA.unsqueeze(-1).unsqueeze(-1) * h
                + X[:, t].unsqueeze(-1) * dB.unsqueeze(2)
            )  # (B, n_heads, head_dim, d_state)

            # Output: y_t = h @ C_t  (contract over d_state)
            # C[:, t]: (B, d_state) → (B, 1, 1, d_state) broadcast
            # sum over last dim (d_state) → (B, n_heads, head_dim)
            y_t = (h * C[:, t].unsqueeze(1).unsqueeze(1)).sum(dim=-1)
            # y_t: (B, n_heads, head_dim)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, T, n_heads, head_dim)
        return y, h


# ---------------------------------------------------------------------------
# Mamba2Block — full Mamba-2 block (Section 5)
# ---------------------------------------------------------------------------

class Mamba2Block(nn.Module):
    """Full Mamba-2 block implementing the SSD architecture.

    Architecture (Section 5 of paper):
        Input x: (B, T, d_model)

        1. Two linear projections:
           - z_proj: d_model → d_model*expand  → z gate
           - in_proj: d_model → d_state + d_state + n_heads + d_inner
                    → [B_ssm, C_ssm, dt_raw, X_flat]
                    where d_inner = n_heads * head_dim

        2. Reshape X_flat → (B, T, n_heads, head_dim)

        3. Discretize: Δ = softplus(dt_raw + dt_bias)

        4. SSD scan: (X, Δ, B_ssm, C_ssm, h0) → (y, h_last)

        5. Apply D skip: y = y + D * X

        6. Merge heads, gate: out_flat = y_flat * SiLU(z)

        7. Output projection: out_proj(out_flat) → (B, T, d_model)

    forward(x, hidden_state=None) → (output, new_hidden_state)
    where hidden_state has shape (B, n_heads, head_dim, d_state)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_state: int = 64,
        head_dim: int = 64,
        expand: int = 2,
        dt_rank: str | int = "auto",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_state = d_state
        self.head_dim = head_dim
        self.expand = expand

        # d_inner: total inner dimension = n_heads * head_dim
        self.d_inner = n_heads * head_dim

        # Resolve dt_rank (one Δ per head if "auto")
        if isinstance(dt_rank, str) and dt_rank == "auto":
            self.dt_rank: int = n_heads
        else:
            self.dt_rank = int(dt_rank)

        # Projection for z gate: d_model → d_model * expand
        # z is used as output gate (SiLU gating)
        self.z_proj = nn.Linear(d_model, d_model * expand, bias=False)

        # Combined projection for SSM inputs:
        # d_model → d_state (B_ssm) + d_state (C_ssm) + n_heads (dt_raw) + d_inner (X)
        in_proj_out = d_state + d_state + n_heads + self.d_inner
        self.in_proj = nn.Linear(d_model, in_proj_out, bias=False)

        # dt_bias: per-head bias added to dt_raw before softplus
        # Initialized with small positive values for stable initial Δ
        self.dt_bias = nn.Parameter(
            torch.full((n_heads,), fill_value=math.log(0.1))
        )

        # SSD core
        self.ssd = SSDLayer(n_heads=n_heads, head_dim=head_dim, d_state=d_state)

        # Output projection: d_inner → d_model  (after gating with z)
        # After gating, the z dimension is d_model*expand but we project d_inner → d_model
        # Gate: element-wise multiply with SiLU(z) broadcast or via separate proj
        # Following paper: gate z has shape d_model*expand; we project d_inner → d_model
        # and gate separately. For simplicity: gate z to d_inner first via linear.
        self.z_gate_proj = nn.Linear(d_model * expand, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        hidden_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, d_model)
            hidden_state: (B, n_heads, head_dim, d_state) or None → zero-init

        Returns:
            output: (B, T, d_model)
            new_hidden_state: (B, n_heads, head_dim, d_state)
        """
        B, T, _ = x.shape

        # --- Gate branch ---
        z = self.z_proj(x)  # (B, T, d_model * expand)

        # --- SSM input branch ---
        ssm_in = self.in_proj(x)  # (B, T, d_state + d_state + n_heads + d_inner)
        B_ssm, C_ssm, dt_raw, X_flat = ssm_in.split(
            [self.d_state, self.d_state, self.n_heads, self.d_inner], dim=-1
        )
        # B_ssm: (B, T, d_state)
        # C_ssm: (B, T, d_state)
        # dt_raw: (B, T, n_heads)
        # X_flat: (B, T, d_inner)

        # Reshape X into multi-head form
        X = X_flat.view(B, T, self.n_heads, self.head_dim)  # (B, T, n_heads, head_dim)

        # Discretize: Δ = softplus(dt_raw + dt_bias)
        # dt_bias: (n_heads,) → broadcast over (B, T, n_heads)
        delta = F.softplus(dt_raw + self.dt_bias)  # (B, T, n_heads)

        # SSD recurrent scan
        y, h_last = self.ssd(X, delta, B_ssm, C_ssm, h0=hidden_state)
        # y: (B, T, n_heads, head_dim)
        # h_last: (B, n_heads, head_dim, d_state)

        # Apply D skip connection: y = y + D * X
        # D: (n_heads,) → (1, 1, n_heads, 1) for broadcast
        y = y + self.ssd.D.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * X
        # y: (B, T, n_heads, head_dim)

        # Flatten heads: (B, T, d_inner)
        y_flat = y.reshape(B, T, self.d_inner)

        # Gate with SiLU(z)
        z_gated = self.z_gate_proj(F.silu(z))  # (B, T, d_inner)
        y_gated = y_flat * z_gated              # (B, T, d_inner)

        # Output projection
        output = self.out_proj(y_gated)  # (B, T, d_model)

        return output, h_last
