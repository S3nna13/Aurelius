"""Mamba-2 Structured State Space Duality (SSD) — v2 implementation.

Reference: Dao & Gu 2024, "Transformers are SSMs: Generalized Models and Efficient
Algorithms Through Structured State Space Duality", arXiv:2405.21060.

Pure PyTorch implementation — no custom CUDA kernels, no mamba_ssm, no causal_conv1d.

Key design (SSD):
- A_t is a scalar per head (structural constraint enabling semiseparable duality).
- Multi-head layout: x is (B, L, n_heads, head_dim), mirroring attention.
- B and C are per-head vectors of size d_state.
- Hidden state: outer-product structure (head_dim x d_state) per head.
- delta is a learned per-head log-timescale; A_bar = exp(-softplus(delta) * |A|).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Mamba2Config:
    d_model: int
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 8
    head_dim: Optional[int] = None   # auto: d_inner // n_heads
    chunk_size: int = 64

    def __post_init__(self) -> None:
        self.d_inner: int = self.d_model * self.expand
        if self.head_dim is None:
            assert self.d_inner % self.n_heads == 0, (
                f"d_inner={self.d_inner} must be divisible by n_heads={self.n_heads}"
            )
            self.head_dim = self.d_inner // self.n_heads


# ---------------------------------------------------------------------------
# SSMKernel — sequential scan
# ---------------------------------------------------------------------------

class SSMKernel(nn.Module):
    """Structured State Space Duality kernel (sequential scan).

    Computes:
        h_t = A_bar_t * h_{t-1} + B_t x_t^T   (outer product state update)
        y_t = C_t @ h_t                          (read-out)

    where:
        A_bar_t = exp(-softplus(delta_t) * |A_t|)   (discretised scalar per head)
        B_t, C_t in R^{d_state}                      (per head)
        x_t in R^{head_dim}                           (per head)
        h_t in R^{head_dim x d_state}                 (outer product state)
    """

    def __init__(self, d_state: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.d_state = d_state
        self.n_heads = n_heads
        self.head_dim = head_dim
        # Learnable log-timescale shift for delta (per head)
        self.delta_param = nn.Parameter(torch.zeros(n_heads))

    def forward(
        self,
        x: Tensor,       # (B, L, nh, dh)
        A: Tensor,       # (B, L, nh)       — raw A logits
        B_mat: Tensor,   # (B, L, nh, ds)
        C_mat: Tensor,   # (B, L, nh, ds)
    ) -> Tensor:
        """Return y with same shape as x: (B, L, nh, dh)."""
        B_sz, L, nh, dh = x.shape
        ds = self.d_state

        # Discretise: delta = softplus(raw_delta + delta_param), A_bar = exp(-delta * |A|)
        delta = F.softplus(A + self.delta_param[None, None, :])  # (B, L, nh)
        A_bar = torch.exp(-delta * A.abs())                       # (B, L, nh)

        # Sequential scan over time
        # h shape: (B, nh, dh, ds)
        h = x.new_zeros(B_sz, nh, dh, ds)
        ys = []
        for t in range(L):
            x_t = x[:, t, :, :]          # (B, nh, dh)
            B_t = B_mat[:, t, :, :]      # (B, nh, ds)
            C_t = C_mat[:, t, :, :]      # (B, nh, ds)
            a_t = A_bar[:, t, :]         # (B, nh)

            # h_t = A_bar_t * h_{t-1} + x_t outer B_t
            # outer product: (B, nh, dh, 1) * (B, nh, 1, ds) -> (B, nh, dh, ds)
            h = a_t[:, :, None, None] * h + (
                x_t[:, :, :, None] * B_t[:, :, None, :]
            )

            # y_t = h_t @ C_t  -> (B, nh, dh, ds) @ (B, nh, ds, 1) -> (B, nh, dh)
            y_t = (h * C_t[:, :, None, :]).sum(-1)  # (B, nh, dh)
            ys.append(y_t)

        # Stack: (B, L, nh, dh)
        y = torch.stack(ys, dim=1)
        return y


# ---------------------------------------------------------------------------
# Mamba2Block
# ---------------------------------------------------------------------------

class Mamba2Block(nn.Module):
    """Single Mamba-2 block with SSD kernel, conv1d mixing, and output gate."""

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_inner = config.d_inner
        d_state = config.d_state
        n_heads = config.n_heads
        head_dim = config.head_dim
        d_conv = config.d_conv

        # Input projection: x + B + C + dt
        self.in_proj = nn.Linear(
            d_model,
            d_inner + d_state * n_heads * 2 + n_heads,
            bias=False,
        )

        # Depthwise conv on x (causal, no future leakage)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            bias=True,
        )
        self._conv_pad = d_conv - 1

        # SSM kernel
        self.ssm = SSMKernel(d_state=d_state, n_heads=n_heads, head_dim=head_dim)

        # Output gate
        self.W_gate = nn.Linear(d_model, d_inner, bias=False)

        # RMSNorm before output projection
        self.norm = nn.RMSNorm(d_inner)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, u: Tensor) -> Tensor:
        """u: (B, L, d_model) -> (B, L, d_model)."""
        B, L, _ = u.shape
        cfg = self.config

        # Project input
        z = self.in_proj(u)  # (B, L, d_inner + 2*d_state*n_heads + n_heads)

        x, B_mat, C_mat, dt = z.split(
            [cfg.d_inner, cfg.d_state * cfg.n_heads, cfg.d_state * cfg.n_heads, cfg.n_heads],
            dim=-1,
        )
        # x: (B, L, d_inner); B_mat/C_mat: (B, L, d_state*n_heads); dt: (B, L, n_heads)

        # Causal depthwise conv on x
        # conv1d expects (B, C, L); pad left so no future context
        x_conv = x.transpose(1, 2)                       # (B, d_inner, L)
        x_conv = F.pad(x_conv, (self._conv_pad, 0))      # (B, d_inner, L + pad)
        x_conv = self.conv1d(x_conv)                     # (B, d_inner, L)
        x = F.silu(x_conv.transpose(1, 2))               # (B, L, d_inner)

        # Reshape for SSM: (B, L, n_heads, head_dim)
        x_ssm = x.view(B, L, cfg.n_heads, cfg.head_dim)

        # Reshape B, C: (B, L, n_heads, d_state)
        B_mat = B_mat.view(B, L, cfg.n_heads, cfg.d_state)
        C_mat = C_mat.view(B, L, cfg.n_heads, cfg.d_state)

        # Run SSM kernel
        y = self.ssm(x_ssm, dt, B_mat, C_mat)           # (B, L, n_heads, head_dim)
        y = y.reshape(B, L, cfg.d_inner)                 # (B, L, d_inner)

        # Output gate: sigmoid(W_gate(u)) * y
        gate = torch.sigmoid(self.W_gate(u))             # (B, L, d_inner)
        y = gate * y

        # RMSNorm
        y = self.norm(y)

        # Output projection
        return self.out_proj(y)                          # (B, L, d_model)


# ---------------------------------------------------------------------------
# Mamba2Model
# ---------------------------------------------------------------------------

class Mamba2Model(nn.Module):
    """Stack of Mamba-2 blocks forming a full sequence model."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        vocab_size: int,
        config: Optional[Mamba2Config] = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = Mamba2Config(d_model=d_model)
        self.config = config

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Mamba2Block(config) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """input_ids: (B, T) LongTensor -> (B, T, vocab_size) logits."""
        x = self.embedding(input_ids)        # (B, T, d_model)
        for block in self.blocks:
            x = x + block(x)                # residual connection
        x = self.norm(x)
        return self.unembed(x)               # (B, T, vocab_size)
