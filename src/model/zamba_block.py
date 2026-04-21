"""Zamba Block: Mamba SSM layers with periodic shared attention injection.

Reference: Zyphra 2025 — "Zamba: A Compact 7B SSM Hybrid Model"
https://arxiv.org/abs/2405.16712

Architecture overview:
    Zamba interleaves Mamba2-style SSM layers with shared attention layers at
    periodic intervals. The key innovation: a single shared attention layer is
    reused across multiple SSM layers via a "shared attention injection" — the
    attention KV is computed once and injected at every attn_every_n positions,
    reducing attention parameter overhead while maintaining global context.

Layer schedule (0-indexed):
    Layer i is an attention position if (i % attn_every_n == attn_every_n - 1),
    else it is an SSM layer. ALL attention positions share the SAME
    ZambaSharedAttention parameter set (weight sharing, not independent copies).

SSM recurrence (simplified Mamba-style):
    x_inner = input_proj(x)  # [B, T, d_inner]
    x_conv  = depthwise_conv1d(x_inner)
    h_t     = A * h_{t-1} + B_t * x_t   (A fixed at -0.5, B learned per step)
    y_t     = C_t * h_t                   (C learned per step)
    out     = output_proj(y_t)

Shared attention:
    Standard causal self-attention with learned Q/K/V projections shared
    across all attention positions in the stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ZambaConfig:
    """Hyperparameters for the Zamba hybrid SSM + shared attention model."""

    d_model: int = 2048
    d_state: int = 64          # SSM state dimension
    d_conv: int = 4            # SSM depthwise conv width
    expand: int = 2            # SSM inner expansion factor
    n_heads: int = 16          # shared attention heads
    head_dim: int = 128        # per-head dimension
    attn_every_n: int = 6      # inject shared attention every N SSM layers
    n_layers: int = 26         # total layers in the stack (SSM + attn)


# ---------------------------------------------------------------------------
# SSM Layer (simplified Mamba-style)
# ---------------------------------------------------------------------------

class ZambaSSMLayer(nn.Module):
    """Simplified Mamba-style SSM layer.

    Implements a linear state-space recurrence:
        h_t = A * h_{t-1} + B_t * x_t
        y_t = C_t * h_t

    A is fixed at -0.5 for stability; B and C are computed per-step from the
    input projection. A depthwise conv1d provides local context mixing before
    the recurrence.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        # Input projection: x → [z, x_ssm, B, C, dt]
        # z: gating,  x_ssm: SSM input,  B/C: state matrices,  dt: unused here
        self.in_proj = nn.Linear(d_model, self.d_inner + 2 * d_state, bias=False)

        # Depthwise conv for local mixing on x_ssm
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # Fixed decay (log-space), not learned
        self.register_buffer(
            "A_log",
            torch.full((self.d_inner,), math.log(0.5)),
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm for residual stability
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SSM layer with residual.

        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_model]
        """
        residual = x
        B, T, _ = x.shape

        # Project input
        projected = self.in_proj(x)                                 # [B, T, d_inner + 2*d_state]
        x_ssm = projected[..., : self.d_inner]                     # [B, T, d_inner]
        B_mat = projected[..., self.d_inner: self.d_inner + self.d_state]      # [B, T, d_state]
        C_mat = projected[..., self.d_inner + self.d_state:]       # [B, T, d_state]

        # Depthwise conv (needs [B, C, T] layout), then trim causal padding
        x_conv = self.conv1d(x_ssm.transpose(1, 2))                # [B, d_inner, T + pad]
        x_conv = x_conv[..., :T]                                    # [B, d_inner, T]
        x_conv = F.silu(x_conv).transpose(1, 2)                    # [B, T, d_inner]

        # Linear SSM recurrence: h_t = A*h_{t-1} + B_t*u_t, y_t = C_t*h_t
        A = -torch.exp(self.A_log)                                  # [d_inner], negative
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        ys: List[Tensor] = []
        for t in range(T):
            u_t = x_conv[:, t, :]                                   # [B, d_inner]
            b_t = B_mat[:, t, :]                                    # [B, d_state]
            c_t = C_mat[:, t, :]                                    # [B, d_state]

            # h: [B, d_inner, d_state]
            # A broadcast: [d_inner] → [1, d_inner, 1]
            # u_t * b_t: outer product [B, d_inner, d_state]
            h = A.view(1, -1, 1) * h + u_t.unsqueeze(-1) * b_t.unsqueeze(1)

            # y_t = sum over d_state: h * c_t  → [B, d_inner]
            y_t = (h * c_t.unsqueeze(1)).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                                  # [B, T, d_inner]

        # Output projection + residual
        out = self.out_proj(y)                                      # [B, T, d_model]
        return self.norm(out + residual)


# ---------------------------------------------------------------------------
# Shared Attention Layer
# ---------------------------------------------------------------------------

class ZambaSharedAttention(nn.Module):
    """Causal self-attention layer shared across multiple positions in the stack.

    A single instance of this module is reused (same weights) at every
    attention injection point. This reduces attention parameter count while
    preserving global context.
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_attn = n_heads * head_dim

        self.q_proj = nn.Linear(d_model, self.d_attn, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_attn, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_attn, bias=False)
        self.out_proj = nn.Linear(self.d_attn, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal self-attention with residual.

        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_model]
        """
        residual = x
        B, T, _ = x.shape
        scale = self.head_dim ** -0.5

        def _split_heads(t: Tensor) -> Tensor:
            # t: [B, T, d_attn] → [B, n_heads, T, head_dim]
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q = _split_heads(self.q_proj(x))   # [B, H, T, head_dim]
        K = _split_heads(self.k_proj(x))
        V = _split_heads(self.v_proj(x))

        # Scaled dot-product attention with causal mask
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, T, T]

        # Causal mask: upper triangle = -inf
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)                    # [B, H, T, head_dim]

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_attn)
        out = self.out_proj(attn_out)                               # [B, T, d_model]
        return self.norm(out + residual)


# ---------------------------------------------------------------------------
# Zamba Block
# ---------------------------------------------------------------------------

class ZambaBlock(nn.Module):
    """Stack of ZambaSSMLayers with periodic shared attention injection.

    The layer schedule is:
        For i in range(n_layers):
            if i % attn_every_n == attn_every_n - 1: → shared attention
            else:                                      → SSM layer

    All attention positions use the SAME ZambaSharedAttention module (shared
    parameters, not separate copies).
    """

    def __init__(self, config: ZambaConfig) -> None:
        super().__init__()
        self.config = config

        # Build the schedule
        self._schedule: List[str] = []
        for i in range(config.n_layers):
            if i % config.attn_every_n == config.attn_every_n - 1:
                self._schedule.append("attn")
            else:
                self._schedule.append("ssm")

        # Instantiate SSM layers (one per SSM position)
        n_ssm = self._schedule.count("ssm")
        self.ssm_layers = nn.ModuleList([
            ZambaSSMLayer(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
            )
            for _ in range(n_ssm)
        ])

        # ONE shared attention module (reused at every attn position)
        self.shared_attn = ZambaSharedAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the full Zamba layer stack.

        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_model]
        """
        ssm_idx = 0
        for kind in self._schedule:
            if kind == "ssm":
                x = self.ssm_layers[ssm_idx](x)
                ssm_idx += 1
            else:  # "attn"
                x = self.shared_attn(x)
        return x

    def n_attention_layers(self) -> int:
        """Number of positions where shared attention is injected."""
        return self._schedule.count("attn")

    def n_ssm_layers(self) -> int:
        """Number of SSM layers in the stack."""
        return self._schedule.count("ssm")

    def parameter_sharing_ratio(self) -> float:
        """Ratio of shared attention parameters to total block parameters.

        Demonstrates that one attention module covers multiple injection
        points, reducing per-layer attention overhead.
        """
        attn_params = sum(p.numel() for p in self.shared_attn.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        return attn_params / total_params if total_params > 0 else 0.0
