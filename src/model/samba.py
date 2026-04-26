"""SAMBA: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.

Reference: Ren et al., 2024, arXiv:2406.07522.
https://arxiv.org/abs/2406.07522

SAMBA interleaves Mamba SSM layers with sliding-window attention (SWA) layers,
combining the strengths of:
  - Mamba SSM: efficient unlimited-context recurrence (O(1) per step memory).
  - SWA: precise local recall via bounded attention windows.

Interleaving pattern: [Mamba, Mamba, SWA] repeating (every swa_every-th layer is SWA).
Each block is wrapped with RMSNorm pre-norm and a residual connection.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rms_norm import RMSNorm

# ---------------------------------------------------------------------------
# SambaMambaBlock — inline Mamba-1 SSM block (no import from mamba.py)
# ---------------------------------------------------------------------------


class SambaMambaBlock(nn.Module):
    """Simplified inline Mamba-1 SSM block for use inside SAMBA.

    Architecture:
        x → in_proj → [x_branch, z]   (d_model → 2*d_inner, split)
        x_branch → causal depthwise conv1d → SiLU
        x_branch → SSM (selective scan with A=-exp, B/C projections, D skip)
        y = ssm_out * SiLU(z)          (multiplicative gating)
        out = out_proj(y)              (d_inner → d_model)

    This is a self-contained implementation — it does NOT import from mamba.py.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model

        d_inner = self.d_inner
        dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        # in_proj: x → [x_branch, z], both of size d_inner
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise conv1d over the x branch
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            bias=True,
            padding=d_conv - 1,  # left-pad; we trim right to restore length
        )

        # Combined x → (delta_raw, B_in, C_in) projection
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

        # delta: dt_rank → d_inner
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # A: log-parameterized state matrix (d_inner, d_state)
        # A_log stores log(|A|); actual A = -exp(A_log) < 0 for stability
        A_log = torch.log(torch.arange(1, d_state + 1).float()).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(A_log.clone())

        # D: skip connection scalar per inner channel
        self.D = nn.Parameter(torch.ones(d_inner))

        # out_proj: d_inner → d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape
        d_inner = self.d_inner

        # Project and split
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Causal depthwise conv1d
        xc = x_branch.transpose(1, 2)  # (B, d_inner, T)
        xc = self.conv1d(xc)  # (B, d_inner, T + d_conv - 1)
        xc = xc[:, :, :T]  # (B, d_inner, T) — trim right
        xc = xc.transpose(1, 2)  # (B, T, d_inner)
        xc = F.silu(xc)

        # Compute input-dependent SSM params
        proj_out = self.x_proj(xc)  # (B, T, dt_rank + 2*d_state)
        delta_raw, B_in, C_in = proj_out.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        # delta_raw: (B, T, dt_rank); B_in, C_in: (B, T, d_state)

        delta = F.softplus(self.dt_proj(delta_raw))  # (B, T, d_inner)

        # Discretize A
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # A_bar: (B, T, d_inner, d_state)
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # B_bar: (B, T, d_inner, d_state)
        B_bar = delta.unsqueeze(-1) * B_in.unsqueeze(2)

        # Sequential SSM scan
        h = torch.zeros(B, d_inner, self.d_state, dtype=x.dtype, device=x.device)
        ys = []
        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * xc[:, t].unsqueeze(-1)
            y_t = (h * C_in[:, t].unsqueeze(1)).sum(dim=-1) + self.D * xc[:, t]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, T, d_inner)

        # Multiplicative gate
        y = y * F.silu(z)  # (B, T, d_inner)

        return self.out_proj(y)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# SambaSWABlock — Sliding-Window Self-Attention block
# ---------------------------------------------------------------------------


class SambaSWABlock(nn.Module):
    """Sliding-window causal self-attention block.

    Each token attends only to the previous `window_size` tokens (including
    itself), enforcing a causal mask within the local window.  Tokens outside
    the window are masked to -inf so they contribute nothing to the softmax.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 64,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"  # noqa: S101
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape
        H = self.n_heads
        D = self.head_dim
        W = self.window_size

        # QKV projection
        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, d_model)

        # Reshape to (B, H, T, D)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Scaled dot-product scores: (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Build combined causal + sliding-window mask.
        # mask[i, j] == True means position j is BLOCKED for query i.
        # Blocked if: j > i (future, causal) OR i - j >= W (outside window).
        device = x.device
        i_idx = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
        j_idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        # Causal: j > i
        causal_mask = j_idx > i_idx  # (T, T)
        # Window: i - j >= W  (j is too far in the past)
        window_mask = (i_idx - j_idx) >= W  # (T, T)
        combined_mask = causal_mask | window_mask  # (T, T)

        # Apply mask: blocked positions → -inf
        scores = scores.masked_fill(combined_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax + weighted sum
        attn = F.softmax(scores, dim=-1)  # (B, H, T, T)
        # Replace NaN from all-masked rows (e.g., when T < W at position 0 is fine,
        # but guard against the degenerate case where a full row is -inf)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)

        return self.out_proj(out)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# SambaModel — interleaved Mamba + SWA stack
# ---------------------------------------------------------------------------


class _SambaLayer(nn.Module):
    """Pre-norm residual wrapper: out = x + block(RMSNorm(x))."""

    def __init__(self, block: nn.Module, d_model: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(self.norm(x))


class SambaModel(nn.Module):
    """SAMBA interleaved Mamba + SWA model.

    Layer pattern: [Mamba, Mamba, ..., SWA] repeating.
    SWA is placed at every layer_idx where ``layer_idx % swa_every == swa_every - 1``.
    All other layers are Mamba blocks.

    forward(input_ids) returns the hidden states (B, T, d_model), NOT logits.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_state: int = 16,
        window_size: int = 64,
        swa_every: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)

        layers: list[nn.Module] = []
        for idx in range(n_layers):
            if idx % swa_every == (swa_every - 1):
                block = SambaSWABlock(d_model=d_model, n_heads=n_heads, window_size=window_size)
            else:
                block = SambaMambaBlock(d_model=d_model, d_state=d_state)
            layers.append(_SambaLayer(block=block, d_model=d_model))

        self.layers = nn.ModuleList(layers)
        self.norm_out = RMSNorm(d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: (B, T) integer token ids

        Returns:
            (B, T, d_model) hidden states after final RMSNorm
        """
        x = self.embedding(input_ids)  # (B, T, d_model)

        for layer in self.layers:
            x = layer(x)

        return self.norm_out(x)  # (B, T, d_model)
