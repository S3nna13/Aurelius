"""Jamba: Hybrid Transformer-Mamba Language Model.

Reference: Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model",
arXiv:2403.19887 (AI21 Labs, 2024).

Architecture:
  - Interleaves Transformer attention layers with Mamba SSM layers.
  - Default ratio: 1 attention layer per attn_every_k=4 layers.
    Layers at indices where (layer_idx - attn_layer_offset) % attn_every_k == 0
    are JambaAttentionBlock; all others are JambaMambaBlock.
  - FFN is standard SwiGLU (no MoE — MoE lives separately in src/model/moe.py).
  - SSM block is Mamba-1 style, inlined (NOT imported from mamba.py).

Variable notation follows the paper notation where applicable:
  - d_model: model (residual) dimension (D in paper)
  - d_state:  SSM state dimension (N in paper)
  - d_conv:   depthwise conv kernel width
  - Δ (delta): input-dependent timescale (softplus-activated)
  - A:        state-transition matrix (diagonal, log-parameterised)
  - B, C:     input/output projection matrices (input-dependent)
  - D:        skip-connection scalar per channel
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# RMSNorm (inline — avoids cross-module coupling in this hybrid file)
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps).to(x.dtype)
        return x_normed * self.weight.to(x.dtype)


# ---------------------------------------------------------------------------
# Inline Mamba-1 SSM (NOT imported from mamba.py — as required)
# ---------------------------------------------------------------------------


class _JambaSSMCore(nn.Module):
    """Mamba-1 style selective SSM core for Jamba hybrid blocks.

    Given x of shape (B, T, d_model), maintains SSM state h of shape
    (B, d_model, d_state).

    Equations (paper notation):
        [Δ_raw, B_in, C_in] = x_proj(x)
        Δ = softplus(Δ_raw)                              [B, T, d_model]
        A = -exp(A_log)                                  [d_model, d_state]
        h_t = diag(exp(Δ_t · A)) h_{t-1} + Δ_t ⊗ B_t · x_t
        y_t = C_t · h_t + D · x_t

    h is carried across calls for stateful generation.
    """

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state  # N in paper

        # Combined projection x → (Δ_raw, B_in, C_in)
        # Δ_raw: (d_model,), B_in: (d_state,), C_in: (d_state,)
        self.x_proj = nn.Linear(d_model, d_model + 2 * d_state, bias=False)

        # A: diagonal state-transition matrix, log-parameterised
        # A_log[i, n] = log(n+1)  →  A = -exp(A_log)  (always negative → stable)
        A_log = (
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(d_model, -1)
        )  # (d_model, d_state)
        self.A_log = nn.Parameter(A_log.clone())

        # D: per-channel skip scalar (paper: D·x_t skip connection)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(
        self,
        x: Tensor,
        h: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, d_model)
            h: optional prior SSM state (B, d_model, d_state); zeros if None.

        Returns:
            y: (B, T, d_model)
            h_new: (B, d_model, d_state)  — final hidden state after T steps
        """
        B, T, _ = x.shape

        if h is None:
            h = torch.zeros(B, self.d_model, self.d_state, dtype=x.dtype, device=x.device)

        # Project input → Δ_raw, B_in, C_in
        proj = self.x_proj(x)  # (B, T, d_model + 2*d_state)
        delta_raw, B_in, C_in = proj.split([self.d_model, self.d_state, self.d_state], dim=-1)
        # delta_raw: (B, T, d_model)
        # B_in, C_in: (B, T, d_state)

        # Δ = softplus(Δ_raw)
        delta = F.softplus(delta_raw)  # (B, T, d_model)

        # A = -exp(A_log)  →  shape (d_model, d_state)
        A = -torch.exp(self.A_log.float())

        # Discretise: A_bar_t = exp(Δ_t * A)  shape (B, T, d_model, d_state)
        # delta: (B, T, d_model, 1),  A: (1, 1, d_model, d_state)
        A_bar = torch.exp(delta.unsqueeze(-1) * A[None, None])  # (B, T, d_model, d_state)

        # B_bar_t = Δ_t ⊗ B_in_t  shape (B, T, d_model, d_state)
        # delta: (B, T, d_model, 1),  B_in: (B, T, 1, d_state)
        B_bar = delta.unsqueeze(-1) * B_in.unsqueeze(2)  # (B, T, d_model, d_state)

        # Sequential scan over T timesteps
        ys = []
        h = h.to(x.dtype)
        for t in range(T):
            # h: (B, d_model, d_state)
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            # C_in[:, t]: (B, d_state) → (B, 1, d_state)
            y_t = (h * C_in[:, t].unsqueeze(1)).sum(dim=-1) + self.D * x[:, t]
            ys.append(y_t)  # (B, d_model)

        y = torch.stack(ys, dim=1)  # (B, T, d_model)
        return y, h.detach().clone()


# ---------------------------------------------------------------------------
# JambaMambaBlock
# ---------------------------------------------------------------------------


class JambaMambaBlock(nn.Module):
    """Jamba Mamba block: pre-norm SSM with SiLU gate + residual.

    Architecture (paper §3.1 Mamba layers):
        h_in  = RMSNorm(x)
        x_ssm, z = split(in_proj(h_in), 2)      [gating branch]
        y_ssm = SSM(x_ssm)
        out   = out_proj(y_ssm * SiLU(z))
        return x + out
    """

    layer_type: str = "mamba"

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.norm = _RMSNorm(d_model)

        # Gated input projection: d_model → 2 * d_model  (x branch + z gate)
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)

        # Causal depthwise conv1d applied to x branch
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            groups=d_model,
            bias=True,
            padding=d_conv - 1,
        )

        self.ssm = _JambaSSMCore(d_model, d_state)

        # Output projection: d_model → d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        h: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, d_model)
            h: optional SSM hidden state (B, d_model, d_state); None → zeros

        Returns:
            out: (B, T, d_model)
            h_new: (B, d_model, d_state)
        """
        B, T, _ = x.shape
        residual = x

        x_normed = self.norm(x)  # (B, T, d_model)

        # Gate split
        xz = self.in_proj(x_normed)  # (B, T, 2*d_model)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, T, d_model)

        # Causal depthwise conv1d
        x_conv = x_branch.transpose(1, 2)  # (B, d_model, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]  # (B, d_model, T)
        x_conv = F.silu(x_conv.transpose(1, 2))  # (B, T, d_model)

        # Selective SSM
        y, h_new = self.ssm(x_conv, h)  # (B, T, d_model)

        # Multiplicative gate
        y = y * F.silu(z)  # (B, T, d_model)

        out = self.out_proj(y)  # (B, T, d_model)
        return residual + out, h_new


# ---------------------------------------------------------------------------
# JambaAttentionBlock
# ---------------------------------------------------------------------------


class _MultiHeadAttention(nn.Module):
    """Grouped-Query Attention (GQA) for Jamba attention blocks.

    Supports n_kv_heads <= n_heads (n_kv_heads == n_heads → standard MHA).
    Uses causal mask.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        # each KV head is shared across (n_heads // n_kv_heads) Q heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads to match Q heads (GQA repeat)
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim)
            k = k.reshape(B, self.n_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim)
            v = v.reshape(B, self.n_heads, T, self.head_dim)

        # Scaled dot-product attention with causal mask
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, -1)  # (B, T, d_model)
        return self.o_proj(out)


class _SwiGLUFFN(nn.Module):
    """Simple SwiGLU FFN (no external deps)."""

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = int(d_model * 8 / 3)
            # round to nearest multiple of 64
            d_ff = (d_ff + 63) // 64 * 64
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class JambaAttentionBlock(nn.Module):
    """Jamba Attention block: pre-norm MHA + pre-norm FFN with residuals.

    Architecture (paper §3.1 Attention layers):
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    layer_type: str = "attention"

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int | None = None,
    ) -> None:
        super().__init__()
        self.attn_norm = _RMSNorm(d_model)
        self.ffn_norm = _RMSNorm(d_model)
        self.attn = _MultiHeadAttention(d_model, n_heads, n_kv_heads)
        self.ffn = _SwiGLUFFN(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# JambaModel
# ---------------------------------------------------------------------------


@dataclass
class JambaConfig:
    """Hyperparameters for JambaModel.

    Attributes:
        d_model:           residual stream / embedding dimension
        n_layers:          total number of layers (attention + mamba)
        n_heads:           number of query heads in attention blocks
        n_kv_heads:        number of key/value heads (GQA; must divide n_heads)
        d_state:           SSM state dimension N (Mamba blocks only)
        d_conv:            depthwise conv kernel width (Mamba blocks only)
        d_ff:              FFN hidden dim; None → auto (8/3 * d_model rounded to 64)
        attn_layer_offset: first layer index that is an attention block
        attn_every_k:      place an attention block every k layers
        rms_norm_eps:      epsilon for RMSNorm
    """

    d_model: int = 2048
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    d_state: int = 16
    d_conv: int = 4
    d_ff: int | None = None
    attn_layer_offset: int = 0
    attn_every_k: int = 4
    rms_norm_eps: float = 1e-6


def _is_attn_layer(layer_idx: int, attn_layer_offset: int, attn_every_k: int) -> bool:
    """Return True if layer_idx should be an attention block."""
    return (layer_idx - attn_layer_offset) % attn_every_k == 0


class JambaModel(nn.Module):
    """Jamba hybrid Transformer-Mamba model.

    Interleaves JambaAttentionBlock and JambaMambaBlock according to
    attn_layer_offset and attn_every_k.

    forward(x, hidden_states=None) → (output, new_hidden_states)
      - x:             (B, T, d_model) — already embedded token representations
      - hidden_states: list of per-Mamba-layer SSM states, each (B, d_model, d_state)
                       or None to zero-initialise
      - output:        (B, T, d_model)
      - new_hidden_states: updated list of SSM states (same order as Mamba layers)
    """

    def __init__(self, cfg: JambaConfig) -> None:
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        for i in range(cfg.n_layers):
            if _is_attn_layer(i, cfg.attn_layer_offset, cfg.attn_every_k):
                layers.append(
                    JambaAttentionBlock(
                        d_model=cfg.d_model,
                        n_heads=cfg.n_heads,
                        n_kv_heads=cfg.n_kv_heads,
                        d_ff=cfg.d_ff,
                    )
                )
            else:
                layers.append(
                    JambaMambaBlock(
                        d_model=cfg.d_model,
                        d_state=cfg.d_state,
                        d_conv=cfg.d_conv,
                    )
                )
        self.layers = nn.ModuleList(layers)
        self.norm = _RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        hidden_states: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """
        Args:
            x:             (B, T, d_model)
            hidden_states: list of Mamba SSM states (one per JambaMambaBlock),
                           each (B, d_model, d_state).  None → zero-init.

        Returns:
            output:            (B, T, d_model)
            new_hidden_states: list[Tensor], each (B, d_model, d_state)
        """
        # Count Mamba layers to validate / initialise hidden_states
        mamba_layers = [line for line in self.layers if isinstance(line, JambaMambaBlock)]
        n_mamba = len(mamba_layers)

        if hidden_states is None:
            B = x.shape[0]
            hidden_states = [
                torch.zeros(B, self.cfg.d_model, self.cfg.d_state, dtype=x.dtype, device=x.device)
                for _ in range(n_mamba)
            ]
        else:
            if len(hidden_states) != n_mamba:
                raise ValueError(
                    f"hidden_states has {len(hidden_states)} entries but model has {n_mamba} Mamba layers"  # noqa: E501
                )

        new_hidden_states: list[Tensor] = []
        mamba_idx = 0

        for layer in self.layers:
            if isinstance(layer, JambaAttentionBlock):
                x = layer(x)
            else:
                h_in = hidden_states[mamba_idx]
                x, h_out = layer(x, h_in)
                new_hidden_states.append(h_out)
                mamba_idx += 1

        x = self.norm(x)
        return x, new_hidden_states


# ---------------------------------------------------------------------------
# Tiny config helper (for tests and quick experiments)
# ---------------------------------------------------------------------------


def jamba_tiny_config() -> JambaConfig:
    """Return tiny Jamba config suitable for unit tests."""
    return JambaConfig(
        d_model=64,
        n_layers=8,
        n_heads=4,
        n_kv_heads=2,
        d_state=16,
        d_conv=4,
        attn_layer_offset=0,
        attn_every_k=4,
    )
