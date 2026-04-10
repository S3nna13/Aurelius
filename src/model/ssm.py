"""Mamba-style Selective State Space Model (S6) layer.

Reference: Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces".
https://arxiv.org/abs/2312.00752

Pure PyTorch implementation — no custom CUDA kernels required.

Key innovation: unlike linear SSMs where A, B, C are fixed, here A, B, C, and Δ (delta)
are INPUT-DEPENDENT, making the model "selective" — it can choose what to remember/forget.

State equations (ZOH discretization):
    A_bar_t = exp(-exp(A) * delta_t)        [always stable since A_log stores log(-A)]
    B_bar_t = delta_t * B_t                  [simplified Euler for B]
    h_t = A_bar_t * h_{t-1} + B_bar_t * u_t [state update]
    y_t = C_t @ h_t + D * u_t               [output with skip connection]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SSMConfig:
    """Hyperparameters for the Selective State Space Model."""

    d_model: int = 128
    d_state: int = 16           # SSM state dimension (N in the paper)
    d_conv: int = 4             # local depthwise convolution width
    expand: int = 2             # inner dim expansion factor (d_inner = expand * d_model)
    dt_rank: str | int = 8      # rank of Δt projection ("auto" → ceil(d_model / 16))
    dropout: float = 0.0
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    bias: bool = False
    conv_bias: bool = True

    def __post_init__(self) -> None:
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


# ---------------------------------------------------------------------------
# Naive (reference) selective scan — no CUDA optimisation
# ---------------------------------------------------------------------------

def selective_scan_naive(
    u: Tensor,      # (B, L, d_inner)   — input sequence
    delta: Tensor,  # (B, L, d_inner) or (B, L, dt_rank) — Δ (softplus applied internally)
    A: Tensor,      # (d_inner, d_state) — log of negative A (A_log)
    B: Tensor,      # (B, L, d_state)   — input-dependent B
    C: Tensor,      # (B, L, d_state)   — input-dependent C
    D: Tensor,      # (d_inner,)        — skip connection weight
) -> Tensor:
    """Naive sequential selective scan.

    For each time step t:
        dt_t     = softplus(delta[:, t, :])          [B, d_inner] (expanded if needed)
        A_bar_t  = exp(-exp(A) * dt_t_expanded)      [B, d_inner, d_state]
        B_bar_t  = dt_t_expanded * B_t               [B, d_inner, d_state]
        h_t      = A_bar_t * h_{t-1} + B_bar_t * u_t [state update]
        y_t      = sum(C_t * h_t, dim=-1) + D * u_t  [B, d_inner]

    Args:
        u:     (B, L, d_inner)  input sequence
        delta: (B, L, d_inner) or (B, L, dt_rank) — if dt_rank < d_inner, tiled to d_inner
        A:     (d_inner, d_state) — A_log (stores log of positive values; sign handled inside)
        B:     (B, L, d_state) — input-dependent B projection
        C:     (B, L, d_state) — input-dependent C projection
        D:     (d_inner,) — skip connection

    Returns:
        (B, L, d_inner)
    """
    B_batch, L, d_inner = u.shape
    assert A.dim() == 2, f"A must be 2D (d_inner, d_state), got {A.shape}"
    d_state = A.shape[1]

    # Expand delta from (B, L, dt_rank) → (B, L, d_inner) if needed by tiling
    dt_rank_in = delta.shape[-1]
    if dt_rank_in != d_inner:
        # Tile the last dimension to reach d_inner
        repeat_factor = math.ceil(d_inner / dt_rank_in)
        delta = delta.repeat(1, 1, repeat_factor)[:, :, :d_inner]  # (B, L, d_inner)

    # softplus to ensure positivity of Δ
    dt = F.softplus(delta)  # (B, L, d_inner)

    # Precompute exp(A) — A stores log(-A_original), so exp(A) = -A_original > 0
    # A_discrete = exp(-exp(A) * dt)
    neg_A = torch.exp(A)  # (d_inner, d_state)

    # Hidden state h: (B, d_inner, d_state)
    h = torch.zeros(B_batch, d_inner, d_state, dtype=u.dtype, device=u.device)
    ys = []

    for t in range(L):
        u_t = u[:, t, :]          # (B, d_inner)
        dt_t = dt[:, t, :]        # (B, d_inner)
        B_t = B[:, t, :]          # (B, d_state)
        C_t = C[:, t, :]          # (B, d_state)

        # A_bar: (B, d_inner, d_state)
        # dt_t: (B, d_inner) → (B, d_inner, 1)
        # neg_A: (d_inner, d_state) → (1, d_inner, d_state)
        A_bar = torch.exp(
            -dt_t.unsqueeze(-1) * neg_A.unsqueeze(0)
        )  # (B, d_inner, d_state)

        # B_bar: (B, d_inner, d_state) = dt_t[:, :, None] * B_t[:, None, :]
        B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, d_inner, d_state)

        # State update: h = A_bar * h + B_bar * u_t
        # u_t: (B, d_inner) → (B, d_inner, 1)
        h = A_bar * h + B_bar * u_t.unsqueeze(-1)  # (B, d_inner, d_state)

        # Output: y_t = sum_over_state(C_t * h_t) + D * u_t
        # C_t: (B, d_state) → (B, 1, d_state)
        y_t = (C_t.unsqueeze(1) * h).sum(dim=-1) + D * u_t  # (B, d_inner)
        ys.append(y_t)

    return torch.stack(ys, dim=1)  # (B, L, d_inner)


# ---------------------------------------------------------------------------
# SelectiveSSM (S6 core)
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """S6: Selective State Space Model core.

    The key innovation: A, B, C, Δ are INPUT-DEPENDENT (selective).

    State equation: h_t = A_t * h_{t-1} + B_t * x_t
    Output:         y_t = C_t * h_t + D * x_t

    Where:
        A_t = exp(-exp(Δ_t) * A)   [discretized, always stable]
        B_t = Δ_t * B_t            [simplified Euler discretization]
        Δ_t, B_t, C_t all from linear projections of x_t
    """

    def __init__(self, cfg: SSMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_inner = cfg.d_model * cfg.expand
        self.d_inner = d_inner

        # Resolve dt_rank (SSMConfig.__post_init__ handles "auto" → int)
        self.dt_rank: int = cfg.dt_rank  # type: ignore[assignment]
        if isinstance(self.dt_rank, str):
            # fallback if SSMConfig didn't run __post_init__ (shouldn't happen)
            self.dt_rank = math.ceil(cfg.d_model / 16)

        # Input-dependent projections: x → (Δ, B, C) combined
        # Output size: dt_rank + 2*d_state
        self.x_proj = nn.Linear(
            d_inner,
            self.dt_rank + 2 * cfg.d_state,
            bias=False,
        )

        # Δ projection: dt_rank → d_inner (expands rank-deficient Δ to full width)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # A_log: log of the state matrix (initialised as log(1..N) for each d_inner dim)
        # Shape (d_inner, d_state) — stores log(-A) so A is always negative → stable
        A = torch.arange(1, cfg.d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # D: skip connection (one scalar per d_inner channel)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        # Initialise dt_proj using log-uniform distribution over [dt_min, dt_max]
        self._init_dt_proj()

    def _init_dt_proj(self) -> None:
        """Initialise dt_proj.bias so that softplus(bias) ≈ dt_init."""
        dt_min, dt_max = self.cfg.dt_min, self.cfg.dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=self.cfg.dt_init_floor)
        # inverse softplus: x = log(exp(dt) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_weight_decay = True  # type: ignore[attr-defined]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, d_inner)

        Returns:
            (B, L, d_inner)
        """
        B, L, d_inner = x.shape

        # Project x to get (Δ, B_ssm, C_ssm)
        xz = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta_raw, B_ssm, C_ssm = xz.split(
            [self.dt_rank, self.cfg.d_state, self.cfg.d_state], dim=-1
        )
        # delta_raw: (B, L, dt_rank)
        # B_ssm, C_ssm: (B, L, d_state)

        # Expand Δ from dt_rank → d_inner
        delta = self.dt_proj(delta_raw)  # (B, L, d_inner) — bias added here

        # A: retrieve the negative state matrix (always positive after exp)
        # A_log stores log(-A_original), so -exp(A_log) = A_original < 0 → stable
        # We pass A_log directly; selective_scan_naive will compute exp(A_log)
        A = self.A_log  # (d_inner, d_state)
        D = self.D       # (d_inner,)

        return selective_scan_naive(x, delta, A, B_ssm, C_ssm, D)


# ---------------------------------------------------------------------------
# MambaBlock — full block with in_proj, conv1d, SSM, out_proj
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Full Mamba block: input projection + conv1d + SSM + output projection.

    Architecture:
        x → in_proj → [x', z]  (split along last dim)
            x' → causal conv1d → silu → SelectiveSSM → * silu(z) → out_proj → y

    This is a drop-in replacement for SwiGLUFFN.
    Returns: (B, L, d_model) — same signature (no aux_loss).
    """

    def __init__(self, config) -> None:
        """
        Args:
            config: AureliusConfig (uses config.d_model; SSM hyperparams use defaults).
        """
        super().__init__()

        # Build internal SSMConfig from AureliusConfig
        self.ssm_cfg = SSMConfig(d_model=config.d_model)
        d_model = config.d_model
        d_inner = self.ssm_cfg.d_model * self.ssm_cfg.expand
        self.d_inner = d_inner
        d_conv = self.ssm_cfg.d_conv

        # in_proj: d_model → 2 * d_inner (produces x' and z gate)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=self.ssm_cfg.bias)

        # Causal depthwise conv1d over x' — groups=d_inner makes it depthwise
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            bias=self.ssm_cfg.conv_bias,
            padding=d_conv - 1,  # left-pad so output length == input length after trim
        )

        # Core SSM
        self.ssm = SelectiveSSM(self.ssm_cfg)

        # out_proj: d_inner → d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=self.ssm_cfg.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, L, d_model)

        Returns:
            (B, L, d_model)
        """
        B, L, _ = x.shape

        # Project and split into x' (ssm branch) and z (gate)
        xz = self.in_proj(x)          # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal depthwise conv1d on x_ssm
        # Conv1d expects (B, C, L); we use left-padding=d_conv-1 then trim right
        x_conv = x_ssm.transpose(1, 2)                  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)                     # (B, d_inner, L + d_conv - 1)
        x_conv = x_conv[:, :, :L]                        # (B, d_inner, L) — trim right
        x_conv = x_conv.transpose(1, 2)                  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # Selective SSM
        y = self.ssm(x_conv)    # (B, L, d_inner)

        # Gate with z branch
        y = y * F.silu(z)       # (B, L, d_inner)

        # Project back to d_model
        return self.out_proj(y)  # (B, L, d_model)


# ---------------------------------------------------------------------------
# selective_scan — public alias matching the spec API
# ---------------------------------------------------------------------------

def selective_scan(
    u: Tensor,      # (B, L, d_inner)
    dt: Tensor,     # (B, L, d_inner) — time step (softplus applied internally)
    A: Tensor,      # (d_inner, d_state) — log scale, negative
    B: Tensor,      # (B, L, d_state)
    C: Tensor,      # (B, L, d_state)
    D: Tensor,      # (d_inner,)
) -> Tensor:
    """Discretized SSM scan matching the spec API.

    Delegates to selective_scan_naive; dt is treated as already in d_inner width.
    Returns: (B, L, d_inner)
    """
    return selective_scan_naive(u, dt, A, B, C, D)


# ---------------------------------------------------------------------------
# MambaLayer — MambaBlock with residual connection + RMSNorm
# ---------------------------------------------------------------------------

class MambaLayer(nn.Module):
    """MambaBlock wrapped with pre-norm (RMSNorm) and residual connection."""

    def __init__(self, config: SSMConfig) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(config.d_model)

        # Build a lightweight AureliusConfig-like namespace for MambaBlock
        class _Cfg:
            d_model = config.d_model

        self.block = _MambaBlockFromSSMConfig(config)

    def forward(self, x: Tensor) -> Tensor:
        """Input (B, T, d_model), output (B, T, d_model) with residual."""
        return x + self.block(self.norm(x))


class _MambaBlockFromSSMConfig(nn.Module):
    """Internal MambaBlock that accepts SSMConfig directly (not AureliusConfig)."""

    def __init__(self, config: SSMConfig) -> None:
        super().__init__()
        d_model = config.d_model
        d_inner = config.d_model * config.expand
        self.d_inner = d_inner
        d_conv = config.d_conv

        # Resolve dt_rank
        dt_rank = config.dt_rank
        if isinstance(dt_rank, str) and dt_rank == "auto":
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            bias=True,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        A = torch.arange(1, config.d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self._d_state = config.d_state

    def forward(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        xz2 = self.x_proj(x_conv)
        delta_raw, B_ssm, C_ssm = xz2.split([self.dt_rank, self._d_state, self._d_state], dim=-1)
        delta = self.dt_proj(delta_raw)
        A = -torch.exp(self.A_log)

        y = selective_scan_naive(x_conv, delta, A, B_ssm, C_ssm, self.D)
        y = y * F.silu(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# MambaLM — stack of MambaLayers for language modelling
# ---------------------------------------------------------------------------

class MambaLM(nn.Module):
    """Stack of Mamba layers for language modeling.

    Returns logits (B, T, vocab_size). Self-contained — does NOT wrap
    AureliusTransformer.
    """

    def __init__(self, config: SSMConfig, n_layers: int, vocab_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.layers = nn.ModuleList([MambaLayer(config) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: (B, T) integer token ids

        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)
