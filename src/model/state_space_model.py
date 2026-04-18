"""
State Space Model with Selective Scan (Mamba/S6-style).

Implements input-dependent (selective) state transitions following the S6 / Mamba
architecture.  Pure native PyTorch only.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# SSMKernel
# ---------------------------------------------------------------------------

class SSMKernel(nn.Module):
    """
    Selective State Space Model kernel (S6).

    Parameters
    ----------
    d_model : int
        Input / output feature dimension (= d_inner of the parent S6Block).
    d_state : int
        SSM state dimension N.
    dt_rank : int
        Rank of the delta-t projection.
    """

    def __init__(self, d_model: int, d_state: int, dt_rank: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Structured A matrix: shape [d_model, d_state], initialised as -I
        # (negative to ensure stable decay; we keep it as a raw parameter and
        #  exponentiate during discretisation so it always remains negative.)
        A_log = torch.zeros(d_model, d_state)
        # initialise so that exp(A_log) gives a near-identity *negative* matrix
        # We store log(-A) so A = -exp(A_log) remains negative.
        nn.init.constant_(A_log, math.log(1.0))  # A = -exp(0) = -1 diagonal-like
        self.A_log = nn.Parameter(A_log)  # [d_model, d_state]

        # Input-dependent projections (make the scan *selective*)
        # dt: compressed rank → full model dimension
        self.dt_proj = nn.Linear(dt_rank, d_model)
        # B and C are computed from the input token at each time step
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        # dt rank projection from input
        self.x_dt_proj = nn.Linear(d_model, dt_rank)

        # Skip-connection scalar per channel
        self.D = nn.Parameter(torch.ones(d_model))

        # dt bias + softplus for positivity
        nn.init.uniform_(self.dt_proj.bias, -4.0, -1.0)

    # ------------------------------------------------------------------
    def discretize(
        self,
        A: torch.Tensor,  # [d_model, d_state]  (already negative real)
        B: torch.Tensor,  # [B, d_model, d_state]
        dt: torch.Tensor,  # [B, d_model]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zero-Order Hold discretisation.

        A_bar = exp(dt * A)
        B_bar = dt * B   (simplified ZOH; avoids inverse for stability)

        Returns
        -------
        A_bar : [B, d_model, d_state]
        B_bar : [B, d_model, d_state]
        """
        # dt: [B, d_model] → [B, d_model, 1]
        dt_exp = dt.unsqueeze(-1)                     # [B, d_model, 1]
        # A:  [d_model, d_state] → [1, d_model, d_state]
        A_bc = A.unsqueeze(0)                         # [1, d_model, d_state]
        A_bar = torch.exp(dt_exp * A_bc)              # [B, d_model, d_state]
        B_bar = dt_exp * B                            # [B, d_model, d_state]
        return A_bar, B_bar

    # ------------------------------------------------------------------
    def selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        """
        Sequential selective scan.

        Parameters
        ----------
        u : [B, T, d_model]

        Returns
        -------
        y : [B, T, d_model]
        """
        B_sz, T, d_model = u.shape
        d_state = self.d_state

        # Recover negative A from log storage: A = -exp(A_log)
        A = -torch.exp(self.A_log)  # [d_model, d_state]

        # ---- input-dependent (selective) projections ----
        # dt: [B, T, d_model]
        dt_rank_hidden = self.x_dt_proj(u)            # [B, T, dt_rank]
        dt = F.softplus(self.dt_proj(dt_rank_hidden)) # [B, T, d_model]

        # B_t and C_t: [B, T, d_state]
        B_t = self.B_proj(u)   # [B, T, d_state]
        C_t = self.C_proj(u)   # [B, T, d_state]

        # ---- sequential scan ----
        # h: [B, d_model, d_state]
        h = torch.zeros(B_sz, d_model, d_state, device=u.device, dtype=u.dtype)

        ys = []
        for t in range(T):
            # Discretise at this time step
            # dt_t: [B, d_model]
            dt_t = dt[:, t, :]                        # [B, d_model]
            # B_bar_t: [B, d_model, d_state]
            B_t_step = B_t[:, t, :].unsqueeze(1).expand(B_sz, d_model, d_state)
            A_bar_t, B_bar_t = self.discretize(A, B_t_step, dt_t)

            # u_t: [B, d_model] → [B, d_model, 1] for broadcasting
            u_t = u[:, t, :].unsqueeze(-1)            # [B, d_model, 1]

            # State update: h = A_bar * h + B_bar * u_t
            h = A_bar_t * h + B_bar_t * u_t          # [B, d_model, d_state]

            # Output: y_t = sum_n C_t[n] * h[..., n]  + D * u_t
            # C_t_step: [B, d_state] → [B, 1, d_state]
            C_t_step = C_t[:, t, :].unsqueeze(1)     # [B, 1, d_state]
            # (h * C): [B, d_model, d_state] → sum over d_state → [B, d_model]
            y_t = (h * C_t_step).sum(dim=-1)         # [B, d_model]
            y_t = y_t + self.D * u[:, t, :]          # skip connection

            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # [B, T, d_model]
        return y


# ---------------------------------------------------------------------------
# S6Block
# ---------------------------------------------------------------------------

class S6Block(nn.Module):
    """
    Mamba-style S6 block with input projection, depthwise conv1d, SSM and
    gated output projection.
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
        self.expand = expand
        self.d_inner = d_model * expand

        # Split into x-path and z-path
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise conv along time
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        dt_rank = max(self.d_inner // 16, 1)
        self.ssm = SSMKernel(self.d_inner, d_state, dt_rank=dt_rank)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, T, d_model]

        Returns
        -------
        [B, T, d_model]
        """
        B_sz, T, _ = x.shape

        # Project to inner dimension, split into two paths
        xz = self.in_proj(x)                               # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)                   # each [B, T, d_inner]

        # Depthwise conv1d expects [B, C, T]
        x_conv = self.conv1d(x_proj.transpose(1, 2))      # [B, d_inner, T + pad]
        x_conv = x_conv[..., :T]                          # trim to T
        x_conv = x_conv.transpose(1, 2)                   # [B, T, d_inner]

        # SSM over SiLU-activated conv output
        x_ssm = self.ssm.selective_scan(F.silu(x_conv))  # [B, T, d_inner]

        # Gated output
        out = self.out_proj(x_ssm * F.silu(z))            # [B, T, d_model]
        return out


# ---------------------------------------------------------------------------
# MambaBlock
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Pre-norm wrapper around S6Block with residual connection."""

    def __init__(self, d_model: int, d_state: int = 16) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.s6 = S6Block(d_model, d_state=d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, T, d_model]

        Returns
        -------
        [B, T, d_model]
        """
        return x + self.s6(self.norm(x))


# ---------------------------------------------------------------------------
# MambaLanguageModel
# ---------------------------------------------------------------------------

class MambaLanguageModel(nn.Module):
    """Stack of MambaBlocks with token embedding and language-model head."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int = 4,
        d_state: int = 16,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)]
        )
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : [B, T]  (long)

        Returns
        -------
        logits : [B, T, vocab_size]
        """
        x = self.embedding(input_ids)          # [B, T, d_model]
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)                     # [B, T, d_model]
        logits = self.lm_head(x)               # [B, T, vocab_size]
        return logits

    def compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Next-token prediction cross-entropy loss.

        Parameters
        ----------
        input_ids : [B, T]

        Returns
        -------
        loss : scalar tensor
        """
        logits = self.forward(input_ids)       # [B, T, vocab_size]
        # Shift: predict token t+1 from context up to t
        shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, vocab]
        shift_labels = input_ids[:, 1:].contiguous()    # [B, T-1]
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss


# ---------------------------------------------------------------------------
# MambaConfig
# ---------------------------------------------------------------------------

@dataclass
class MambaConfig:
    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 4
    d_state: int = 8
    d_conv: int = 4
    expand: int = 2
