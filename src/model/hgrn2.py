"""HGRN2: Gated Linear RNNs with State Expansion.

Reference: Qin et al., 2024 -- "HGRN2: Gated Linear RNNs with State Expansion".
https://arxiv.org/abs/2404.07904

Overview:
    HGRN2 improves on HGRN1 by adding a *state expansion* trick: each of the
    d_model channels is expanded to E copies in the hidden state, giving the
    recurrence a richer memory capacity while keeping the output size fixed.

Core recurrence equations (per timestep t):
    f_t = lower_bound + (1 - lower_bound) * sigmoid(W_f @ x_t)   # forget gate ∈ [lb, 1]
    u_t = silu(W_u @ x_t)                                         # input value  (d_model)
    forget_exp = f_t.repeat_interleave(E)                         # (d_model * E)
    i_t = 1 - forget_exp                                          # input gate (HGRN constraint)
    u_exp = u_t.repeat_interleave(E)                              # (d_model * E)
    h_t = forget_exp * h_{t-1} + i_t * u_exp                     # state update (d_model*E)
    y_t = h_t.reshape(d_model, E).sum(-1) / sqrt(E)              # contract    (d_model)
    o_t = sigmoid(W_o @ x_t)                                      # output gate
    out_t = o_t * y_t                                             # gated output (d_model)

Classes:
    HGRN2Cell   -- single-step recurrence, method: step(x, h) -> (out, h_new)
    HGRN2Layer  -- full-sequence processing by unrolling HGRN2Cell, forward(x) -> (B,T,d)
    HGRN2Block  -- HGRN2Layer + SwiGLU FFN + RMSNorm (pre-norm), forward(x) -> (B,T,d)
    HGRN2Model  -- embedding + stack of HGRN2Blocks + final RMSNorm, forward(ids) -> (B,T,d)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rms_norm import RMSNorm

# ---------------------------------------------------------------------------
# HGRN2Cell: single-step recurrence
# ---------------------------------------------------------------------------


class HGRN2Cell(nn.Module):
    """Single-step HGRN2 recurrence.

    Processes one timestep at a time; the hidden state h has size d_model * expand.

    Args:
        d_model:     Input/output channel dimension.
        expand:      State expansion factor E.  Hidden state dim = d_model * expand.
        lower_bound: Minimum value of the forget gate (prevents vanishing gradients).
    """

    def __init__(
        self,
        d_model: int,
        expand: int = 8,
        lower_bound: float = 1.0 / 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.lower_bound = lower_bound

        # Forget gate projection
        self.W_f = nn.Linear(d_model, d_model, bias=True)
        # Input value projection
        self.W_u = nn.Linear(d_model, d_model, bias=True)
        # Output gate projection
        self.W_o = nn.Linear(d_model, d_model, bias=True)

    def step(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """Run one step of HGRN2.

        Args:
            x: (B, d_model) input at the current timestep.
            h: (B, d_model * expand) previous hidden state.

        Returns:
            out:   (B, d_model) gated output.
            h_new: (B, d_model * expand) updated hidden state.
        """
        B = x.shape[0]
        E = self.expand
        lb = self.lower_bound

        # Forget gate: clipped to [lower_bound, 1]
        f = lb + (1.0 - lb) * torch.sigmoid(self.W_f(x))  # (B, d_model)

        # Input value via SiLU
        u = F.silu(self.W_u(x))  # (B, d_model)

        # State-expanded versions of forget gate and value
        forget_exp = f.repeat_interleave(E, dim=-1)  # (B, d_model*E)
        i_exp = 1.0 - forget_exp  # (B, d_model*E)
        u_exp = u.repeat_interleave(E, dim=-1)  # (B, d_model*E)

        # Recurrence
        h_new = forget_exp * h + i_exp * u_exp  # (B, d_model*E)

        # Contract: reshape → sum over E axis, normalise
        y = h_new.reshape(B, self.d_model, E).sum(dim=-1) / math.sqrt(E)  # (B, d_model)

        # Output gate
        o = torch.sigmoid(self.W_o(x))  # (B, d_model)

        out = o * y  # (B, d_model)
        return out, h_new


# ---------------------------------------------------------------------------
# HGRN2Layer: full-sequence processing
# ---------------------------------------------------------------------------


class HGRN2Layer(nn.Module):
    """Full-sequence HGRN2 layer.

    Unrolls HGRN2Cell over the time dimension T using a Python loop.

    Args:
        d_model:     Channel dimension.
        expand:      State expansion factor E.
        lower_bound: Minimum forget-gate value.
    """

    def __init__(
        self,
        d_model: int,
        expand: int = 8,
        lower_bound: float = 1.0 / 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.cell = HGRN2Cell(d_model, expand=expand, lower_bound=lower_bound)

    def forward(self, x: Tensor) -> Tensor:
        """Process a full sequence.

        Args:
            x: (B, T, d_model) input sequence.

        Returns:
            out: (B, T, d_model) output sequence.
        """
        B, T, D = x.shape
        h = x.new_zeros(B, D * self.expand)

        outputs = []
        for t in range(T):
            out_t, h = self.cell.step(x[:, t, :], h)
            outputs.append(out_t)

        return torch.stack(outputs, dim=1)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# SwiGLU FFN (local, no external deps)
# ---------------------------------------------------------------------------


class _SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network: gate_proj × silu(up_proj) → down_proj."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# HGRN2Block: HGRN2Layer + SwiGLU FFN with pre-norm and residuals
# ---------------------------------------------------------------------------


class HGRN2Block(nn.Module):
    """HGRN2 block: pre-norm HGRN2Layer + pre-norm SwiGLU FFN with residuals.

    Forward:
        x = x + HGRN2Layer(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    Args:
        d_model: Channel dimension.
        d_ff:    Inner dimension of the SwiGLU FFN.
        expand:  State expansion factor for HGRN2Layer.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        expand: int = 8,
    ) -> None:
        super().__init__()
        self.rnn_norm = RMSNorm(d_model)
        self.rnn = HGRN2Layer(d_model, expand=expand)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = _SwiGLUFFN(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        x = x + self.rnn(self.rnn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# HGRN2Model: full language model backbone
# ---------------------------------------------------------------------------


class HGRN2Model(nn.Module):
    """HGRN2 language model backbone.

    Architecture:
        Embedding → n_layers x HGRN2Block → RMSNorm → (B, T, d_model)

    Args:
        vocab_size:   Vocabulary size for the token embedding.
        d_model:      Model dimension.
        d_ff:         Inner dimension of each SwiGLU FFN.
        n_layers:     Number of HGRN2Blocks to stack.
        expand:       State expansion factor for each HGRN2Layer.
        max_seq_len:  Maximum sequence length (currently unused; reserved for
                      future positional encodings).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        n_layers: int,
        expand: int = 8,
        max_seq_len: int = 8192,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [HGRN2Block(d_model, d_ff, expand=expand) for _ in range(n_layers)]
        )
        self.norm_f = RMSNorm(d_model)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input_ids: (B, T) long tensor of token indices.

        Returns:
            (B, T, d_model) float tensor of hidden states.
        """
        x = self.embed(input_ids)  # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)  # (B, T, d_model)
        return x
