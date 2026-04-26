"""Hawk: Pure Real-Gated Linear Recurrence Model.

Reference: De et al., 2024 — "Griffin: Mixing Gated Linear Recurrences with
Local Attention for Efficient Language Models". https://arxiv.org/abs/2402.19427
Section 3.2 describes the RG-LRU (Real-Gated Linear Recurrence Unit).

Hawk is the pure-recurrence variant of Griffin: no local attention layers,
every block is an RG-LRU block. This gives O(1) inference memory w.r.t.
sequence length.

RG-LRU equations (per timestep t):
    r_t = sigmoid(W_r x_t + b_r)                         [recurrence gate]
    i_t = sigmoid(W_i x_t + b_i)                         [input gate]
    a_t = exp(-8 * softplus(Λ) * r_t)                    [per-channel decay]
    h_t = a_t ⊙ h_{t-1} + sqrt(1 - a_t²) ⊙ (i_t ⊙ x_t) [state update]
    y_t = h_t                                             [output]

a_t ∈ (0, 1) by construction → guaranteed stable contractions.
sqrt(1 - a_t²) preserves output variance when x_t has unit variance.

Hawk block:
    1. RMSNorm(x)
    2. Linear projection: [x_proj, gate] = split(Linear(d_model → 2*d_model)(x))
    3. y = RG_LRU(x_proj)
    4. gate_out = SiLU(gate) ⊙ y
    5. out_proj: Linear(d_model → d_model)(gate_out)
    6. residual: x + out_proj

HawkModel stacks n_layers Hawk blocks. The recurrent state h_t has shape
(B, d_model) per layer, and is threaded through forward() calls to enable
stateful (streaming / cached) inference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rms_norm import RMSNorm

# ---------------------------------------------------------------------------
# RG-LRU: Real-Gated Linear Recurrence Unit
# ---------------------------------------------------------------------------


class RGLRU(nn.Module):
    """Real-Gated Linear Recurrence Unit — the core of Hawk.

    Operates on sequences of shape (B, T, d_model) by scanning over T steps
    with a Python loop. Returns the full output sequence and the final hidden
    state h_T of shape (B, d_model).

    Args:
        d_model: channel dimension (both input and hidden state).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

        # Recurrence gate: r_t = sigmoid(W_r x_t + b_r)
        self.W_r = nn.Linear(d_model, d_model, bias=True)
        # Input gate: i_t = sigmoid(W_i x_t + b_i)
        self.W_i = nn.Linear(d_model, d_model, bias=True)
        # Learnable per-channel log-decay parameter Λ, log-spaced initialisation.
        # a_t = exp(-8 * softplus(Λ) * r_t); softplus(Λ) > 0 → a_t ∈ (0,1).
        # We initialise Λ with log-spaced values so channels have diverse
        # decay timescales at start of training.
        lambda_init = torch.linspace(math.log(0.01), math.log(0.99), d_model)  # log-spaced decay
        self.Lambda = nn.Parameter(lambda_init)

    def forward(
        self,
        x: Tensor,
        h0: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Scan x over time and return (outputs, h_T).

        Args:
            x:  (B, T, d_model) input sequence.
            h0: (B, d_model) initial hidden state; zeros if None.

        Returns:
            outputs: (B, T, d_model) — the sequence of h_t values.
            h_T:     (B, d_model)    — hidden state after the last step.
        """
        B, T, D = x.shape

        if h0 is None:
            h = x.new_zeros(B, D)
        else:
            h = h0

        outputs: list[Tensor] = []

        for t in range(T):
            x_t = x[:, t, :]  # (B, D)

            r_t = torch.sigmoid(self.W_r(x_t))  # (B, D) recurrence gate
            i_t = torch.sigmoid(self.W_i(x_t))  # (B, D) input gate

            # a_t = exp(-8 * softplus(Λ) * r_t) ∈ (0, 1)
            # softplus ensures the argument to exp is always negative.
            decay = 8.0 * F.softplus(self.Lambda)  # (D,) positive
            a_t = torch.exp(-decay * r_t)  # (B, D), a_t ∈ (0,1)

            # Variance-preserving state update
            # sqrt(1 - a²) factor keeps output variance ≈ 1 when x has unit var
            scale = torch.sqrt(torch.clamp(1.0 - a_t * a_t, min=0.0))  # (B, D)
            h = a_t * h + scale * (i_t * x_t)  # (B, D)

            outputs.append(h)

        # Stack along time dimension: (B, T, D)
        output_seq = torch.stack(outputs, dim=1)
        return output_seq, h


# ---------------------------------------------------------------------------
# Hawk Block
# ---------------------------------------------------------------------------


class HawkBlock(nn.Module):
    """A single Hawk recurrence block.

    Pre-norm residual block wrapping an RG-LRU layer with a gated projection.

    Forward:
        normed = RMSNorm(x)
        [x_proj, gate] = split(Linear(d_model → 2*d_model)(normed))
        y = RG_LRU(x_proj)
        gate_out = SiLU(gate) ⊙ y
        out = Linear(d_model → d_model)(gate_out)
        return x + out, h_T
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=True)
        self.rglru = RGLRU(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        x: Tensor,
        h0: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x:  (B, T, d_model)
            h0: (B, d_model) or None

        Returns:
            out: (B, T, d_model)
            h_T: (B, d_model)
        """
        residual = x
        normed = self.norm(x)  # (B, T, D)

        projected = self.in_proj(normed)  # (B, T, 2D)
        x_proj, gate = projected.chunk(2, dim=-1)  # each (B, T, D)

        y, h_T = self.rglru(x_proj, h0=h0)  # (B, T, D), (B, D)

        gate_out = F.silu(gate) * y  # (B, T, D)
        out = self.out_proj(gate_out)  # (B, T, D)

        return residual + out, h_T


# ---------------------------------------------------------------------------
# Hawk Model
# ---------------------------------------------------------------------------


class HawkModel(nn.Module):
    """Hawk: stacked RG-LRU blocks (pure recurrence, no attention).

    Args:
        d_model:  model / channel dimension.
        n_layers: number of HawkBlock layers.
        d_state:  dimension of per-layer hidden state (defaults to d_model).
                  Currently only d_state == d_model is supported — the RG-LRU
                  hidden state has the same size as the input channels.

    forward(x, hidden_states=None) → (output, new_hidden_states)
        x:             (B, T, d_model)
        hidden_states: list of n_layers tensors, each (B, d_model), or None.
        output:        (B, T, d_model)
        new_hidden_states: list of n_layers tensors, each (B, d_model)
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int | None = None,
    ) -> None:
        super().__init__()
        if d_state is not None and d_state != d_model:
            raise ValueError(
                f"HawkModel currently requires d_state == d_model, "
                f"got d_state={d_state}, d_model={d_model}."
            )
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_model  # hidden state dimension equals d_model

        self.layers = nn.ModuleList([HawkBlock(d_model) for _ in range(n_layers)])

    def forward(
        self,
        x: Tensor,
        hidden_states: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """
        Args:
            x:             (B, T, d_model) input tensor.
            hidden_states: list of n_layers (B, d_model) tensors (h_0 per layer),
                           or None to default to zero initialisation.

        Returns:
            output:            (B, T, d_model)
            new_hidden_states: list of n_layers (B, d_model) tensors (h_T per layer).
        """
        if hidden_states is None:
            hidden_states = [None] * self.n_layers  # type: ignore[list-item]
        elif len(hidden_states) != self.n_layers:
            raise ValueError(
                f"Expected hidden_states of length {self.n_layers}, got {len(hidden_states)}."
            )

        new_hidden_states: list[Tensor] = []
        h = x
        for layer, h0 in zip(self.layers, hidden_states):
            h, h_T = layer(h, h0=h0)
            new_hidden_states.append(h_T)

        return h, new_hidden_states
