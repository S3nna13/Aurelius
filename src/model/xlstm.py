"""xLSTM: Extended Long Short-Term Memory.

Reference: Beck et al., 2024 — "xLSTM: Extended Long Short-Term Memory".
arXiv:2405.04517.  Variable names follow paper notation directly.

Architecture overview
---------------------
xLSTM introduces two new LSTM variants that overcome classical LSTM limits:

sLSTM (Section 2) — scalar memory with exponential gating + normaliser state:
    i_t  = exp(z_i_t)                            # exponential input gate
    f_t  = exp(z_f_t) or sigmoid(z_f_t)         # exponential/sigmoid forget gate
    m_t  = max(f_t * m_{t-1}, i_t)              # stabiliser (prevents overflow)
    i'_t = i_t / m_t                            # normalised input gate
    f'_t = f_t * m_{t-1} / m_t                  # normalised forget gate
    c_t  = f'_t * c_{t-1} + i'_t * z_v_t        # cell state
    n_t  = f'_t * n_{t-1} + i'_t                # normaliser state
    h_t  = z_o_t * tanh(c_t) / max(|n_t|, 1)   # output

mLSTM (Section 3) — matrix memory with query/key/value projections:
    q_t  = W_q x_t  ∈ R^d
    k_t  = W_k x_t  ∈ R^d
    v_t  = W_v x_t  ∈ R^d
    i_t  = exp(z_i_t),  f_t = exp(z_f_t)
    m_t  = max(f_t * m_{t-1}, i_t)
    i'_t = i_t / m_t,   f'_t = f_t * m_{t-1} / m_t
    C_t  = f'_t * C_{t-1} + i'_t * (v_t ⊗ k_t)   # outer-product update
    n_t  = f'_t * n_{t-1} + i'_t * k_t
    h_t  = o_t * (C_t q_t) / max(|n_t^T q_t|, 1)

xLSTM block: cell + pre-norm (RMSNorm) + residual connection.
xLSTM model: stack of xLSTM blocks with configurable block_types list.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rms_norm import RMSNorm

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# sLSTM state: (c, n, m) each (B, d_model)
sLSTMState = Tuple[Tensor, Tensor, Tensor]

# mLSTM state: (C, n, m) where C: (B, d_head, d_head), n: (B, d_head), m: (B,)
mLSTMState = Tuple[Tensor, Tensor, Tensor]


# ---------------------------------------------------------------------------
# sLSTM Cell  (Section 2.1)
# ---------------------------------------------------------------------------

class sLSTMCell(nn.Module):
    """Scalar-memory LSTM cell with exponential gating and normaliser state.

    Single-step recurrence: forward(x_t, state) -> (h_t, new_state).

    State tuple: (c, n, m) where
        c — cell state,       shape (B, d_model)
        n — normaliser state, shape (B, d_model)
        m — stabiliser log,   shape (B, d_model)  [stores running max in log-space]

    Args:
        d_model: Dimension of input and output.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

        # Input projections: produce z_i, z_f, z_v, z_o from x_t
        # All are linear projections of the input (no bias needed for gates,
        # bias for value/output keeps expressivity).
        self.W_i = nn.Linear(d_model, d_model, bias=True)   # z_i_t (log of input gate)
        self.W_f = nn.Linear(d_model, d_model, bias=True)   # z_f_t (log of forget gate)
        self.W_v = nn.Linear(d_model, d_model, bias=True)   # z_v_t (value / candidate)
        self.W_o = nn.Linear(d_model, d_model, bias=True)   # z_o_t (output gate, sigmoid)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> sLSTMState:
        """Return zero-initialised state (c, n, m)."""
        zeros = torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
        # m starts at -inf so the first max picks up i_t unconditionally
        m0 = torch.full((batch_size, self.d_model), float('-inf'), device=device, dtype=dtype)
        return (zeros, zeros, m0)

    def forward(
        self,
        x_t: Tensor,                          # (B, d_model)
        state: Optional[sLSTMState] = None,   # (c, n, m) or None
    ) -> Tuple[Tensor, sLSTMState]:           # (h_t, new_state)
        """Single-step sLSTM forward pass.

        Args:
            x_t:   Input at time t, shape (B, d_model).
            state: Previous (c_{t-1}, n_{t-1}, m_{t-1}) or None for zero init.

        Returns:
            h_t:       Output at time t, shape (B, d_model).
            new_state: Updated (c_t, n_t, m_t).
        """
        B = x_t.shape[0]
        device, dtype = x_t.device, x_t.dtype

        if state is None:
            state = self.init_state(B, device, dtype)

        c_prev, n_prev, m_prev = state

        # --- Gate pre-activations (linear projections) ---
        z_i_t = self.W_i(x_t)            # (B, d_model)  — log of input gate
        z_f_t = self.W_f(x_t)            # (B, d_model)  — log of forget gate
        z_v_t = self.W_v(x_t)            # (B, d_model)  — value / candidate
        z_o_t = torch.sigmoid(self.W_o(x_t))  # (B, d_model)  — output gate ∈ (0,1)

        # --- Exponential gates (Section 2.1, eq. 3–4) ---
        # Work in log-space for the stabiliser to avoid large intermediates.
        # m_t = max(z_f_t + m_{t-1}, z_i_t)  [log-space stabiliser]
        m_t = torch.maximum(z_f_t + m_prev, z_i_t)        # (B, d_model)

        # Normalised gates (divide by exp(m_t) to keep values in [0,1])
        # i'_t = exp(z_i_t - m_t)
        i_prime = torch.exp(z_i_t - m_t)                  # (B, d_model)
        # f'_t = exp(z_f_t + m_{t-1} - m_t)
        f_prime = torch.exp(z_f_t + m_prev - m_t)         # (B, d_model)

        # --- State updates ---
        c_t = f_prime * c_prev + i_prime * z_v_t           # cell state
        n_t = f_prime * n_prev + i_prime                   # normaliser state

        # --- Output (eq. 7) ---
        # h_t = o_t * tanh(c_t) / max(|n_t|, 1)
        denom = torch.clamp(n_t.abs(), min=1.0)            # (B, d_model)
        h_t = z_o_t * torch.tanh(c_t) / denom             # (B, d_model)

        return h_t, (c_t, n_t, m_t)


# ---------------------------------------------------------------------------
# mLSTM Cell  (Section 3.1)
# ---------------------------------------------------------------------------

class mLSTMCell(nn.Module):
    """Matrix-memory LSTM cell with query/key/value projections.

    Replaces the scalar cell state c_t ∈ R^d with a matrix C_t ∈ R^{d×d},
    enabling associative memory retrieval via outer-product writes.

    Single-step recurrence: forward(x_t, state) -> (h_t, new_state).

    State tuple: (C, n, m) where
        C — matrix cell state, shape (B, d_head, d_head)
        n — normaliser vector, shape (B, d_head)
        m — stabiliser scalar, shape (B,)

    Args:
        d_model: Input / output dimension.
        d_head:  Head dimension for Q/K/V and matrix memory.  Defaults to d_model.
    """

    def __init__(self, d_model: int, d_head: Optional[int] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head if d_head is not None else d_model

        d = self.d_head
        # Q/K/V projections
        self.W_q = nn.Linear(d_model, d, bias=False)
        self.W_k = nn.Linear(d_model, d, bias=False)
        self.W_v = nn.Linear(d_model, d, bias=False)
        # Scalar gate pre-activations
        self.w_i = nn.Linear(d_model, 1, bias=True)   # z_i_t (log input gate)
        self.w_f = nn.Linear(d_model, 1, bias=True)   # z_f_t (log forget gate)
        # Output gate
        self.W_o = nn.Linear(d_model, d, bias=True)   # produces o_t via sigmoid
        # Output projection back to d_model
        self.out_proj = nn.Linear(d, d_model, bias=False)

        # Scale for query (1/sqrt(d_head) as in paper)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> mLSTMState:
        """Return zero-initialised state (C, n, m)."""
        d = self.d_head
        C0 = torch.zeros(batch_size, d, d, device=device, dtype=dtype)
        n0 = torch.zeros(batch_size, d, device=device, dtype=dtype)
        m0 = torch.full((batch_size,), float('-inf'), device=device, dtype=dtype)
        return (C0, n0, m0)

    def forward(
        self,
        x_t: Tensor,                          # (B, d_model)
        state: Optional[mLSTMState] = None,   # (C, n, m) or None
    ) -> Tuple[Tensor, mLSTMState]:           # (h_t, new_state)
        """Single-step mLSTM forward pass.

        Args:
            x_t:   Input at time t, shape (B, d_model).
            state: Previous (C_{t-1}, n_{t-1}, m_{t-1}) or None for zero init.

        Returns:
            h_t:       Output at time t, shape (B, d_model).
            new_state: Updated (C_t, n_t, m_t).
        """
        B = x_t.shape[0]
        device, dtype = x_t.device, x_t.dtype

        if state is None:
            state = self.init_state(B, device, dtype)

        C_prev, n_prev, m_prev = state   # (B,d,d), (B,d), (B,)

        # --- Q / K / V (Section 3.1) ---
        q_t = self.W_q(x_t) * self.scale    # (B, d_head)
        k_t = self.W_k(x_t) * self.scale    # (B, d_head)
        v_t = self.W_v(x_t)                 # (B, d_head)

        # --- Scalar gate pre-activations ---
        z_i_t = self.w_i(x_t).squeeze(-1)   # (B,)  log input gate
        z_f_t = self.w_f(x_t).squeeze(-1)   # (B,)  log forget gate

        # --- Log-space stabiliser ---
        m_t = torch.maximum(z_f_t + m_prev, z_i_t)   # (B,)

        i_prime = torch.exp(z_i_t - m_t)              # (B,)
        f_prime = torch.exp(z_f_t + m_prev - m_t)     # (B,)

        # --- Matrix memory update: C_t = f'_t * C_{t-1} + i'_t * (v_t ⊗ k_t) ---
        # v_t ⊗ k_t is the outer product: (B, d, 1) @ (B, 1, d) = (B, d, d)
        outer = torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1))   # (B, d_head, d_head)

        # Broadcast scalars over (d_head, d_head)
        C_t = f_prime[:, None, None] * C_prev + i_prime[:, None, None] * outer  # (B,d,d)

        # --- Normaliser update: n_t = f'_t * n_{t-1} + i'_t * k_t ---
        n_t = f_prime[:, None] * n_prev + i_prime[:, None] * k_t   # (B, d_head)

        # --- Output (Section 3.1, eq. 12) ---
        # h_t = o_t * (C_t q_t) / max(|n_t^T q_t|, 1)
        o_t = torch.sigmoid(self.W_o(x_t))              # (B, d_head)

        # C_t q_t: (B, d, d) @ (B, d, 1) → (B, d, 1) → (B, d)
        Cq = torch.bmm(C_t, q_t.unsqueeze(2)).squeeze(2)  # (B, d_head)

        # Denominator: |n_t^T q_t|, at least 1
        denom = torch.clamp((n_t * q_t).sum(dim=-1, keepdim=True).abs(), min=1.0)  # (B, 1)

        h_head = o_t * Cq / denom           # (B, d_head)
        h_t = self.out_proj(h_head)         # (B, d_model)

        return h_t, (C_t, n_t, m_t)


# ---------------------------------------------------------------------------
# xLSTM Block  (Section 4)
# ---------------------------------------------------------------------------

class xLSTMBlock(nn.Module):
    """Single xLSTM block: pre-norm + cell + residual, processes full sequence.

    Args:
        d_model:    Model dimension.
        block_type: ``'slstm'`` or ``'mlstm'``.
        d_head:     Head dimension for mLSTM (ignored for sLSTM).
    """

    VALID_TYPES = frozenset({"slstm", "mlstm"})

    def __init__(
        self,
        d_model: int,
        block_type: str = "mlstm",
        d_head: Optional[int] = None,
    ) -> None:
        if block_type not in self.VALID_TYPES:
            raise ValueError(
                f"block_type must be one of {sorted(self.VALID_TYPES)}, got '{block_type}'"
            )
        super().__init__()
        self.block_type = block_type
        self.norm = RMSNorm(d_model)

        if block_type == "slstm":
            self.cell: nn.Module = sLSTMCell(d_model)
        else:
            self.cell = mLSTMCell(d_model, d_head=d_head)

    def forward(
        self,
        x: Tensor,             # (B, T, d_model)
        state=None,            # per-cell state or None
    ) -> Tuple[Tensor, object]:
        """Process sequence x step-by-step through the cell.

        Args:
            x:     (B, T, d_model)
            state: Initial cell state or None (will be zero-inited inside cell).

        Returns:
            (output, final_state):
                output      — (B, T, d_model) with residual added
                final_state — state tuple after the last time step
        """
        B, T, d = x.shape
        normed = self.norm(x)    # pre-norm

        outputs = []
        h_state = state
        for t in range(T):
            h_t, h_state = self.cell(normed[:, t, :], h_state)
            outputs.append(h_t)

        out = torch.stack(outputs, dim=1)   # (B, T, d_model)
        return x + out, h_state             # residual connection


# ---------------------------------------------------------------------------
# xLSTM Model
# ---------------------------------------------------------------------------

class xLSTMModel(nn.Module):
    """Stack of xLSTM blocks forming a sequence model.

    Args:
        d_model:     Model dimension.
        n_layers:    Number of blocks.
        block_types: List of ``'slstm'`` / ``'mlstm'`` strings of length n_layers.
                     Defaults to alternating ``['mlstm', 'slstm', ...]``.
        d_head:      Head dimension for mLSTM cells (defaults to d_model).

    Forward::

        output, new_hidden_states = model(x, hidden_states=None)

    where ``x`` is (B, T, d_model) and ``hidden_states`` is a list of per-block
    state tuples (or None to zero-initialise).
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        block_types: Optional[List[str]] = None,
        d_head: Optional[int] = None,
    ) -> None:
        super().__init__()

        if block_types is None:
            # Default: alternating mlstm / slstm
            block_types = ["mlstm" if i % 2 == 0 else "slstm" for i in range(n_layers)]

        if len(block_types) != n_layers:
            raise ValueError(
                f"len(block_types)={len(block_types)} must equal n_layers={n_layers}"
            )

        self.d_model = d_model
        self.n_layers = n_layers
        self.block_types = list(block_types)

        self.blocks = nn.ModuleList([
            xLSTMBlock(d_model, bt, d_head=d_head)
            for bt in block_types
        ])
        self.norm_f = RMSNorm(d_model)

    def forward(
        self,
        x: Tensor,                              # (B, T, d_model)
        hidden_states: Optional[List] = None,   # list[state] of length n_layers or None
    ) -> Tuple[Tensor, List]:                   # (output, new_hidden_states)
        """Full forward pass.

        Args:
            x:             Input tensor (B, T, d_model).
            hidden_states: Optional list of per-block states, length n_layers.
                           Pass None to zero-initialise all states.

        Returns:
            (output, new_hidden_states):
                output           — (B, T, d_model) after final RMSNorm
                new_hidden_states — list of updated per-block states
        """
        if hidden_states is None:
            hidden_states = [None] * self.n_layers
        elif len(hidden_states) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} hidden_states, got {len(hidden_states)}"
            )

        new_hidden: List = []
        for i, block in enumerate(self.blocks):
            x, new_s = block(x, hidden_states[i])
            new_hidden.append(new_s)

        x = self.norm_f(x)
        return x, new_hidden
