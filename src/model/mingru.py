"""minGRU: Minimal Gated Recurrent Unit with Parallel Training.

Reference: Feng et al., 2024 — "Were RNNs All We Needed?"
           https://arxiv.org/abs/2410.01201

Key insight: by removing h_{t-1} from the gate computation, minGRU's
recurrence becomes a *linear* recurrence that can be parallelized via
the associative (parallel prefix) scan, enabling O(log T) parallel
training while retaining O(1) inference memory.

minGRU equations (paper notation):
    z_t  = sigmoid(Linear_z(x_t))               # forget gate
    h̃_t = Linear_h(x_t)                         # candidate hidden state
    h_t  = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t   # update rule

The recurrence h_t = a_t * h_{t-1} + b_t is an affine map with:
    a_t = 1 - z_t   (coefficient, in (0,1))
    b_t = z_t * h̃_t (value, any real)

Composition of two affine maps (g ∘ f)(x) = g(f(x)):
    (a_g, b_g) ∘ (a_f, b_f) = (a_g * a_f,  a_g * b_f + b_g)

This associativity enables the parallel prefix scan.

For numerical stability we carry log|a| separately and apply log-space
accumulation for the coefficient product, while keeping b in linear space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class MinGRUConfig:
    d_model: int = 64
    expand_factor: float = 1.0  # d_inner = int(d_model * expand_factor)

    @property
    def d_inner(self) -> int:
        return int(self.d_model * self.expand_factor)


# ---------------------------------------------------------------------------
# Associative scan — affine map composition
# ---------------------------------------------------------------------------

def _parallel_scan_affine(log_a: Tensor, b: Tensor) -> Tensor:
    """Parallel prefix scan for the linear recurrence h_t = a_t * h_{t-1} + b_t.

    Uses iterative doubling (upsweep).  After the scan, b[t] holds h_t.

    The associative operator for affine maps f(x) = a*x + b:
        (log_a2, b2) ∘ (log_a1, b1)  =  (log_a1 + log_a2,  exp(log_a2)*b1 + b2)

    We work with log|a| + sign_a to handle a_t ∈ (0,1) safely (log is stable).
    Since a_t = 1 - z_t ∈ (0,1) all log values are well-defined and negative.

    Args:
        log_a: log(a_t) = log(1 - z_t),  shape (B, T, d)
        b:     b_t = z_t * h̃_t,          shape (B, T, d)

    Returns:
        h: shape (B, T, d) — hidden states h_1 .. h_T (assuming h_0 = 0)
    """
    T = log_a.shape[1]

    # Working copies
    la = log_a.clone()   # (B, T, d)
    bv = b.clone()       # (B, T, d)

    stride = 1
    while stride < T:
        # Shift left by stride (bring previous element into alignment)
        la_prev = torch.roll(la, shifts=stride, dims=1)
        bv_prev = torch.roll(bv, shifts=stride, dims=1)

        # Zero out wrapped-around positions:
        #   log_a = 0  =>  a = 1  (neutral coefficient)
        #   b     = 0  =>  addend is 0
        la_prev[:, :stride, :] = 0.0
        bv_prev[:, :stride, :] = 0.0

        # Combined operator: new_la = la + la_prev (log-domain multiply)
        #                    new_bv = exp(la) * bv_prev + bv
        new_bv = torch.exp(la) * bv_prev + bv
        new_la = la + la_prev

        la = new_la
        bv = new_bv
        stride *= 2

    return bv  # h_t for all t


# ---------------------------------------------------------------------------
# MinGRU module
# ---------------------------------------------------------------------------

class MinGRU(nn.Module):
    """Minimal GRU (minGRU) supporting both sequential and parallel modes.

    Sequential mode: classic step-by-step recurrence for inference.
    Parallel mode:   associative scan over affine maps for O(log T) training.
    """

    def __init__(self, config: MinGRUConfig) -> None:
        super().__init__()
        d = config.d_model
        d_inner = config.d_inner

        # z_t = sigmoid(linear_z(x_t))  — forget gate
        self.linear_z = nn.Linear(d, d_inner, bias=True)
        # h̃_t = linear_h(x_t)          — candidate hidden state
        self.linear_h = nn.Linear(d, d_inner, bias=True)
        # Optional output projection when expand_factor != 1
        if d_inner != d:
            self.out_proj = nn.Linear(d_inner, d, bias=False)
        else:
            self.out_proj = nn.Identity()

        self.d_model = d
        self.d_inner = d_inner

    # ------------------------------------------------------------------
    # Sequential (inference) mode
    # ------------------------------------------------------------------

    def forward_sequential(
        self, x: Tensor, h0: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Sequential recurrence for inference.

        Args:
            x:  (B, T, d_model)
            h0: (B, d_inner) or None — initial hidden state (zeros if None)

        Returns:
            output: (B, T, d_model)
            h_T:    (B, d_inner) — final hidden state
        """
        B, T, _ = x.shape
        h = x.new_zeros(B, self.d_inner) if h0 is None else h0

        outputs = []
        for t in range(T):
            xt = x[:, t, :]                        # (B, d_model)
            z = torch.sigmoid(self.linear_z(xt))   # (B, d_inner)
            h_tilde = self.linear_h(xt)            # (B, d_inner)
            h = (1.0 - z) * h + z * h_tilde        # (B, d_inner)
            outputs.append(h)

        out = torch.stack(outputs, dim=1)          # (B, T, d_inner)
        return self.out_proj(out), h

    # ------------------------------------------------------------------
    # Parallel (training) mode — associative scan on affine maps
    # ------------------------------------------------------------------

    def forward_parallel(self, x: Tensor) -> Tensor:
        """Parallel training via associative scan (h_0 = 0 assumed).

        Args:
            x: (B, T, d_model)

        Returns:
            output: (B, T, d_model)
        """
        z = torch.sigmoid(self.linear_z(x))        # (B, T, d_inner)
        h_tilde = self.linear_h(x)                 # (B, T, d_inner)

        # Affine map parameters:  h_t = a_t * h_{t-1} + b_t
        #   a_t = 1 - z_t  ∈ (0, 1)   →  log_a = log(1 - z_t) < 0 (stable)
        #   b_t = z_t * h̃_t           ∈ ℝ
        log_a = torch.log(1.0 - z + 1e-8)         # (B, T, d_inner)
        b = z * h_tilde                             # (B, T, d_inner)

        h_out = _parallel_scan_affine(log_a, b)    # (B, T, d_inner)
        return self.out_proj(h_out)

    # ------------------------------------------------------------------
    # Unified forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        h0: Tensor | None = None,
        use_parallel: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        """Unified forward pass.

        Args:
            x:            (B, T, d_model)
            h0:           (B, d_inner) or None — initial hidden state
            use_parallel: if True and h0 is None, use the parallel scan

        Returns:
            output: (B, T, d_model)
            h_T:    (B, d_inner) if sequential, else None
        """
        if use_parallel and h0 is None:
            return self.forward_parallel(x), None
        return self.forward_sequential(x, h0)
