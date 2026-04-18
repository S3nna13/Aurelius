"""Gated Delta Net — Gated Delta Rule Linear Transformer (Schiff et al., arXiv:2412.06464).

Reference: "Gated Delta Networks: Improving Mamba2 with Delta Rule",
           Schiff et al., 2024.

Key idea: Augments the delta rule with a *forget gate* α_t (inspired by
Mamba's selective SSM) that allows the model to adaptively erase stale
information from its state, and a *write gate* β_t that controls how
strongly each new (v, k) pair is written in.

State update rule (matches paper notation):
    S_t = α_t ⊙ S_{t-1} + β_t · (v_t ⊗ k_t)
    y_t = S_t q_t

Where:
    k_t = normalize(K x_t)      key projection + L2 normalisation
    q_t = Q x_t                 query projection
    v_t = V x_t                 value projection
    α_t = sigmoid(A x_t)        forget gate — scalar broadcast over state
    β_t = sigmoid(B x_t)        write gate — scalar per head

The difference from the plain delta rule (DeltaNet): there is an explicit
forget gate α_t that multiplies the *entire* previous state, enabling
selective memory like Mamba's selective scan.  (Plain DeltaNet has no
such term — it only error-corrects the state via the residual
(v - W k) ⊗ k.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GatedDeltaNetConfig:
    """Hyper-parameters for a single GatedDeltaNet layer.

    Attributes:
        d_model:    Model (embedding) dimension.
        d_head:     Per-head dimension.  Must divide evenly into d_model * n_heads
                    (they are independent: inner dim = n_heads * d_head).
        n_heads:    Number of independent gated-delta-rule heads.
        beta_init:  Initial bias for the write-gate linear layer (shifts the
                    sigmoid so that β starts near ``beta_init`` at init time).
    """

    d_model: int = 64
    d_head: int = 16
    n_heads: int = 4
    beta_init: float = 0.5


# ---------------------------------------------------------------------------
# GatedDeltaNetLayer
# ---------------------------------------------------------------------------


class GatedDeltaNetLayer(nn.Module):
    """Single-layer Gated Delta Net over a (B, T, d_model) sequence.

    Projects the input to q, k, v, α (forget gate) and β (write gate) for
    each head, then unrolls the gated delta rule recurrence over the time
    dimension, and finally projects the concatenated head outputs back to
    d_model.

    Args:
        config: GatedDeltaNetConfig with all hyper-parameters.
    """

    def __init__(self, config: GatedDeltaNetConfig) -> None:
        super().__init__()
        self.config = config
        H = config.n_heads
        D = config.d_head
        inner = H * D  # total inner dimension (output of all heads)

        # Query, key, value projections
        self.q_proj = nn.Linear(config.d_model, inner, bias=False)
        self.k_proj = nn.Linear(config.d_model, inner, bias=False)
        self.v_proj = nn.Linear(config.d_model, inner, bias=False)

        # Forget-gate projection: α_t ∈ (0, 1), one scalar per head
        self.alpha_proj = nn.Linear(config.d_model, H, bias=True)

        # Write-gate projection: β_t ∈ (0, 1), one scalar per head
        self.beta_proj = nn.Linear(config.d_model, H, bias=True)

        # Initialise beta bias so sigmoid(bias) ≈ beta_init
        if config.beta_init > 0.0 and config.beta_init < 1.0:
            import math
            init_bias = math.log(config.beta_init / (1.0 - config.beta_init))
            nn.init.constant_(self.beta_proj.bias, init_bias)

        # Output projection: inner → d_model
        self.out_proj = nn.Linear(inner, config.d_model, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence through the gated delta rule.

        Args:
            x:     Input tensor of shape (B, T, d_model).
            state: Optional initial state of shape
                   (B, n_heads, d_head, d_head).  Pass ``None`` to start
                   from an all-zeros state.

        Returns:
            output: Tensor of shape (B, T, d_model).
            final_state: Tensor of shape (B, n_heads, d_head, d_head)
                         representing the state after the last time step.
        """
        B, T, _ = x.shape
        H, D = self.config.n_heads, self.config.d_head

        # ---- projections ------------------------------------------------
        Q = self.q_proj(x).view(B, T, H, D)     # (B, T, H, D)
        K = self.k_proj(x).view(B, T, H, D)     # (B, T, H, D)
        V = self.v_proj(x).view(B, T, H, D)     # (B, T, H, D)
        # Gates: (B, T, H)  — in (0, 1)
        alpha = torch.sigmoid(self.alpha_proj(x))  # forget gate
        beta  = torch.sigmoid(self.beta_proj(x))   # write gate

        # L2-normalise keys along the head dimension (prevents state explosion)
        K = F.normalize(K, p=2, dim=-1)             # (B, T, H, D)

        # ---- initialise recurrent state ---------------------------------
        if state is None:
            S = x.new_zeros(B, H, D, D)
        else:
            S = state  # (B, H, D, D)

        # ---- recurrence over time --------------------------------------
        # S_t = α_t ⊙ S_{t-1} + β_t · (v_t ⊗ k_t)
        # y_t = S_t q_t
        outputs: list[torch.Tensor] = []
        for t in range(T):
            q_t     = Q[:, t, :, :]      # (B, H, D)
            k_t     = K[:, t, :, :]      # (B, H, D)
            v_t     = V[:, t, :, :]      # (B, H, D)
            alpha_t = alpha[:, t, :]     # (B, H)  — scalar per head
            beta_t  = beta[:, t, :]      # (B, H)  — scalar per head

            # Outer product: v_t ⊗ k_t  — shape (B, H, D, D)
            # [b, h, i, j] = v_t[b,h,i] * k_t[b,h,j]
            outer = torch.einsum("bhi,bhj->bhij", v_t, k_t)

            # Broadcast gates to (B, H, 1, 1) for elementwise state ops
            alpha_exp = alpha_t.unsqueeze(-1).unsqueeze(-1)   # (B, H, 1, 1)
            beta_exp  = beta_t.unsqueeze(-1).unsqueeze(-1)    # (B, H, 1, 1)

            # State update: S_t = α_t * S_{t-1} + β_t * (v_t ⊗ k_t)
            S = alpha_exp * S + beta_exp * outer              # (B, H, D, D)

            # Read out: y_t = S_t q_t
            # S[b,h,:,:] @ q_t[b,h,:]  =>  (B, H, D)
            y_t = torch.einsum("bhij,bhj->bhi", S, q_t)      # (B, H, D)
            outputs.append(y_t.unsqueeze(1))                  # (B, 1, H, D)

        # (B, T, H, D) -> (B, T, H*D)
        out = torch.cat(outputs, dim=1).view(B, T, H * D)

        return self.out_proj(out), S   # (B, T, d_model), (B, H, D, D)
