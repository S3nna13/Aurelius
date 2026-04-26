"""DeltaNet — Delta Rule Linear Transformer (Yang et al., arXiv:2406.06484).

Reference: "Parallelizing Linear Transformers with the Delta Rule over
           Sequence Length", Yang et al., 2024.

Key idea: Instead of plain linear attention (which accumulates outer products),
DeltaNet applies the *delta rule* — an error-correcting Hebbian update.  At
each step the state is updated only by the *prediction error* (v_t - W_{t-1}
k_t), scaled by a per-token learning rate β_t.

    W_t = W_{t-1} + β_t * (v_t - W_{t-1} k_t) ⊗ k_t

where ⊗ denotes the outer product.  Normalising k_t prevents state explosion.
Output at each step: o_t = W_t q_t.

This formulation is recurrent (O(1) state size) and causal by construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# DeltaNetCell — single-step recurrence
# ---------------------------------------------------------------------------


class DeltaNetCell(nn.Module):
    """Single-step DeltaNet recurrence.

    Maintains a weight-matrix state W ∈ R^{d_head × d_head} that is updated
    according to the delta rule.

    Args:
        d_head: Head dimension (both key and value dimension).
    """

    def __init__(self, d_head: int) -> None:
        super().__init__()
        self.d_head = d_head

    def step(
        self,
        q: torch.Tensor,  # (..., d_head)
        k: torch.Tensor,  # (..., d_head)  — already L2-normalised
        v: torch.Tensor,  # (..., d_head)
        beta: torch.Tensor,  # (...,)          — scalar per sample
        W_prev: torch.Tensor,  # (..., d_head, d_head)  W[i,j] = state[value_i, key_j]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one step of the delta rule.

        Update:
            prediction  = W_prev @ k               (..., d_head)
            error       = v - prediction            (..., d_head)
            W_new       = W_prev + β * error ⊗ k   (..., d_head, d_head)
            o           = W_new @ q                 (..., d_head)

        Args:
            q:      Query vector, shape (..., d_head).
            k:      Key vector (normalised), shape (..., d_head).
            v:      Value vector, shape (..., d_head).
            beta:   Per-step learning rate, shape (...,).
            W_prev: Previous state, shape (..., d_head, d_head).

        Returns:
            o:     Output vector, shape (..., d_head).
            W_new: Updated state, shape (..., d_head, d_head).
        """
        # prediction: (..., d_head)   W_prev @ k
        prediction = torch.einsum("...ij,...j->...i", W_prev, k)

        # error-correction term
        error = v - prediction  # (..., d_head)

        # outer product: error ⊗ k  — shape (..., d_head, d_head)
        # [i, j] = error_i * k_j
        outer = torch.einsum("...i,...j->...ij", error, k)

        # broadcast beta to (..., 1, 1) for state update
        beta_expanded = beta.unsqueeze(-1).unsqueeze(-1)  # (..., 1, 1)

        W_new = W_prev + beta_expanded * outer  # (..., d_head, d_head)

        # output: W_new @ q
        o = torch.einsum("...ij,...j->...i", W_new, q)  # (..., d_head)

        return o, W_new


# ---------------------------------------------------------------------------
# DeltaNetLayer — full sequence processing by unrolling DeltaNetCell
# ---------------------------------------------------------------------------


class DeltaNetLayer(nn.Module):
    """DeltaNet over a full (B, T, d_model) sequence.

    Projects input to q, k, v, beta per head, unrolls DeltaNetCell over time,
    then projects the concatenated head outputs back to d_model.

    Args:
        d_model:  Model dimension.
        n_heads:  Number of independent delta-rule heads.
        d_head:   Head dimension.  Defaults to d_model // n_heads.
        expand_k: Multiplicative factor for key/value dimension (default 1.0).
                  Kept for API compatibility; currently the key/value dim equals
                  d_head regardless of this value.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int | None = None,
        expand_k: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head if d_head is not None else d_model // n_heads
        inner = n_heads * self.d_head

        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        self.out_proj = nn.Linear(inner, d_model, bias=False)

        self.cell = DeltaNetCell(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — unrolls the delta rule over the time dimension.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head

        # Project all time steps at once
        Q = self.q_proj(x).view(B, T, H, D)  # (B, T, H, D)
        K = self.k_proj(x).view(B, T, H, D)  # (B, T, H, D)
        V = self.v_proj(x).view(B, T, H, D)  # (B, T, H, D)
        # beta: (B, T, H) — per head, per timestep learning rate in (0, 1)
        beta = torch.sigmoid(self.beta_proj(x))  # (B, T, H)

        # Normalise keys to prevent state explosion
        K = F.normalize(K, p=2, dim=-1)  # unit-norm along d_head

        # Initialise per-head state to zeros: (B, H, D, D)
        W = x.new_zeros(B, H, D, D)

        outputs = []
        for t in range(T):
            q_t = Q[:, t, :, :]  # (B, H, D)
            k_t = K[:, t, :, :]  # (B, H, D)
            v_t = V[:, t, :, :]  # (B, H, D)
            beta_t = beta[:, t, :]  # (B, H)

            o_t, W = self.cell.step(q_t, k_t, v_t, beta_t, W)  # (B, H, D), (B, H, D, D)
            outputs.append(o_t.unsqueeze(1))  # (B, 1, H, D)

        # (B, T, H, D) -> (B, T, H*D)
        out = torch.cat(outputs, dim=1).view(B, T, H * D)
        return self.out_proj(out)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# DeltaNetBlock — DeltaNet + FFN + RMSNorm (pre-norm, residual)
# ---------------------------------------------------------------------------


class DeltaNetBlock(nn.Module):
    """Full transformer block with DeltaNet attention + SwiGLU FFN.

    Follows pre-norm residual style:
        x = x + DeltaNetLayer(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Args:
        d_model: Model dimension.
        n_heads: Number of delta-rule heads.
        d_ff:    FFN hidden dimension.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
    ) -> None:
        super().__init__()
        from .rms_norm import RMSNorm

        self.norm1 = RMSNorm(d_model)
        self.delta_net = DeltaNetLayer(d_model=d_model, n_heads=n_heads)
        self.norm2 = RMSNorm(d_model)

        # Inline SwiGLU FFN to avoid config dependency
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm forward pass.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        x = x + self.delta_net(self.norm1(x))
        x = x + self._ffn(self.norm2(x))
        return x
