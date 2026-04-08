"""GLA — Gated Linear Attention (Yang et al. 2024).

Reference: "Gated Linear Attention Transformers with Hardware-Efficient Training"
           Yang et al., 2024.

Key idea: Linear attention augmented with a data-dependent gate that controls how
much each hidden state is retained or forgotten.  This allows the model to focus
on recent tokens (low gate) or remember long history (high gate), all within an
O(N) recurrent formulation.

Recurrent state update:
    h_t = G_t ⊙ h_{t-1} + k_t^T v_t
    o_t = q_t @ h_t

where G_t = sigmoid(gate_proj(x_t)) is in (0, 1)^{head_dim × head_dim},
broadcast over the outer-product state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gated Linear Attention
# ---------------------------------------------------------------------------


class GatedLinearAttention(nn.Module):
    """Gated Linear Attention.

    Recurrent state update per head h:
        h_t = G_t ⊙ h_{t-1} + k_t^T v_t      (head_dim × head_dim state)
        o_t = q_t @ h_t                         (1 × head_dim output)

    G_t = sigmoid(gate_proj(x_t))  ∈ (0, 1)^{n_heads × head_dim}

    The gate vector is broadcast over the key dimension of the state, allowing
    each value dimension to decay at its own learned rate.

    Args:
        d_model:  model dimension.
        n_heads:  number of attention heads.
        head_dim: key/value dimension per head.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner_dim = n_heads * head_dim

        self.q_proj    = nn.Linear(d_model, inner_dim, bias=False)
        self.k_proj    = nn.Linear(d_model, inner_dim, bias=False)
        self.v_proj    = nn.Linear(d_model, inner_dim, bias=False)
        self.gate_proj = nn.Linear(d_model, inner_dim, bias=True)
        self.out_proj  = nn.Linear(inner_dim, d_model, bias=False)

    # ------------------------------------------------------------------
    # Recurrent mode
    # ------------------------------------------------------------------

    def forward_recurrent(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process the full sequence token-by-token (recurrent mode).

        Args:
            x:             (B, L, d_model)
            initial_state: (B, n_heads, head_dim, head_dim) or None.

        Returns:
            output:      (B, L, d_model)
            final_state: (B, n_heads, head_dim, head_dim)
        """
        B, L, _ = x.shape
        H, D = self.n_heads, self.head_dim

        # Project all positions at once — cheaper than per-step linear
        Q = self.q_proj(x).view(B, L, H, D)        # (B, L, H, D)
        K = self.k_proj(x).view(B, L, H, D)        # (B, L, H, D)
        V = self.v_proj(x).view(B, L, H, D)        # (B, L, H, D)
        G = torch.sigmoid(
            self.gate_proj(x).view(B, L, H, D)     # (B, L, H, D)
        )

        # Running state: (B, H, D, D)
        if initial_state is None:
            state = x.new_zeros(B, H, D, D)
        else:
            state = initial_state

        outputs = []
        for t in range(L):
            q_t = Q[:, t, :, :]   # (B, H, D)
            k_t = K[:, t, :, :]   # (B, H, D)
            v_t = V[:, t, :, :]   # (B, H, D)
            g_t = G[:, t, :, :]   # (B, H, D)  gate ∈ (0,1)

            # kv outer product: (B, H, D, D)
            kv_t = torch.einsum("bhd,bhe->bhde", k_t, v_t)

            # Gate broadcast over key dim: g_t is (B, H, D) → (B, H, 1, D)
            # This gates the value dimension of the state
            g_expanded = g_t.unsqueeze(2)  # (B, H, 1, D)

            state = g_expanded * state + kv_t  # (B, H, D, D)

            # Output: q_t @ state   (B, H, D) × (B, H, D, D) → (B, H, D)
            o_t = torch.einsum("bhd,bhde->bhe", q_t, state)  # (B, H, D)
            outputs.append(o_t.unsqueeze(1))  # (B, 1, H, D)

        # (B, L, H, D) → (B, L, H*D)
        out = torch.cat(outputs, dim=1).view(B, L, H * D)
        out = self.out_proj(out)   # (B, L, d_model)
        return out, state

    # ------------------------------------------------------------------
    # Default forward — recurrent mode
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Default forward pass (recurrent mode, returns output only).

        Args:
            x: (B, L, d_model)

        Returns:
            Tensor of shape (B, L, d_model).
        """
        out, _ = self.forward_recurrent(x)
        return out


# ---------------------------------------------------------------------------
# GLA Block
# ---------------------------------------------------------------------------


class GLABlock(nn.Module):
    """A single GLA layer: Gated Linear Attention + SwiGLU FFN with pre-norm.

    Matches the Aurelius block interface (pre-norm + residual, accepts **kwargs).

    Args:
        config: AureliusConfig (uses d_model, n_heads, head_dim, rms_norm_eps,
                d_ff, dropout).
    """

    def __init__(self, config) -> None:
        super().__init__()
        from .ffn import SwiGLUFFN
        from .rms_norm import RMSNorm

        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.gla = GatedLinearAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
        )
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Pre-norm residual forward pass.

        Args:
            x: (B, L, d_model)

        Returns:
            Tensor of shape (B, L, d_model).
        """
        x = x + self.gla(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
