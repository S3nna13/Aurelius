"""TTT-Linear Layer — Test-Time Training as a sequence model.

Reference: Sun et al. 2024, "Learning to (Learn at Test Time): RNNs with
Expressive Hidden States", arXiv:2407.04620.

Notation matches the paper:
  W   — hidden state (weight matrix, d×d)
  W_0 — learnable initial / target weight matrix
  x_t — input token at position t
  ŷ_t — prediction:  ŷ_t = W_t x_t
  z_t — target:      z_t = W_0 x_t
  ℓ_t — reconstruction loss: ‖ŷ_t − z_t‖² / 2
  η   — inner learning rate (lr)
  W_{t+1} = W_t − η · ∂ℓ_t/∂W_t  =  W_t − η · (ŷ_t − z_t) xₜᵀ

Key design choices
------------------
* Sequential implementation — avoids parallel-scan complexity while
  being fully correct and differentiable through W_0.
* W is maintained as a *local* variable per sequence; only W_0 (the
  initial/target weight) is a trained parameter.
* Input/output projections (θ_K, θ_V, θ_Q) keep the interface
  compatible with a standard transformer residual stream, following
  Section 2 of the paper (the "TTT layer" framing).
* use_ln=True applies layer-norm on the final output, as recommended
  in the paper's ablation (Appendix B).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TTTConfig:
    """Hyperparameters for a TTT-Linear layer (paper §3)."""

    d_model: int = 64  # token dimension d
    mini_batch_size: int = 16  # tokens per W-update mini-batch (b in paper)
    lr: float = 0.01  # inner learning rate η
    use_ln: bool = True  # apply LayerNorm to output (paper Appendix B)


# ---------------------------------------------------------------------------
# TTT-Linear Layer
# ---------------------------------------------------------------------------


class TTTLinearLayer(nn.Module):
    """TTT-Linear: an RNN whose hidden state is a weight matrix W ∈ ℝ^{d×d}.

    For each input sequence the layer:
    1. Initialises W ← W_0  (the learnable initial state).
    2. Iterates over T time-steps, updating W via a gradient step on the
       self-supervised reconstruction loss  ℓ_t = ‖W x_t − W_0 x_t‖² / 2.
    3. Emits output o_t = W_{t+1} x_t  after every update.

    The layer also wraps the standard input / output projections from the
    paper (θ_K for keys / queries, θ_V for values, θ_Q for output query),
    so it can be used as a drop-in attention replacement in a transformer.

    Args:
        config: TTTConfig instance.
    """

    def __init__(self, config: TTTConfig) -> None:
        super().__init__()
        d = config.d_model

        self.d_model = d
        self.mini_batch_size = config.mini_batch_size
        self.lr = config.lr
        self.use_ln = config.use_ln

        # W_0 — learnable initial state / reconstruction target
        # Shape (d, d); initialised as a scaled identity for stability.
        self.W_0 = nn.Parameter(torch.eye(d) * 0.01)

        # Input projections:  x_t → k_t / q_t (keys & queries share θ_K)
        self.theta_K = nn.Linear(d, d, bias=False)
        # Value projection: x_t → v_t  (used as the "target" token)
        self.theta_V = nn.Linear(d, d, bias=False)
        # Output query: x_t → q_t'  (paper eq. 6)
        self.theta_Q = nn.Linear(d, d, bias=False)

        # Optional layer-norm on the output (paper Appendix B)
        if self.use_ln:
            self.ln = nn.LayerNorm(d)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_W(self, W: Tensor, k_t: Tensor, v_t: Tensor) -> Tensor:
        """Perform one gradient-descent step on W for token t.

        Paper eq. (3):
            ŷ_t = W k_t
            z_t = W_0 k_t     (W_0 is the target network — fixed wrt this step)
            ℓ_t = ‖ŷ_t − z_t‖² / 2
            W ← W − η · (ŷ_t − z_t) kₜᵀ

        Args:
            W:   current hidden state,  shape (d, d)
            k_t: key vector for token t, shape (d,)
            v_t: value vector (unused by TTT-Linear target; kept for API
                 consistency with TTT-MLP variant), shape (d,)

        Returns:
            Updated W,  shape (d, d).
        """
        # Prediction using current W
        y_hat = W @ k_t  # (d,)
        # Target using fixed W_0
        z = self.W_0 @ k_t  # (d,)
        # Gradient of ℓ_t w.r.t. W  is  (ŷ_t − z_t) kₜᵀ
        grad = torch.outer(y_hat - z, k_t)  # (d, d)
        return W - self.lr * grad

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Apply TTT-Linear sequentially over the token dimension.

        Args:
            x: input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, d = x.shape
        assert d == self.d_model, f"Input last-dim {d} ≠ d_model {self.d_model}"  # noqa: S101

        # Project inputs (paper §2.2)
        K = self.theta_K(x)  # (B, T, d)  — keys
        V = self.theta_V(x)  # (B, T, d)  — values (unused in TTT-Linear loss)
        Q = self.theta_Q(x)  # (B, T, d)  — output queries

        outputs = torch.zeros(B, T, d, dtype=x.dtype, device=x.device)

        for b in range(B):
            # Each sequence gets its own W trajectory (no cross-batch leakage)
            W = self.W_0.clone()  # (d, d)

            for t in range(T):
                k_t = K[b, t]  # (d,)
                v_t = V[b, t]  # (d,)

                # Update W via gradient step on ℓ_t  (paper eq. 3)
                W = self._update_W(W, k_t, v_t)

                # Output: query the updated W  (paper eq. 4 / 6)
                q_t = Q[b, t]  # (d,)
                o_t = W @ q_t  # (d,)
                outputs[b, t] = o_t

        if self.use_ln:
            outputs = self.ln(outputs)

        return outputs
