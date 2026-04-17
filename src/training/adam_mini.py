"""Adam-mini optimizer — Use Fewer Learning Rates To Gain More.

Reference: Zhang et al., "Adam-mini: Use Fewer Learning Rates To Gain More",
           arXiv:2406.16793 (2024).

Key idea: Adam maintains a per-element second moment v_t, which dominates
optimizer memory. Adam-mini replaces v_t with a per-block scalar v_block_t
that is the mean of squared gradients within the block. This cuts v memory
by ~n_heads× for attention weight matrices while preserving convergence.

Update rule (Section 3):
    Standard Adam:  θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)
    Adam-mini:      v_block_t = mean(g_t²)  over elements in block
                    θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_block_t) + ε)

    where m̂_t and v̂_block_t are bias-corrected first and second moments.

Blocking strategy (Section 3.2):
    - Attention Q/K/V/O weight matrices (2D, shape matches d_model × d_model):
        one block per attention head — n_heads blocks, each of size head_dim
        rows. Requires n_heads and head_dim to be provided.
    - All other parameters (FFN, embeddings, biases, norms):
        single block — entire parameter shares one scalar v.

Usage:
    optimizer = AdamMini(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        n_heads=16,
        head_dim=64,
    )
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import Optimizer


class AdamMini(Optimizer):
    """Adam-mini optimizer with head-wise blocked second moments.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (α in the paper). Default: 1e-3.
        betas: Coefficients (β₁, β₂) for first and second moment estimates.
               Default: (0.9, 0.999).
        eps: ε term added to denominator for numerical stability. Default: 1e-8.
        weight_decay: L2 regularisation coefficient (λ). Default: 0.0.
        n_heads: Number of attention heads. When set together with head_dim,
                 enables head-wise blocking for qualifying 2D parameters.
                 Default: None (single-block mode).
        head_dim: Dimension of each attention head (d_head = d_model / n_heads).
                  Must be provided together with n_heads. Default: None.

    Raises:
        ValueError: If only one of n_heads / head_dim is provided, or if
                    hyperparameters are out of range.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        n_heads: int | None = None,
        head_dim: int | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if (n_heads is None) != (head_dim is None):
            raise ValueError(
                "n_heads and head_dim must both be provided or both omitted; "
                f"got n_heads={n_heads}, head_dim={head_dim}"
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_head_blocked(self, param: torch.Tensor, n_heads: int, head_dim: int) -> bool:
        """Return True if *param* qualifies for head-wise blocking.

        A 2D weight matrix qualifies when its first dimension equals
        n_heads * head_dim (i.e. it looks like an attention projection
        matrix of shape [d_model, d_model] where d_model = n_heads * head_dim).
        """
        if param.ndim != 2:
            return False
        d_model = n_heads * head_dim
        return param.shape[0] == d_model

    def _init_state(
        self,
        param: torch.Tensor,
        n_heads: int | None,
        head_dim: int | None,
    ) -> dict:
        """Initialise per-parameter optimizer state.

        State keys:
            step   : int, number of updates applied.
            m      : Tensor[same shape as param], first moment (per-element).
            v      : Tensor[n_heads] if head-blocked, else Tensor[] (scalar),
                     second moment (per-block).
        """
        state: dict = {"step": 0}
        # First moment — always per-element
        state["m"] = torch.zeros_like(param, memory_format=torch.preserve_format)

        # Second moment — per-block scalar(s)
        use_head_blocking = (
            n_heads is not None
            and head_dim is not None
            and self._is_head_blocked(param, n_heads, head_dim)
        )
        if use_head_blocking:
            # One scalar per attention head
            state["v"] = torch.zeros(n_heads, dtype=param.dtype, device=param.device)
            state["n_blocks"] = n_heads
            state["head_dim"] = head_dim
        else:
            # Single block — scalar second moment
            state["v"] = torch.zeros((), dtype=param.dtype, device=param.device)
            state["n_blocks"] = 1
            state["head_dim"] = None

        return state

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Perform a single optimisation step.

        Args:
            closure: Optional closure that re-evaluates the model and
                     returns the loss.

        Returns:
            The loss value if *closure* was provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            β1, β2 = group["betas"]
            lr = group["lr"]
            ε = group["eps"]
            λ = group["weight_decay"]
            n_heads = group["n_heads"]
            head_dim = group["head_dim"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad  # raw gradient g_t

                if g.is_sparse:
                    raise RuntimeError("Adam-mini does not support sparse gradients.")

                state = self.state[p]

                # Lazy initialisation
                if len(state) == 0:
                    state.update(self._init_state(p, n_heads, head_dim))

                state["step"] += 1
                t = state["step"]

                m = state["m"]   # first moment  (shape = p.shape)
                v = state["v"]   # second moment  (scalar or (n_heads,))

                # ---- First moment update: m_t = β₁ m_{t-1} + (1 - β₁) g_t ----
                m.mul_(β1).add_(g, alpha=1.0 - β1)

                # ---- Second moment update (per block) --------------------------
                # v_block_t = β₂ v_block_{t-1} + (1 - β₂) mean(g_t²)_block
                n_blocks = state["n_blocks"]
                blk_head_dim = state["head_dim"]

                if n_blocks > 1 and blk_head_dim is not None:
                    # Head-wise blocking: param shape is (n_heads * head_dim, d_in)
                    # Reshape g to (n_heads, head_dim * d_in) and mean per head
                    g_blocks = g.reshape(n_blocks, -1)      # (n_heads, head_dim * d_in)
                    g2_mean = g_blocks.pow(2).mean(dim=1)   # (n_heads,) — v_block per head
                    v.mul_(β2).add_(g2_mean, alpha=1.0 - β2)
                else:
                    # Single block — scalar mean over all elements
                    g2_mean = g.pow(2).mean()               # scalar
                    v.mul_(β2).add_(g2_mean, alpha=1.0 - β2)

                # ---- Bias correction -------------------------------------------
                bias_corr1 = 1.0 - β1 ** t
                bias_corr2 = 1.0 - β2 ** t

                m_hat = m / bias_corr1          # m̂_t  (same shape as p)

                if n_blocks > 1 and blk_head_dim is not None:
                    # v has shape (n_heads,); expand to (n_heads, 1) for broadcasting
                    v_hat = v / bias_corr2      # (n_heads,)
                    denom = (v_hat.sqrt().unsqueeze(1) + ε)   # (n_heads, 1)

                    # m_hat reshaped to (n_heads, head_dim * d_in)
                    m_hat_blocks = m_hat.reshape(n_blocks, -1)
                    update = (m_hat_blocks / denom).reshape_as(p)
                else:
                    v_hat = v / bias_corr2      # scalar
                    denom = v_hat.sqrt() + ε    # scalar
                    update = m_hat / denom

                # ---- Weight decay (decoupled, AdamW-style) ---------------------
                if λ != 0.0:
                    p.mul_(1.0 - lr * λ)

                # ---- Parameter update ------------------------------------------
                p.add_(update, alpha=-lr)

        return loss
