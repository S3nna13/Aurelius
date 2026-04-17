"""Shampoo: Preconditioned Stochastic Tensor Optimization.

Implements the Shampoo optimizer from Gupta et al. 2018 (arXiv:1802.09568).
Variable notation matches the paper throughout.

For a matrix weight W ∈ R^{m×n} at step t, Shampoo maintains:
    L_t = Σ_{i=1}^{t} G_i G_i^T  ∈ R^{m×m}   (left  Kronecker factor, §2)
    R_t = Σ_{i=1}^{t} G_i^T G_i  ∈ R^{n×n}   (right Kronecker factor, §2)

Preconditioned update (§3):
    θ_t = θ_{t-1} - lr * (L_t + ε I)^{-1/4} G_t (R_t + ε I)^{-1/4}

For 1-D parameters (biases, LayerNorm scales, etc.) there are no Kronecker
factors — a standard SGD-with-momentum update is applied instead (paper §4).

Practical notes:
- Preconditioners are recomputed only every ``update_freq`` steps (default 10)
  to amortise the O(m^3) / O(n^3) eigendecomposition cost.
- Matrix inverse fourth-roots are computed via torch.linalg.eigh (symmetric
  eigendecomposition): M = V Λ V^T  →  M^{-1/4} = V Λ^{-1/4} V^T.
- ε regularisation keeps Λ^{-1/4} well-defined even for near-singular L/R.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Helper: symmetric matrix inverse fourth-root  M^{-1/4}
# ---------------------------------------------------------------------------

def _matrix_inverse_fourth_root(M: Tensor, epsilon: float) -> Tensor:
    """Compute (M + ε I)^{-1/4} via symmetric eigendecomposition.

    Args:
        M:       Square symmetric matrix, shape (k, k).
        epsilon: Regularisation added to the diagonal before inversion.

    Returns:
        Tensor of shape (k, k): the inverse fourth-root of M + ε I.
    """
    k = M.shape[0]
    M_reg = M + epsilon * torch.eye(k, dtype=M.dtype, device=M.device)
    # torch.linalg.eigh returns eigenvalues in ascending order, vectors as cols
    Lambda, V = torch.linalg.eigh(M_reg)                  # Λ, V (§3)
    Lambda_inv4 = Lambda.clamp(min=epsilon).pow(-0.25)    # Λ^{-1/4}
    return V @ torch.diag(Lambda_inv4) @ V.t()            # V Λ^{-1/4} V^T


# ---------------------------------------------------------------------------
# ShampooOptimizer
# ---------------------------------------------------------------------------

class ShampooOptimizer(Optimizer):
    """Shampoo optimizer (Gupta et al. 2018, arXiv:1802.09568).

    Args:
        params:       Iterable of parameters or parameter groups.
        lr:           Global learning rate (α in the paper).
        momentum:     SGD momentum coefficient used for 1-D parameters.
        weight_decay: L2 regularisation coefficient (decoupled).
        update_freq:  Steps between preconditioner recomputations (K, §3).
        epsilon:      Regularisation added to L and R before inversion (ε, §3).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        update_freq: int = 10,
        epsilon: float = 1e-6,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if update_freq < 1:
            raise ValueError(f"update_freq must be >= 1, got {update_freq}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            update_freq=update_freq,
            epsilon=epsilon,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Perform a single optimisation step.

        Args:
            closure: Optional closure that re-evaluates the model and returns
                     the loss (follows the standard PyTorch convention).

        Returns:
            Scalar loss if ``closure`` is provided, else ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group["lr"]
            momentum     = group["momentum"]
            weight_decay = group["weight_decay"]
            update_freq  = group["update_freq"]
            epsilon      = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                G_t: Tensor = p.grad  # gradient at step t

                # Optional weight-decay: modify gradient in-place copy
                if weight_decay != 0.0:
                    G_t = G_t.add(p, alpha=weight_decay)

                state = self.state[p]

                # ----------------------------------------------------------
                # Initialise state on first call
                # ----------------------------------------------------------
                if len(state) == 0:
                    state["step"] = 0
                    if p.dim() == 2:
                        m, n = p.shape
                        # L_t  ∈ R^{m×m},  R_t  ∈ R^{n×n}  (§2)
                        state["L_t"] = torch.zeros(m, m, dtype=p.dtype, device=p.device)
                        state["R_t"] = torch.zeros(n, n, dtype=p.dtype, device=p.device)
                        # Cached preconditioner factors (recomputed every K steps)
                        state["L_inv4"] = torch.eye(m, dtype=p.dtype, device=p.device)
                        state["R_inv4"] = torch.eye(n, dtype=p.dtype, device=p.device)
                    else:
                        # 1-D fallback: keep momentum buffer for SGD
                        state["momentum_buf"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                # ----------------------------------------------------------
                # 2-D parameters: full Shampoo update
                # ----------------------------------------------------------
                if p.dim() == 2:
                    m, n = p.shape  # noqa: F841

                    # Accumulate Kronecker factor statistics (§2)
                    # L_t += G_t G_t^T
                    state["L_t"].add_(G_t @ G_t.t())
                    # R_t += G_t^T G_t
                    state["R_t"].add_(G_t.t() @ G_t)

                    # Recompute preconditioners every K steps (§3 efficiency note)
                    if t % update_freq == 0:
                        state["L_inv4"] = _matrix_inverse_fourth_root(
                            state["L_t"], epsilon
                        )
                        state["R_inv4"] = _matrix_inverse_fourth_root(
                            state["R_t"], epsilon
                        )

                    # Preconditioned update (§3):
                    # θ_t ← θ_{t-1} - lr * L_t^{-1/4} G_t R_t^{-1/4}
                    preconditioned_G = state["L_inv4"] @ G_t @ state["R_inv4"]
                    p.add_(preconditioned_G, alpha=-lr)

                # ----------------------------------------------------------
                # 1-D parameters: standard SGD with momentum (§4 / paper note)
                # ----------------------------------------------------------
                else:
                    buf: Tensor = state["momentum_buf"]
                    buf.mul_(momentum).add_(G_t)
                    p.add_(buf, alpha=-lr)

        return loss
