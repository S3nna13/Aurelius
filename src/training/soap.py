"""SOAP optimizer — Shampoo as Adam Preconditioner.

Reference: Vyas et al., "SOAP: Improving and Stabilizing Shampoo using Adam",
arXiv:2409.11321 (2024).

Key idea: For each 2D weight matrix W (m×n), maintain Kronecker-factor
preconditioners L (m×m) and R (n×n) as EMAs of G@G^T and G^T@G respectively.
Every `precond_freq` steps, eigen-decompose L and R to obtain orthonormal bases
Q_L and Q_R.  Adam's first and second moments are maintained in the projected
eigenspace; the preconditioned update is projected back to parameter space.
1D parameters (biases, norms) fall back to plain Adam.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class SOAP(Optimizer):
    """SOAP: Shampoo as Adam Preconditioner.

    For each 2D parameter W (m×n):
      - L (m×m): left Kronecker factor EMA of G@G^T
      - R (n×n): right Kronecker factor EMA of G^T@G
      - Every `precond_freq` steps: eigen-decompose L→(Q_L, λ_L), R→(Q_R, λ_R)
      - Project gradient into eigenspace: G_hat = Q_L^T @ G @ Q_R
      - Apply Adam (m1, m2) in eigenspace
      - Project back: update = Q_L @ adam_update @ Q_R^T

    For 1D parameters (bias, scale): plain Adam.

    Args:
        params: model parameters
        lr: learning rate (default 3e-4)
        betas: (beta1, beta2) for Adam moments (default (0.95, 0.95))
        eps: Adam epsilon (default 1e-8)
        weight_decay: L2 weight decay applied as gradient penalty (default 0.01)
        precond_freq: steps between preconditioner updates (default 10)
        precond_beta: EMA decay for Kronecker factors (default 0.999)
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precond_freq: int = 10,
        precond_beta: float = 0.999,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            precond_freq=precond_freq,
            precond_beta=precond_beta,
        )
        super().__init__(params, defaults)

    def _update_preconditioner(self, state: dict, grad: torch.Tensor) -> None:
        """Update Kronecker factors and their eigenbases.

        L = precond_beta * L + (1 - precond_beta) * G @ G^T
        R = precond_beta * R + (1 - precond_beta) * G^T @ G

        Eigendecomposition is performed every `precond_freq` steps (but NOT on
        step 0 — no gradient history yet).  Q_L / Q_R are stored in state.
        """
        precond_beta = state["precond_beta"]
        step = state["step"]
        precond_freq = state["precond_freq"]

        L = state["L"]
        R = state["R"]

        # EMA update of Kronecker factors
        L.mul_(precond_beta).add_(grad @ grad.T, alpha=1.0 - precond_beta)
        R.mul_(precond_beta).add_(grad.T @ grad, alpha=1.0 - precond_beta)

        # Refresh eigenbases at the requested frequency (skip step 0)
        if step > 0 and step % precond_freq == 0:
            # torch.linalg.eigh returns eigenvalues in ascending order;
            # eigenvectors form columns of Q.
            _, Q_L = torch.linalg.eigh(L)
            _, Q_R = torch.linalg.eigh(R)
            state["Q_L"].copy_(Q_L)
            state["Q_R"].copy_(Q_R)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimisation step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            precond_freq = group["precond_freq"]
            precond_beta = group["precond_beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Weight decay as gradient penalty
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                is_2d = p.dim() == 2

                # ── Initialise state on first encounter ──────────────────────
                if len(state) == 0:
                    state["step"] = 0
                    if is_2d:
                        m, n = p.shape
                        state["exp_avg"] = torch.zeros(m, n, device=p.device, dtype=p.dtype)
                        state["exp_avg_sq"] = torch.zeros(m, n, device=p.device, dtype=p.dtype)
                        state["L"] = torch.zeros(m, m, device=p.device, dtype=p.dtype)
                        state["R"] = torch.zeros(n, n, device=p.device, dtype=p.dtype)
                        state["Q_L"] = torch.eye(m, device=p.device, dtype=p.dtype)
                        state["Q_R"] = torch.eye(n, device=p.device, dtype=p.dtype)
                        # Stash hyper-params for use inside _update_preconditioner
                        state["precond_freq"] = precond_freq
                        state["precond_beta"] = precond_beta
                    else:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if is_2d:
                    # ── Update Kronecker factors ──────────────────────────────
                    self._update_preconditioner(state, grad)

                    Q_L = state["Q_L"]  # (m, m)
                    Q_R = state["Q_R"]  # (n, n)

                    # Project gradient into eigenspace
                    g_hat = Q_L.T @ grad @ Q_R  # (m, n)

                    # Adam moment updates in eigenspace
                    exp_avg.mul_(beta1).add_(g_hat, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g_hat, g_hat, value=1.0 - beta2)

                    # Bias-corrected moments
                    bc1 = 1.0 - beta1**t
                    bc2 = 1.0 - beta2**t
                    m_hat = exp_avg / bc1
                    v_hat = exp_avg_sq / bc2

                    # Adam update in eigenspace
                    adam_update = m_hat / (v_hat.sqrt().add_(eps))

                    # Project back to parameter space
                    update = Q_L @ adam_update @ Q_R.T

                else:
                    # ── Plain Adam for 1D parameters ─────────────────────────
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bc1 = 1.0 - beta1**t
                    bc2 = 1.0 - beta2**t
                    m_hat = exp_avg / bc1
                    v_hat = exp_avg_sq / bc2

                    update = m_hat / (v_hat.sqrt().add_(eps))

                p.add_(update, alpha=-lr)

        return loss
