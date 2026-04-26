"""SOAP optimizer: ShampoO with Adam in the Preconditioner's eigenbasis.

SOAP runs Adam in the eigenbasis of Shampoo's preconditioner. For each 2D
parameter with shape (m, n):
  - Maintain left covariance L (m, m) and right covariance R (n, n) as EMAs
    of G G^T and G^T G respectively.
  - Every `precondition_frequency` steps, compute eigenvectors Q_L, Q_R via
    torch.linalg.eigh of L and R.
  - Rotate the gradient: G' = Q_L^T @ G @ Q_R.
  - Run standard Adam on G' (with exp_avg / exp_avg_sq maintained in the
    rotated eigenbasis).
  - Rotate the Adam update back: U = Q_L @ Adam_update(G') @ Q_R^T.
  - Apply decoupled weight decay: theta <- theta - lr * (U + wd * theta).

For parameters with ndim != 2 the optimizer falls back to AdamW.

Reference: Vyas et al. 2024, "SOAP: Improving and Stabilizing Shampoo using
Adam", arXiv:2409.11321.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class SOAP(Optimizer):
    """SOAP (ShampoO with Adam in the Preconditioner's eigenbasis) optimizer.

    Args:
        params: Iterable of parameters or parameter groups.
        lr: Learning rate (default: 3e-3).
        betas: Adam momentum coefficients (default: (0.95, 0.95)).
        shampoo_beta: EMA coefficient for the Shampoo L/R covariance matrices
            (default: 0.95).
        eps: Term added to the denominator for numerical stability
            (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 0.01).
        precondition_frequency: Number of optimizer steps between eigen-
            decomposition refreshes of Q_L and Q_R (default: 10).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: tuple = (0.95, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"Invalid betas (must be 2-tuple): {betas}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= shampoo_beta < 1.0:
            raise ValueError(f"Invalid shampoo_beta: {shampoo_beta}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not isinstance(precondition_frequency, int) or precondition_frequency < 1:
            raise ValueError(f"Invalid precondition_frequency: {precondition_frequency}")

        defaults = dict(
            lr=lr,
            betas=tuple(betas),
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            eps = group["eps"]
            wd = group["weight_decay"]
            freq = group["precondition_frequency"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SOAP does not support sparse gradients.")

                state = self.state[p]

                if p.ndim == 2:
                    self._step_2d(
                        p,
                        grad,
                        state,
                        lr,
                        beta1,
                        beta2,
                        shampoo_beta,
                        eps,
                        wd,
                        freq,
                    )
                else:
                    self._step_adamw(
                        p,
                        grad,
                        state,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        wd,
                    )

        return loss

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _step_2d(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        lr: float,
        beta1: float,
        beta2: float,
        shampoo_beta: float,
        eps: float,
        wd: float,
        freq: int,
    ) -> None:
        m, n = p.shape

        # Lazy state initialisation.
        if len(state) == 0:
            state["step"] = 0
            state["L"] = torch.zeros(m, m, device=p.device, dtype=p.dtype)
            state["R"] = torch.zeros(n, n, device=p.device, dtype=p.dtype)
            state["Q_L"] = None
            state["Q_R"] = None
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        state["step"] += 1
        step = state["step"]

        # Update Shampoo covariance EMAs.
        L = state["L"]
        R = state["R"]
        L.mul_(shampoo_beta).add_(grad @ grad.t(), alpha=1.0 - shampoo_beta)
        R.mul_(shampoo_beta).add_(grad.t() @ grad, alpha=1.0 - shampoo_beta)

        # Refresh eigenvectors periodically (and on the very first step so
        # we always have a valid rotation available).
        if state["Q_L"] is None or (step % freq == 0):
            # eigh returns ascending eigenvalues; we only need eigenvectors.
            # Symmetrise for numerical stability.
            L_sym = 0.5 * (L + L.t())
            R_sym = 0.5 * (R + R.t())
            try:
                _, Q_L = torch.linalg.eigh(L_sym)
                _, Q_R = torch.linalg.eigh(R_sym)
                state["Q_L"] = Q_L
                state["Q_R"] = Q_R
            except Exception:
                # If decomposition fails (e.g. all-zero covariance), fall
                # back to identity rotations for this step.
                if state["Q_L"] is None:
                    state["Q_L"] = torch.eye(m, device=p.device, dtype=p.dtype)
                    state["Q_R"] = torch.eye(n, device=p.device, dtype=p.dtype)

        Q_L = state["Q_L"]
        Q_R = state["Q_R"]

        # Rotate the gradient into the eigenbasis.
        grad_rot = Q_L.t() @ grad @ Q_R

        # Adam moments in the rotated space.
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad_rot, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_rot, grad_rot, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step

        denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
        update_rot = (exp_avg / bias_correction1) / denom

        # Rotate the update back to the parameter space.
        update = Q_L @ update_rot @ Q_R.t()

        # Decoupled weight decay.
        if wd != 0:
            p.add_(p, alpha=-lr * wd)
        p.add_(update, alpha=-lr)

    def _step_adamw(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        wd: float,
    ) -> None:
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        state["step"] += 1
        step = state["step"]

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step

        denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
        update = (exp_avg / bias_correction1) / denom

        if wd != 0:
            p.add_(p, alpha=-lr * wd)
        p.add_(update, alpha=-lr)
