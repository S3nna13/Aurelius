"""SOAP optimizer with lightweight matrix preconditioning.

This implementation follows the spirit of second-order adaptive optimizers by
maintaining running row/column covariance estimates for matrix-shaped
parameters, then preconditioning gradients with inverse square-root factors.
Vectors and scalars fall back to Adam-style diagonal preconditioning.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


def _matrix_inverse_root(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute a symmetric inverse square root for a PSD matrix."""
    eye = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    evals, evecs = torch.linalg.eigh(matrix + eps * eye)
    inv_root = evecs @ torch.diag_embed(evals.clamp_min(eps).rsqrt()) @ evecs.transpose(-1, -2)
    return inv_root


def _flatten_to_matrix(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Flatten an arbitrary tensor to a 2D matrix for preconditioning."""
    original_shape = tensor.shape
    if tensor.dim() == 0:
        return tensor.reshape(1, 1), original_shape
    if tensor.dim() == 1:
        return tensor.reshape(tensor.numel(), 1), original_shape
    return tensor.reshape(tensor.shape[0], -1), original_shape


def _restore_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Restore a flattened matrix back to the original tensor shape."""
    return matrix.reshape(shape)


class SOAP(Optimizer):
    """Second-order adaptive optimizer with matrix preconditioning."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        precondition_frequency: int = 1,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if not 0 <= shampoo_beta < 1:
            raise ValueError(f"shampoo_beta must be in [0, 1), got {shampoo_beta}")
        if precondition_frequency < 1:
            raise ValueError(
                f"precondition_frequency must be >= 1, got {precondition_frequency}"
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            precondition_frequency = group["precondition_frequency"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if weight_decay != 0.0:
                    grad = grad.add(param, alpha=weight_decay)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                    grad_matrix, _ = _flatten_to_matrix(grad)
                    row_dim, col_dim = grad_matrix.shape
                    state["row_cov"] = torch.zeros(
                        row_dim, row_dim, device=param.device, dtype=param.dtype
                    )
                    state["col_cov"] = torch.zeros(
                        col_dim, col_dim, device=param.device, dtype=param.dtype
                    )
                    state["row_inv_root"] = torch.eye(
                        row_dim, device=param.device, dtype=param.dtype
                    )
                    state["col_inv_root"] = torch.eye(
                        col_dim, device=param.device, dtype=param.dtype
                    )

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                grad_matrix, original_shape = _flatten_to_matrix(grad)
                row_cov = state["row_cov"]
                col_cov = state["col_cov"]
                row_cov.mul_(shampoo_beta).add_(grad_matrix @ grad_matrix.transpose(0, 1), alpha=1 - shampoo_beta)
                col_cov.mul_(shampoo_beta).add_(grad_matrix.transpose(0, 1) @ grad_matrix, alpha=1 - shampoo_beta)

                if state["step"] % precondition_frequency == 0:
                    state["row_inv_root"] = _matrix_inverse_root(row_cov, eps)
                    state["col_inv_root"] = _matrix_inverse_root(col_cov, eps)

                preconditioned = state["row_inv_root"] @ grad_matrix @ state["col_inv_root"]
                preconditioned = _restore_shape(preconditioned, original_shape)

                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg / denom + preconditioned
                param.add_(update, alpha=-lr)

        return loss
