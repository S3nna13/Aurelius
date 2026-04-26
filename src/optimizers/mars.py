"""
src/optimizers/mars.py

MARS: Make vAriance Reduction Shine (Yuan et al., 2024, arXiv:2411.10438).

Variance-reduced gradient estimator combined with AdamW-style adaptive moments.
The core idea is a STORM/Spider-style correction that subtracts a scaled
difference between consecutive gradients:

    g_tilde_k = g_k + gamma * (g_k - g_{k-1}) / ||g_k - g_{k-1}|| * ||g_k||

and then feeds this variance-reduced estimator into an AdamW-style update:

    m_k = beta1 * m_{k-1} + (1 - beta1) * g_tilde_k
    v_k = beta2 * v_{k-1} + (1 - beta2) * g_tilde_k^2
    theta <- theta - lr * m_hat / (sqrt(v_hat) + eps) - lr * wd * theta
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class Mars(Optimizer):
    """MARS optimizer with variance-reduced gradient estimator.

    Args:
        params:       Iterable of parameters or parameter groups.
        lr:           Learning rate. Default: 3e-3.
        betas:        Coefficients (beta1, beta2) for the first/second moments.
                      Default: (0.95, 0.99).
        eps:          Small constant for numerical stability. Default: 1e-8.
        weight_decay: Decoupled weight decay coefficient. Default: 0.01.
        gamma:        Variance-reduction coefficient. Default: 0.025.
                      Setting gamma=0 recovers vanilla AdamW.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.95, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        gamma: float = 0.025,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not (isinstance(betas, (tuple, list)) and len(betas) == 2):
            raise ValueError(f"Invalid betas (must be length-2 tuple): {betas}")
        b1, b2 = betas
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid beta1: {b1}")
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"Invalid beta2: {b2}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma (must be >= 0): {gamma}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            gamma=gamma,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: Optional callable that re-evaluates the model and returns loss.

        Returns:
            loss value returned by ``closure`` if provided, else ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            gamma = group["gamma"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MARS does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # prev_grad=None signals the first step (no variance reduction).
                    state["prev_grad"] = None

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                prev_grad = state["prev_grad"]

                # Variance-reduced gradient estimator.
                if prev_grad is None or gamma == 0.0:
                    g_tilde = grad
                else:
                    diff = grad - prev_grad
                    diff_norm = diff.norm()
                    if diff_norm > 0:
                        grad_norm = grad.norm()
                        # g_tilde = g + gamma * (diff / ||diff||) * ||g||
                        g_tilde = grad + diff * (gamma * grad_norm / diff_norm)
                    else:
                        g_tilde = grad

                # Adam-style moment updates on the variance-reduced estimator.
                exp_avg.mul_(beta1).add_(g_tilde, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_tilde, g_tilde, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                update = m_hat / (v_hat.sqrt().add_(eps))
                p.add_(update, alpha=-lr)

                # Decoupled weight decay (applied after adaptive update).
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

                # Save current grad for next step's variance reduction.
                state["prev_grad"] = grad.detach().clone()

        return loss
