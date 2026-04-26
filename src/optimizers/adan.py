"""
src/optimizers/adan.py

Adan: adaptive Nesterov momentum optimizer with three gradient moments.
More stable than Adam for transformer training with large batch sizes.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class Adan(Optimizer):
    """Adan optimizer: Adaptive Nesterov Momentum Algorithm.

    Uses three gradient moments for more stable convergence:
      - m1: EMA of gradients
      - m2: EMA of gradient differences (Nesterov term)
      - m3: EMA of squared (grad + beta2 * diff)

    Args:
        params:       Iterable of parameters or parameter groups.
        lr:           Learning rate. Default: 1e-3.
        betas:        Coefficients (beta1, beta2, beta3) for the three moments.
                      Default: (0.98, 0.92, 0.99).
        eps:          Small constant for numerical stability. Default: 1e-8.
        weight_decay: Decoupled weight decay coefficient. Default: 0.02.
        no_prox:      If True, apply weight decay directly (multiply param by
                      1/(1 + lr * wd)) instead of proximal update. Default: False.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
        no_prox: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, no_prox=no_prox)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: Optional callable that re-evaluates the model and returns loss.

        Returns:
            loss (optional): Value returned by closure if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            no_prox = group["no_prox"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)  # m1
                    state["exp_avg_diff"] = torch.zeros_like(p)  # m2
                    state["exp_avg_sq"] = torch.zeros_like(p)  # m3
                    state["prev_grad"] = grad.clone()

                m1 = state["exp_avg"]
                m2 = state["exp_avg_diff"]
                m3 = state["exp_avg_sq"]
                prev_grad = state["prev_grad"]

                state["step"] += 1

                # Gradient difference (Nesterov term)
                if state["step"] == 1:
                    diff = torch.zeros_like(grad)
                else:
                    diff = grad - prev_grad

                # Update three moments
                m1.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                m2.mul_(beta2).add_(diff, alpha=1.0 - beta2)
                # Combined term for m3: (grad + beta2 * diff)^2
                combined = grad + beta2 * diff
                m3.mul_(beta3).addcmul_(combined, combined, value=1.0 - beta3)

                # Parameter update
                denom = m3.sqrt().add_(eps)
                update = (m1 + beta2 * m2) / denom
                p.add_(update, alpha=-lr)

                # Weight decay
                if wd != 0.0:
                    if no_prox:
                        p.mul_(1.0 / (1.0 + lr * wd))
                    else:
                        p.add_(p, alpha=-lr * wd)

                # Save current grad for next step's difference computation
                state["prev_grad"] = grad.clone()

        return loss
