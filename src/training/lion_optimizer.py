"""Lion optimizer: EvoLved Sign Momentum.

Lion is a sign-based optimizer discovered via evolutionary search. It uses
two separate momentum coefficients: ``beta1`` for the interpolated gradient
used in the sign update, and ``beta2`` for the momentum buffer state update.
Only the sign of the interpolated momentum drives the update, which makes
the step magnitude uniform across parameters (and typically requires an LR
roughly 10x smaller than AdamW).

Update rule per step::

    c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    theta_t = theta_{t-1} - lr * (sign(c_t) + weight_decay * theta_{t-1})
    m_t = beta2 * m_{t-1} + (1 - beta2) * g_t

Reference: Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms",
arXiv:2302.06675.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion (EvoLved Sign Momentum) optimizer.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 1e-4). Typically ~10x smaller than AdamW.
        betas: ``(beta1, beta2)`` where ``beta1`` controls the momentum
            interpolation for the current update and ``beta2`` controls the
            momentum buffer state update (default: (0.9, 0.99)).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).
        maximize: If True, maximize the objective instead of minimizing
            (default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        """Perform a single optimization step.

        Args:
            closure: A closure that re-evaluates the model and returns the loss.

        Returns:
            The loss returned by ``closure``, or ``None`` if ``closure`` is None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            maximize = group.get("maximize", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if maximize:
                    grad = -grad

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]

                # c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                # Compute update direction from interpolated momentum, then take sign.
                update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1).sign_()

                # theta_t = theta - lr * (sign(c_t) + weight_decay * theta)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr)

                # m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
