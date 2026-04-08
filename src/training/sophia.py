"""Sophia: Stochastic Second-order Optimizer for LLM pre-training.

Uses diagonal Hessian estimate (squared gradient) as preconditioner.
Clips the normalized gradient to [-gamma, gamma] for stability.

Reference: Liu et al. 2023, arXiv:2305.14342
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer


class Sophia(Optimizer):
    """Sophia optimizer with Gauss-Newton Hessian estimate.

    Args:
        params: Model parameters
        lr: Learning rate (typically 1e-3 to 3e-3 — larger than Adam)
        betas: (beta1, beta2) — EMA coefficients for gradient and Hessian
        rho: Clipping parameter for normalized gradient (default 0.04)
        weight_decay: Decoupled weight decay
        update_period: How often to update Hessian estimate (every k steps)
        eps: Denominator stability constant
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        rho: float = 0.04,
        weight_decay: float = 0.1,
        update_period: int = 10,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            update_period=update_period,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        """Perform one Sophia step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            wd = group["weight_decay"]
            k = group["update_period"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.float()
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["h"] = torch.ones_like(p, dtype=torch.float32)

                state["step"] += 1
                m, h = state["m"], state["h"]

                # Update first moment
                m.mul_(beta1).add_(g, alpha=1 - beta1)

                # Update diagonal Hessian estimate every k steps
                if state["step"] % k == 0:
                    h.mul_(beta2).addcmul_(g, g, value=1 - beta2)  # g² estimate

                # Normalized gradient clipped to [-rho, rho]
                normalized = m / h.clamp(min=eps)
                normalized.clamp_(-rho, rho)

                # Update: apply weight decay + normalized clipped step
                p.add_(normalized + wd * p.float(), alpha=-lr)

        return loss
