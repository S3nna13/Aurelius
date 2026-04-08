"""Lion optimizer: Sign of momentum for memory-efficient optimization.

Uses sign of EMA-interpolated gradient rather than gradient magnitude.
Only one momentum buffer vs Adam's two — more memory efficient.

Reference: Chen et al. 2023, arXiv:2302.06675
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion optimizer.

    Args:
        params: Model parameters
        lr: Learning rate (typically 1e-4 to 3e-4, ~3-10x smaller than Adam lr)
        betas: (beta1, beta2) — (0.9, 0.99) recommended
        weight_decay: L2 regularization coefficient
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        """Perform one Lion optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                # Get/initialize momentum buffer
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]

                # Compute update direction: sign of β1*m + (1-β1)*g
                update = (beta1 * m + (1 - beta1) * g).sign_()

                # Apply weight decay + sign update
                p.add_(update + wd * p, alpha=-lr)

                # Update momentum: β2*m + (1-β2)*g
                m.mul_(beta2).add_(g, alpha=1 - beta2)

        return loss
