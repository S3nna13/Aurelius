"""Adan (Adaptive Nesterov Momentum) optimizer.

Incorporates Nesterov momentum through estimates of three terms:
(1) the current gradient, (2) the gradient change (diff), and
(3) a second-moment term for adaptive learning rates.

Reference: Xie et al. 2022, arXiv:2208.06677
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import Optimizer


class Adan(Optimizer):
    """Adan optimizer: Adaptive Nesterov Momentum.

    Maintains three moment estimates per parameter:
    - exp_avg (m):      first moment (EMA of gradients)
    - exp_avg_diff (v): Nesterov moment (EMA of gradient + scaled gradient diff)
    - exp_avg_sq (n):   second moment (EMA of squared Nesterov estimate)

    Args:
        params:       Model parameters or param groups.
        lr:           Learning rate. Default: 1e-3.
        betas:        (β1, β2, β3) momentum coefficients. Default: (0.98, 0.92, 0.99).
        eps:          Numerical stability term. Default: 1e-8.
        weight_decay: Weight decay coefficient. Default: 0.02.
        no_prox:      If True, use L2 weight decay (applied to gradient) instead of
                      the proximal update. Default: False.
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
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"Invalid beta parameters: {betas}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            no_prox=no_prox,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        """Perform a single Adan optimization step.

        Args:
            closure: Optional callable that re-evaluates the model and returns the loss.

        Returns:
            Loss value if closure was provided, else None.
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

                g = p.grad

                state = self.state[p]

                # Initialise state on first step
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_diff"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["previous_grad"] = torch.zeros_like(p)

                step = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_diff"]
                n = state["exp_avg_sq"]
                prev_g = state["previous_grad"]

                state["step"] = step + 1

                # Gradient difference (zero on first step — previous_grad initialised to 0)
                diff = g - prev_g

                # Nesterov gradient estimate used for v and n updates
                nesterov_g = g + (1.0 - beta3) * diff

                # First moment update: m_k = (1-β1)*m_{k-1} + β1*g_k
                m.mul_(1.0 - beta1).add_(g, alpha=beta1)

                # Nesterov moment update: v_k = (1-β2)*v_{k-1} + β2*(g + (1-β3)*diff)
                v.mul_(1.0 - beta2).add_(nesterov_g, alpha=beta2)

                # Second moment update: n_k = (1-β3)*n_{k-1} + β3*(g + (1-β3)*diff)²
                n.mul_(1.0 - beta3).addcmul_(nesterov_g, nesterov_g, value=beta3)

                # Bias corrections
                step_val = float(state["step"])
                bias_c1 = 1.0 - (1.0 - beta1) ** step_val
                bias_c2 = 1.0 - (1.0 - beta2) ** step_val
                bias_c3 = 1.0 - (1.0 - beta3) ** step_val

                m_hat = m / bias_c1
                v_hat = v / bias_c2
                n_hat = n / bias_c3

                # Adaptive step size: η = lr / (sqrt(n_hat) + eps)
                denom = n_hat.sqrt().add_(eps)

                # Compute update direction: m_hat + (1-β3)*v_hat
                update = m_hat.add(v_hat, alpha=1.0 - beta3)

                if no_prox:
                    # L2 weight decay: add wd*p to the gradient before applying
                    update.add_(p, alpha=wd)
                    p.addcdiv_(update, denom, value=-lr)
                else:
                    # Proximal (decoupled) update:
                    # θ_{k+1} = θ_k / (1 + λ*lr) - η * (m_hat + (1-β3)*v_hat)
                    p.div_(1.0 + wd * lr)
                    p.addcdiv_(update, denom, value=-lr)

                # Store current gradient for next step's diff computation
                state["previous_grad"] = g.clone()

        return loss

    def get_lr(self) -> list[float]:
        """Return current learning rates for all param groups."""
        return [group["lr"] for group in self.param_groups]

    def restart_opt(self) -> None:
        """Reset step count and all moment buffers to zero.

        Useful for restart strategies (e.g., warm restarts, loss-spike recovery).
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    continue
                state["step"] = 0
                state["exp_avg"].zero_()
                state["exp_avg_diff"].zero_()
                state["exp_avg_sq"].zero_()
                state["previous_grad"].zero_()


@dataclass
class AdanConfig:
    """Configuration dataclass for the Adan optimizer.

    Attributes:
        lr:           Learning rate. Default: 1e-3.
        betas:        (β1, β2, β3) momentum coefficients. Default: (0.98, 0.92, 0.99).
        eps:          Numerical stability term. Default: 1e-8.
        weight_decay: Weight decay coefficient. Default: 0.02.
        no_prox:      Use L2 weight decay rather than proximal update. Default: False.
    """

    lr: float = 1e-3
    betas: tuple[float, float, float] = (0.98, 0.92, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.02
    no_prox: bool = False
