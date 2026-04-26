"""MARS optimizer: Momentum-based Adaptive Recent-gradient Scaling.

Bridges the gap between Adam and SGDM by using a variance-reduced gradient
(c_t) computed as a moving correction of the current and last gradients,
then applying an Adam-style adaptive update on top of that corrected gradient.

Key insight: cubic regularization + variance reduction yields better convergence
than either pure Adam or pure SGDM in both convex and non-convex settings.

Reference: Yuan & Gao 2024, "MARS: Unleashing the Power of Variance Reduction
for Training Large Models", arXiv:2411.10438
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class MARSOptimizer(Optimizer):
    """MARS (Momentum-based Adaptive Recent-gradient Scaling) optimizer.

    Computes a variance-reduced gradient estimate c_t at each step:

        c_t = grad_t - gamma * (grad_t - grad_{t-1})    # on first step: c_t = grad_t

    Then applies an Adam-style update using c_t instead of the raw gradient.
    Three update rules are supported via `mars_type`:

    - 'mars-adamw': Adam-style adaptive scaling with decoupled weight decay
    - 'mars-lion':  Sign-based update (Lion-style) using variance-reduced gradient
    - 'mars-shampoo': Simplified diagonal Shampoo-style scaling

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running averages of gradient and its
            square (default: (0.9, 0.99)).
        eps: Term added to denominator for numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 0.0).
        gamma: Variance-reduction coefficient. Controls how much the correction
            term (grad - last_grad) is subtracted. gamma=0 recovers standard
            Adam (no variance reduction). (default: 0.025)
        mars_type: Update rule variant — 'mars-adamw', 'mars-lion', or
            'mars-shampoo' (default: 'mars-adamw').
    """

    VALID_TYPES = ("mars-adamw", "mars-lion", "mars-shampoo")

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        gamma: float = 0.025,
        mars_type: str = "mars-adamw",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if mars_type not in self.VALID_TYPES:
            raise ValueError(f"mars_type must be one of {self.VALID_TYPES}, got '{mars_type}'")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            gamma=gamma,
            mars_type=mars_type,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_lr(self) -> list[float]:
        """Return the current learning rate for each parameter group.

        Returns:
            List of learning rates, one per parameter group.
        """
        return [group["lr"] for group in self.param_groups]

    def zero_variance_reduction(self) -> None:
        """Reset last_grad state to None for all tracked parameters.

        Call this after a significant distribution shift (e.g. resuming from a
        checkpoint, changing the dataset) to prevent stale gradient corrections
        from corrupting the variance-reduction term.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "last_grad" in state:
                    state["last_grad"] = None

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        """Perform a single MARS optimization step.

        Args:
            closure: Optional closure that re-evaluates the model and returns
                the loss.

        Returns:
            Loss value if closure was provided, else None.
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
            mars_type = group["mars_type"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Initialize state on first step
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["last_grad"] = None

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                last_grad: torch.Tensor | None = state["last_grad"]

                # ----------------------------------------------------------
                # 1. Compute variance-reduced gradient c_t
                #    c_t = grad - gamma * (grad - last_grad)
                #         = (1 - gamma) * grad + gamma * last_grad
                # On first step last_grad is None, so c_t = grad (no correction).
                # ----------------------------------------------------------
                if last_grad is not None:
                    c_t = grad - gamma * (grad - last_grad)
                else:
                    c_t = grad.clone()

                # Store current grad for next step's variance reduction
                state["last_grad"] = grad.clone()

                # ----------------------------------------------------------
                # 2. Apply decoupled weight decay (AdamW style)
                # ----------------------------------------------------------
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # ----------------------------------------------------------
                # 3. Update first and second moments using variance-reduced c_t
                # ----------------------------------------------------------
                if mars_type == "mars-adamw":
                    # Standard Adam moment updates on c_t
                    exp_avg.mul_(beta1).add_(c_t, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t, value=1.0 - beta2)

                    # Bias correction
                    bias_correction1 = 1.0 - beta1**step
                    bias_correction2 = 1.0 - beta2**step

                    # Adam update: m_hat / (sqrt(v_hat) + eps)
                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

                elif mars_type == "mars-lion":
                    # Lion-style: sign of interpolated momentum, using c_t
                    # update = sign(beta1 * m + (1 - beta1) * c_t)
                    update = (beta1 * exp_avg + (1.0 - beta1) * c_t).sign_()
                    p.add_(update, alpha=-lr)
                    # Momentum update with c_t
                    exp_avg.mul_(beta2).add_(c_t, alpha=1.0 - beta2)
                    # exp_avg_sq not used in lion variant but keep zeroed
                    # (no-op to preserve state consistency)

                elif mars_type == "mars-shampoo":
                    # Simplified diagonal Shampoo: scale by per-element
                    # running norm (similar to RMSProp but with momentum)
                    exp_avg.mul_(beta1).add_(c_t, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t, value=1.0 - beta2)

                    # Diagonal preconditioner: 1 / sqrt(v + eps)
                    # No bias correction for Shampoo variant (common practice)
                    denom = exp_avg_sq.sqrt().add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-lr)

        return loss
