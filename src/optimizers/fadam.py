"""FAdam optimizer — Adam as a natural gradient optimizer.

Reference: "FAdam: Adam is a natural gradient optimizer using diagonal
empirical Fisher information" (arXiv:2405.14429).

Variable notation matches the paper:
  m_t  — first moment (momentum)
  F_t  — diagonal empirical Fisher (estimated from m_t, NOT g_t)
  m̂_t  — bias-corrected first moment
  F̂_t  — bias-corrected Fisher diagonal
  α    — learning rate
  β1   — first-moment decay coefficient
  β2   — Fisher decay coefficient
  ε    — numerical stability constant
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class FAdam(Optimizer):
    """FAdam: Adam as a natural gradient optimizer.

    The key difference from standard Adam: the Fisher information matrix
    F_t is estimated from the momentum m_t (i.e., F_t += β2*F_{t-1} +
    (1-β2)*m_t^2) rather than from the raw gradient g_t^2.  This removes
    a cross-term bias and provides a theoretically-grounded natural
    gradient interpretation.

    Update rule (paper notation):
        m_t = β1 * m_{t-1} + (1-β1) * g_t
        F_t = β2 * F_{t-1} + (1-β2) * m_t^2
        m̂_t = m_t / (1 - β1^t)
        F̂_t = F_t / (1 - β2^t)
        θ_t = θ_{t-1} - α * m̂_t / (sqrt(F̂_t) + ε)

    Args:
        params: Iterable of parameters or param groups.
        lr (α): Learning rate. Default: 1e-3.
        betas (β1, β2): Decay coefficients for the first moment and
            Fisher diagonal respectively. Default: (0.9, 0.999).
        eps (ε): Term added to the denominator for numerical stability.
            Default: 1e-8.
        weight_decay: Decoupled weight-decay coefficient (AdamW-style).
            Applied as θ ← θ*(1 - lr*λ) before the gradient step.
            Default: 0.0 (disabled).
        clip_value: If > 0, raw gradients are clipped element-wise to
            [-clip_value, clip_value] before the update. Default: 1.0.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_value: float = 1.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if not 0.0 < beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 < beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if clip_value < 0.0:
            raise ValueError(f"Invalid clip_value: {clip_value}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clip_value=clip_value,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that re-evaluates the model and returns the
                loss (optional).

        Returns:
            The loss returned by the closure, or ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1: float
            beta2: float
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            clip_value: float = group["clip_value"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # ------------------------------------------------------------------ #
                # Raw gradient g_t                                                    #
                # ------------------------------------------------------------------ #
                g_t = p.grad.detach()

                # Optional gradient clipping (element-wise)
                if clip_value > 0.0:
                    g_t = g_t.clamp(-clip_value, clip_value)

                # Optional decoupled weight decay: θ ← θ * (1 - α*λ)
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # ------------------------------------------------------------------ #
                # Initialise state                                                     #
                # ------------------------------------------------------------------ #
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # m_t: first moment — same shape as parameter
                    state["exp_avg"] = torch.zeros_like(p)
                    # F_t: diagonal Fisher — same shape as parameter
                    state["fisher_diag"] = torch.zeros_like(p)

                state["step"] += 1
                t: int = state["step"]

                m_t = state["exp_avg"]      # in-place reference (paper: m_t)
                F_t = state["fisher_diag"]  # in-place reference (paper: F_t)

                # ------------------------------------------------------------------ #
                # Step 1: update first moment                                         #
                #   m_t = β1 * m_{t-1} + (1-β1) * g_t                               #
                # ------------------------------------------------------------------ #
                m_t.mul_(beta1).add_(g_t, alpha=1.0 - beta1)

                # ------------------------------------------------------------------ #
                # Step 2: update Fisher diagonal from MOMENTUM (not from g_t!)        #
                #   F_t = β2 * F_{t-1} + (1-β2) * m_t^2                              #
                # ------------------------------------------------------------------ #
                F_t.mul_(beta2).addcmul_(m_t, m_t, value=1.0 - beta2)

                # ------------------------------------------------------------------ #
                # Steps 3-4: bias-corrected estimates                                 #
                #   m̂_t = m_t / (1 - β1^t)                                          #
                #   F̂_t = F_t / (1 - β2^t)                                          #
                # ------------------------------------------------------------------ #
                bias_correction1: float = 1.0 - beta1 ** t   # (1 - β1^t)
                bias_correction2: float = 1.0 - beta2 ** t   # (1 - β2^t)

                m_hat = m_t / bias_correction1   # m̂_t
                F_hat = F_t / bias_correction2   # F̂_t

                # ------------------------------------------------------------------ #
                # Step 5: natural gradient update                                     #
                #   θ_t = θ_{t-1} - α * m̂_t / (sqrt(F̂_t) + ε)                     #
                # ------------------------------------------------------------------ #
                denom = F_hat.sqrt().add_(eps)  # sqrt(F̂_t) + ε
                p.addcdiv_(m_hat, denom, value=-lr)

        return loss
