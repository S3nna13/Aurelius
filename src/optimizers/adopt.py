"""ADOPT optimizer — Modified Adam with optimal O(1/sqrt(T)) convergence.

Reference: "ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate"
           arXiv:2411.02853

Key insight: standard Adam's convergence issue arises from the correlation between
the gradient g_t and the denominator sqrt(v_t) when both use the same g_t.
ADOPT breaks this correlation by normalising the gradient with the *previous* step's
second moment before updating the first moment.

Algorithm (paper notation):
  Initialise: m_0 = 0, v_0 = 0

  At each step t (1-indexed):
    g_t  ← gradient of θ_{t-1}

    Special case t = 1 (no v_{t-1} yet):
      θ_1 = θ_0 − α · g_1
      v_1 = (1 − β2) · g_1²   (initialise second moment)
      continue

    g̃_t  = g_t / max(sqrt(v_{t-1}), ε)          ← normalise with previous denom
    m_t   = β1 · m_{t-1} + (1 − β1) · g̃_t       ← first moment of normalised grad
    v_t   = β2 · v_{t-1} + (1 − β2) · g_t²       ← second moment of raw grad
    θ_t   = θ_{t-1} − α · m_t                    ← parameter update (no /sqrt(v))
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class ADOPT(Optimizer):
    """ADOPT: Modified Adam that converges with any β2 at the optimal rate.

    Args:
        params:        Iterable of parameters or parameter groups.
        lr:            Learning rate α. Default: 1e-3.
        betas:         Coefficients (β1, β2) for running averages of gradient
                       and squared gradient.  Default: (0.9, 0.9999).
        eps:           ε — small constant for numerical stability.  Default: 1e-6.
        weight_decay:  L2 penalty coefficient.  Default: 0.0.
        decoupled:     If True, apply AdamW-style *decoupled* weight decay
                       (multiply params by (1 − α·λ) before the gradient step)
                       instead of adding λ·p to the gradient.  Default: False.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.9999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        decoupled: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decoupled=decoupled,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single ADOPT optimisation step.

        Args:
            closure: Optional callable that re-evaluates the model and returns the loss.

        Returns:
            Loss value if a closure was supplied, else None.
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
            decoupled: bool = group["decoupled"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # g_t — raw gradient (detached, flat copy of the grad tensor)
                g_t = p.grad.detach()

                state = self.state[p]

                # ------------------------------------------------------------------ #
                # State initialisation                                                #
                # ------------------------------------------------------------------ #
                if len(state) == 0:
                    state["step"] = 0
                    # m_0 = 0  (first moment of *normalised* gradient)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v_0 = 0  (second moment of raw gradient)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                t: int = state["step"]

                m: torch.Tensor = state["exp_avg"]       # m_{t-1}
                v: torch.Tensor = state["exp_avg_sq"]    # v_{t-1}

                # ------------------------------------------------------------------ #
                # Coupled weight decay (L2 regularisation via gradient augmentation) #
                # Applied before update so wd is coupled to Adam denominator in      #
                # standard Adam; here we apply it as a gradient add-on when          #
                # decoupled=False.                                                    #
                # ------------------------------------------------------------------ #
                if weight_decay != 0.0 and not decoupled:
                    g_t = g_t.add(p, alpha=weight_decay)

                # ------------------------------------------------------------------ #
                # Special case: t = 1                                                 #
                # v_{t-1} is zero — we cannot normalise yet.  Do a pure gradient     #
                # step and initialise v_1.                                            #
                # ------------------------------------------------------------------ #
                if t == 1:
                    # Decoupled weight decay applied multiplicatively before the step
                    if weight_decay != 0.0 and decoupled:
                        p.mul_(1.0 - lr * weight_decay)

                    # θ_1 = θ_0 − α · g_1
                    p.add_(g_t, alpha=-lr)

                    # Initialise v_1 = (1 − β2) · g_1²
                    v.addcmul_(g_t, g_t, value=1.0 - beta2)
                    continue

                # ------------------------------------------------------------------ #
                # General step t ≥ 2                                                  #
                # ------------------------------------------------------------------ #
                # g̃_t = g_t / max(sqrt(v_{t-1}), ε)
                denom = v.sqrt().clamp_(min=eps)   # max(sqrt(v_{t-1}), ε)
                g_tilde = g_t / denom              # g̃_t

                # m_t = β1 · m_{t-1} + (1 − β1) · g̃_t
                m.mul_(beta1).add_(g_tilde, alpha=1.0 - beta1)

                # v_t = β2 · v_{t-1} + (1 − β2) · g_t²
                v.mul_(beta2).addcmul_(g_t, g_t, value=1.0 - beta2)

                # Decoupled weight decay: p ← p · (1 − α · λ)
                if weight_decay != 0.0 and decoupled:
                    p.mul_(1.0 - lr * weight_decay)

                # θ_t = θ_{t-1} − α · m_t
                p.add_(m, alpha=-lr)

        return loss
