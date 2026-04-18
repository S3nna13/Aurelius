"""Signum optimizer — sign of momentum with optional gradient-norm scaling.

Reference: "Signum: The Sharp, Invariant Gradient Method"
           Balles & Hennig, ICML 2018.
           Also related to: signSGD (arXiv:1802.04434).

Key distinction from signSGD / MSignSGD (src/training/sign_sgd.py):
  - signSGD with momentum takes sign(momentum-smoothed gradient), but the
    motivation is purely 1-bit compression.
  - Signum (this implementation) is derived as the *natural gradient descent*
    variant that uses sign(momentum) and, optionally, normalises the effective
    learning rate by the gradient L2 norm for scale invariance (Signum-N).

Algorithm
---------
Given parameters θ, learning rate α, momentum coefficient β ∈ [0, 1):

  At each step t:
    g_t  ← ∇L(θ_{t-1})                               (raw gradient)
    m_t  = β · m_{t-1} + (1 − β) · g_t               (EMA momentum)
    If norm_scaling:
        α_eff = α / (||g_t|| + ε)                     (scale-invariant lr)
    Else:
        α_eff = α
    θ_t  = θ_{t-1} − α_eff · sign(m_t)               (sign-of-momentum step)

When β = 0 the EMA degenerates to m_t = g_t, giving plain signSGD
(sign of raw gradient), matching Algorithm 1 of arXiv:1802.04434.

Weight decay
------------
Applied as L2 regularisation by adding λ·θ to the gradient before the
momentum update:  g̃_t = g_t + λ·θ_{t-1}.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.optim import Optimizer


class Signum(Optimizer):
    """Signum: sign-of-momentum update with optional gradient-norm scaling.

    This is a *distinct* optimizer from MSignSGD (src/training/sign_sgd.py).
    The key difference is that Signum is derived as the sharp invariant gradient
    method, where the optional ``norm_scaling`` flag gives learning-rate
    invariance to the gradient magnitude (Signum-N variant from the paper).

    Args:
        params:        Iterable of parameters or parameter groups.
        lr:            Base learning rate α.  Default: 1e-3.
        momentum:      EMA coefficient β ∈ [0, 1) for the momentum buffer
                       m_t.  When 0, no state is stored and the update
                       degenerates to plain signSGD (sign of raw gradient).
                       Default: 0.9.
        weight_decay:  L2 regularisation coefficient λ.  The raw gradient is
                       augmented as g̃_t = g_t + λ·θ before the momentum
                       update.  Default: 0.0.
        norm_scaling:  If True, scale the effective learning rate by
                       1 / (||g_t||_2 + ε) before the sign step, giving
                       Signum-N — invariance to gradient magnitude rescaling.
                       Default: False.
        eps:           Small constant ε added to the gradient norm when
                       ``norm_scaling=True`` to avoid division by zero.
                       Default: 1e-8.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        norm_scaling: bool = False,
        eps: float = 1e-8,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            norm_scaling=norm_scaling,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        """Perform a single Signum optimisation step.

        Args:
            closure: Optional callable that re-evaluates the model and returns
                the loss.  Executed inside ``torch.enable_grad()``.

        Returns:
            Loss returned by ``closure``, or ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta: float = group["momentum"]
            weight_decay: float = group["weight_decay"]
            norm_scaling: bool = group["norm_scaling"]
            eps: float = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g_t = p.grad

                # ---------------------------------------------------------------- #
                # Weight decay: augment gradient with L2 penalty                   #
                # g̃_t = g_t + λ · θ_{t-1}                                          #
                # ---------------------------------------------------------------- #
                if weight_decay != 0.0:
                    g_t = g_t.add(p, alpha=weight_decay)

                # ---------------------------------------------------------------- #
                # Optional gradient-norm scaling (Signum-N)                        #
                # α_eff = α / (||g_t||_2 + ε)                                      #
                # ---------------------------------------------------------------- #
                if norm_scaling:
                    g_norm = g_t.norm(p=2).item()
                    lr_eff = lr / (g_norm + eps)
                else:
                    lr_eff = lr

                # ---------------------------------------------------------------- #
                # Momentum update + sign step                                       #
                # ---------------------------------------------------------------- #
                if beta == 0.0:
                    # Degenerate case: m_t = g_t → sign(m_t) = sign(g_t).
                    # Equivalent to plain signSGD with no state.
                    p.add_(g_t.sign(), alpha=-lr_eff)
                else:
                    state = self.state[p]
                    if len(state) == 0:
                        # Initialise momentum buffer: m_0 = g_1 (first gradient).
                        state["exp_avg"] = g_t.clone()
                    else:
                        # m_t = β · m_{t-1} + (1 − β) · g_t
                        state["exp_avg"].mul_(beta).add_(g_t, alpha=1.0 - beta)

                    # θ_t = θ_{t-1} − α_eff · sign(m_t)
                    p.add_(state["exp_avg"].sign(), alpha=-lr_eff)

        return loss
