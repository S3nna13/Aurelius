"""
src/optimizers/adabelief_optimizer.py

AdaBelief (Zhuang et al., ICML 2020, arXiv:2010.07468).

Variant of Adam that uses the BELIEF in the observed gradient — i.e. the EMA of
(g - m)^2 instead of g^2 — making the step-size adapt to the "trustworthiness"
of the gradient direction.

Pure-PyTorch implementation.  No external dependencies.

Recurrence per parameter, step k:

    m_k     = beta1 * m_{k-1} + (1 - beta1) * g_k
    s_k     = beta2 * s_{k-1} + (1 - beta2) * (g_k - m_k)^2 + eps
    m_hat   = m_k / (1 - beta1^k)
    s_hat   = s_k / (1 - beta2^k)
    theta_k = theta_{k-1} - lr * m_hat / (sqrt(s_hat) + eps)
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer


class AdaBelief(Optimizer):
    """AdaBelief optimizer.

    Args:
        params:        Iterable of parameters or parameter groups.
        lr:            Learning rate.  Default: ``1e-3``.
        betas:         ``(beta1, beta2)``.  Default: ``(0.9, 0.999)``.
        eps:           Numerical stabiliser added inside the sqrt (also used
                       inside the ``s`` update to avoid division by zero).
                       Default: ``1e-16``.
        weight_decay:  Decoupled weight-decay coefficient.  Default: ``0.0``.
        amsgrad:       Whether to use the AMSGrad variant (keep max of ``s``).
                       Default: ``False``.
        fixed_decay:   If ``True``, apply weight decay directly to the
                       parameter before the gradient update (multiply by
                       ``1 - lr * weight_decay``).  Default: ``False``.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        fixed_decay: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError(f"Invalid betas (must be a 2-tuple): {betas}")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0):
            raise ValueError(f"Invalid beta1: {b1}")
        if not (0.0 <= b2 < 1.0):
            raise ValueError(f"Invalid beta2: {b2}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=(float(b1), float(b2)),
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            fixed_decay=fixed_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimisation step.

        Args:
            closure: Optional callable that re-evaluates the model and
                     returns the loss.

        Returns:
            The value returned by ``closure`` (if provided), else ``None``.
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
            amsgrad = group["amsgrad"]
            fixed_decay = group["fixed_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients.")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_var"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        state["max_exp_avg_var"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg = state["exp_avg"]
                exp_avg_var = state["exp_avg_var"]
                state["step"] += 1
                step = state["step"]

                # Bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                # Fixed-decay weight decay (applied before the gradient step)
                if wd != 0.0 and fixed_decay:
                    p.mul_(1.0 - lr * wd)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Gradient residual (g - m)
                grad_residual = grad - exp_avg

                # Update biased second moment estimate (belief)
                exp_avg_var.mul_(beta2).addcmul_(
                    grad_residual, grad_residual, value=1.0 - beta2
                )
                exp_avg_var.add_(eps)

                if amsgrad:
                    max_exp_avg_var = state["max_exp_avg_var"]
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    denom = (max_exp_avg_var / bias_correction2).sqrt().add_(eps)
                else:
                    denom = (exp_avg_var / bias_correction2).sqrt().add_(eps)

                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Decoupled weight decay (applied after the gradient step)
                if wd != 0.0 and not fixed_decay:
                    p.add_(p, alpha=-lr * wd)

        return loss
