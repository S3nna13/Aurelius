"""Rectified Adam (RAdam) optimizer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.optim import Optimizer


@dataclass(frozen=True)
class RAdamConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


def radam_rho_inf(beta2: float) -> float:
    return 2.0 / (1.0 - beta2) - 1.0


def radam_rho_t(step: int, beta2: float) -> float:
    if step <= 0:
        raise ValueError("step must be positive")
    beta2_t = beta2**step
    rho_inf = radam_rho_inf(beta2)
    return rho_inf - (2.0 * step * beta2_t) / (1.0 - beta2_t)


def radam_rectification(step: int, beta2: float) -> float:
    rho_t = radam_rho_t(step, beta2)
    if rho_t <= 4.0:
        return 0.0
    rho_inf = radam_rho_inf(beta2)
    numerator = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf
    denominator = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t
    return math.sqrt(numerator / denominator)


class RAdam(Optimizer):
    """Adam with variance rectification during the warmup phase."""

    def __init__(self, params, **kwargs) -> None:
        cfg = RAdamConfig(**kwargs)
        if cfg.lr <= 0:
            raise ValueError("lr must be positive")
        beta1, beta2 = cfg.betas
        if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
            raise ValueError("betas must be in (0, 1)")
        defaults = dict(lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                rho_t = radam_rho_t(step, beta2)
                state["rho_t"] = rho_t

                bias_correction1 = 1.0 - beta1**step
                if rho_t > 4.0:
                    rectification = radam_rectification(step, beta2)
                    state["rectification"] = rectification
                    step_size = lr * rectification / bias_correction1
                    denom = exp_avg_sq.sqrt().add(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    state["rectification"] = 0.0
                    step_size = lr / bias_correction1
                    p.add_(exp_avg, alpha=-step_size)

        return loss
