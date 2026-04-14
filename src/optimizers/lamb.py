"""Layer-wise Adaptive Moments (LAMB) optimizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import Optimizer


@dataclass(frozen=True)
class LAMBConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-6
    weight_decay: float = 0.0
    trust_clip: float = 10.0


def lamb_trust_ratio(
    param: torch.Tensor,
    update: torch.Tensor,
    eps: float = 1e-6,
    trust_clip: float | None = None,
) -> float:
    """Compute the LAMB trust ratio for one parameter tensor."""

    param_norm = param.norm().item()
    update_norm = update.norm().item()
    if param_norm == 0.0 or update_norm == 0.0:
        return 1.0
    ratio = param_norm / (update_norm + eps)
    if trust_clip is not None:
        ratio = min(ratio, trust_clip)
    return float(ratio)


class LAMB(Optimizer):
    """Adam with layer-wise trust ratio scaling."""

    def __init__(self, params, **kwargs) -> None:
        cfg = LAMBConfig(**kwargs)
        if cfg.lr <= 0:
            raise ValueError("lr must be positive")
        beta1, beta2 = cfg.betas
        if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
            raise ValueError("betas must be in (0, 1)")
        defaults = dict(
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            trust_clip=cfg.trust_clip,
        )
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
            trust_clip = group["trust_clip"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
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

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                adam_step = (exp_avg / bias_correction1) / (
                    exp_avg_sq / bias_correction2
                ).sqrt().add(eps)

                if weight_decay != 0.0:
                    adam_step = adam_step.add(p, alpha=weight_decay)

                trust_ratio = lamb_trust_ratio(p, adam_step, eps=eps, trust_clip=trust_clip)
                state["trust_ratio"] = trust_ratio
                state["param_norm"] = p.norm().item()
                state["update_norm"] = adam_step.norm().item()

                p.add_(adam_step, alpha=-lr * trust_ratio)

        return loss
