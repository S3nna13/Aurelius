"""Adafactor optimizer.

Implements factorized second-moment estimation for matrix parameters and a
full second-moment fallback for vectors and scalars.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import Optimizer


@dataclass(frozen=True)
class AdafactorConfig:
    lr: float = 1e-2
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    clip_threshold: float = 1.0
    factorize: bool = True


def adafactor_factorized_second_moment(grad_sq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row and column statistics for a 2D squared gradient."""

    if grad_sq.ndim != 2:
        raise ValueError("factorized second moment requires a 2D tensor")
    return grad_sq.mean(dim=1), grad_sq.mean(dim=0)


def adafactor_reconstruct_second_moment(
    row_stats: torch.Tensor,
    col_stats: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reconstruct a dense second-moment estimate from factorized statistics."""

    denom = col_stats.mean().clamp_min(eps)
    return torch.outer(row_stats, col_stats) / denom


class Adafactor(Optimizer):
    """Memory-efficient adaptive optimizer with factored second moments."""

    def __init__(self, params, **kwargs) -> None:
        cfg = AdafactorConfig(**kwargs)
        if cfg.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 < cfg.beta2 < 1.0:
            raise ValueError("beta2 must be in (0, 1)")
        if cfg.clip_threshold <= 0:
            raise ValueError("clip_threshold must be positive")
        defaults = dict(
            lr=cfg.lr,
            beta2=cfg.beta2,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            clip_threshold=cfg.clip_threshold,
            factorize=cfg.factorize,
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
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            clip_threshold = group["clip_threshold"]
            factorize = group["factorize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if factorize and p.ndim == 2:
                        state["exp_avg_sq_row"] = torch.zeros(
                            p.size(0), device=p.device, dtype=p.dtype
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            p.size(1), device=p.device, dtype=p.dtype
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1

                if "exp_avg_sq_row" in state:
                    grad_sq = grad.square().add(eps)
                    row_stats, col_stats = adafactor_factorized_second_moment(grad_sq)
                    state["exp_avg_sq_row"].mul_(beta2).add_(row_stats, alpha=1.0 - beta2)
                    state["exp_avg_sq_col"].mul_(beta2).add_(col_stats, alpha=1.0 - beta2)
                    second_moment = adafactor_reconstruct_second_moment(
                        state["exp_avg_sq_row"],
                        state["exp_avg_sq_col"],
                        eps=eps,
                    )
                else:
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    second_moment = state["exp_avg_sq"]

                denom = second_moment.sqrt().add(eps)
                update = grad / denom
                update_rms = update.square().mean().sqrt().clamp_min(eps)
                if update_rms > clip_threshold:
                    update = update * (clip_threshold / update_rms)

                p.add_(update, alpha=-lr)

        return loss
