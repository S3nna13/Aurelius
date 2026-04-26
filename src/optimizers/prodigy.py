"""
src/optimizers/prodigy.py

Prodigy optimizer (Mishchenko & Defazio 2023, arXiv:2306.06101).

Automatic learning rate adaptation via the D-adaptation principle. Estimates
the optimal step-size online so there is no need to tune the learning rate.
The AdamW-style variant is implemented here, matching the reference
implementation from the official Prodigy repository.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer


class Prodigy(Optimizer):
    """Prodigy optimizer with automatic learning rate adaptation.

    Args:
        params:            Iterable of parameters or parameter groups.
        lr:                External multiplier (``gamma``). Default: 1.0.
        betas:             Coefficients for running averages. Default: (0.9, 0.999).
        eps:               Small constant for numerical stability. Default: 1e-8.
        weight_decay:      Decoupled (AdamW-style) weight decay. Default: 0.0.
        d_coef:            Multiplier for the adaptive ``d`` estimate. Default: 1.0.
        growth_rate:       Maximum multiplicative growth of ``d`` per step.
                           ``inf`` disables the cap. Default: ``inf``.
        fsdp_in_use:       If True, perform an all-reduce over the numerator /
                           denominator to keep ``d`` synchronized across FSDP
                           ranks. Accepted as a flag; if no process group is
                           initialized this is a silent no-op. Default: False.
        safeguard_warmup:  If True, the running sum ``s`` is not updated with
                           the current ``d_k`` weighting, which keeps early
                           ``d`` estimates small during warm-up. Default: False.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        d_coef: float = 1.0,
        growth_rate: float = float("inf"),
        fsdp_in_use: bool = False,
        safeguard_warmup: bool = False,
    ) -> None:
        if lr is None or lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not (isinstance(betas, (tuple, list)) and len(betas) == 2):
            raise ValueError(f"Invalid betas (must be 2-tuple): {betas}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if d_coef <= 0.0:
            raise ValueError(f"Invalid d_coef: {d_coef}")
        if growth_rate <= 1.0:
            raise ValueError(f"Invalid growth_rate (must be > 1): {growth_rate}")

        # ``d0`` is the initial distance estimate. Small positive constant.
        d0 = 1e-6
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            d=d0,
            d0=d0,
            d_max=d0,
            d_numerator=0.0,
            d_coef=d_coef,
            growth_rate=growth_rate,
            fsdp_in_use=fsdp_in_use,
            safeguard_warmup=safeguard_warmup,
            k=0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # ``d`` is shared across all parameter groups (single scalar estimate).
        group0 = self.param_groups[0]
        d = group0["d"]
        d_max = group0["d_max"]
        d_coef = group0["d_coef"]
        growth_rate = group0["growth_rate"]
        fsdp_in_use = group0["fsdp_in_use"]
        safeguard_warmup = group0["safeguard_warmup"]
        beta1, beta2 = group0["betas"]
        k = group0["k"]

        # beta3 controls the decay of the running numerator / s estimates.
        # Default beta3 = sqrt(beta2), matching the reference implementation.
        beta3 = beta2**0.5

        # Running numerator (scalar) carried across calls.
        d_numerator = group0["d_numerator"] * beta3

        # Bias corrections from the original implementation.
        bias_correction = ((1.0 - beta2 ** (k + 1)) ** 0.5) / (1.0 - beta1 ** (k + 1))
        dlr = d * group0["lr"] * bias_correction

        # --- First pass: accumulate numerator and denominator contributions. ---
        d_denom = torch.zeros((), dtype=torch.float32)
        numerator_acc = torch.zeros((), dtype=torch.float32)

        for group in self.param_groups:
            wd = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Prodigy does not support sparse gradients.")

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["s"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["p0"] = p.detach().clone()
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                s = state["s"]
                p0 = state["p0"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Accumulate numerator: (d / d0) * dlr * <grad, p0 - p>
                # (matches the official Prodigy reference implementation).
                d0 = group0["d0"]
                numerator_acc = numerator_acc + (
                    (d / d0)
                    * dlr
                    * (p0.to(grad.dtype) - p.to(grad.dtype)).mul(grad).sum().float().cpu()
                )

                # Moments update (AdamW-style, following the official Prodigy
                # repo: exp_avg scaled by d, exp_avg_sq scaled by d^2 so that
                # the ratio ``exp_avg / sqrt(exp_avg_sq)`` stays dimensionless).
                exp_avg.mul_(beta1).add_(grad, alpha=d * (1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1.0 - beta2))

                # Running sum ``s`` used to form the D-adaptation denominator.
                d0 = group0["d0"]
                if safeguard_warmup:
                    s.mul_(beta3).add_(grad, alpha=(d / d0) * d)
                else:
                    s.mul_(beta3).add_(grad, alpha=(d / d0) * dlr)

                d_denom = d_denom + s.abs().sum().float().cpu()

        # All-reduce across FSDP ranks if requested and a pg exists.
        if fsdp_in_use:
            try:
                import torch.distributed as dist  # noqa: WPS433

                if dist.is_available() and dist.is_initialized():
                    t = (
                        torch.stack([numerator_acc, d_denom]).cuda()
                        if torch.cuda.is_available()
                        else torch.stack([numerator_acc, d_denom])
                    )
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    numerator_acc = t[0].cpu()
                    d_denom = t[1].cpu()
            except Exception:  # noqa: S110
                pass

        d_numerator = d_numerator + numerator_acc.item()

        if d_denom.item() > 0:
            d_hat = d_coef * d_numerator / d_denom.item()
            # On the very first step d grows up to d_hat if larger.
            if d == group0["d0"]:
                d = max(d, d_hat)
            # Cap growth by growth_rate * previous d.
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

        # --- Second pass: parameter update. ---
        for group in self.param_groups:
            wd = group["weight_decay"]
            eps = group["eps"]
            # Persist updated scalars back to every group to keep them in sync.
            group["d"] = d
            group["d_max"] = d_max
            group["d_numerator"] = d_numerator
            group["k"] = k + 1

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                denom = exp_avg_sq.sqrt().add_(d * eps)

                if wd != 0.0:
                    p.mul_(1.0 - dlr * wd)

                p.addcdiv_(exp_avg, denom, value=-dlr)

                state["step"] += 1

        return loss
