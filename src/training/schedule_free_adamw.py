"""Schedule-Free AdamW and SGD optimizers.

Implements Schedule-Free learning from Defazio et al. (arXiv:2405.15682):
"Schedule Free Learning -- A New Way to Train".

The core idea: eliminate the learning rate schedule by maintaining two iterate
sequences:
  z  -- the "lazy" gradient-update point (used during training)
  x  -- the Polyak-Ruppert running average of z (used for evaluation)

At each step, z is updated via the underlying rule (Adam or SGD), and the
running average x is advanced.  The model's .data tensors always hold z during
training.  For inference, call optimizer.switch_to_eval() to swap param.data
to x, then optimizer.train() to swap back.
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class ScheduleFreeSGD(Optimizer):
    """SGD with schedule-free iterate averaging (Defazio et al. 2024).

    Maintains:
        z  -- the SGD-updated sequence (stored in param.data during training)
        x  -- the Polyak-Ruppert running average of z (used for evaluation)

    Args:
        params: Iterable of parameters or param groups.
        lr: Base learning rate.
        momentum: SGD momentum coefficient (default 0.9).
        weight_decay: L2 regularisation applied to z (default 0.0).
        warmup_steps: Linear LR ramp over first N steps (default 0).
        r: Averaging exponent -- 0.0 = uniform Polyak-Ruppert,
           >0 = recent-biased (default 0.0).
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            r=r,
        )
        super().__init__(params, defaults)
        self._mode = "train"

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Switch parameters to z-sequence (training mode)."""
        if self._mode == "train":
            return
        self._swap_to("z")
        self._mode = "train"

    def switch_to_eval(self) -> None:
        """Switch parameters to x-sequence (evaluation / inference mode)."""
        if self._mode == "eval":
            return
        self._swap_to("x")
        self._mode = "eval"

    def _swap_to(self, target: str) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if not state:
                    continue
                if target == "x":
                    p.data.copy_(state["x"])
                else:
                    p.data.copy_(state["z"])

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # type: ignore[override]
        """Perform one schedule-free SGD step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            warmup_steps = group["warmup_steps"]
            r = group["r"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.clone()
                state = self.state[p]

                # ----- Initialise state on first step -----
                if len(state) == 0:
                    state["step"] = 0
                    state["z"] = p.data.clone()
                    state["x"] = p.data.clone()
                    if momentum != 0.0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                z = state["z"]
                x = state["x"]

                # ----- Warmup schedule -----
                if warmup_steps > 0:
                    warmup_factor = min(t / warmup_steps, 1.0)
                else:
                    warmup_factor = 1.0
                effective_lr = lr * warmup_factor

                # ----- Weight decay: add wd * z to gradient -----
                if wd != 0.0:
                    grad = grad.add(z, alpha=wd)

                # ----- Momentum buffer -----
                if momentum != 0.0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    update = buf
                else:
                    update = grad

                # ----- Update z -----
                z.add_(update, alpha=-effective_lr)

                # ----- Polyak-Ruppert averaging: advance x -----
                if r == 0.0:
                    x.add_(z - x, alpha=1.0 / t)
                else:
                    if "cum_weight" not in state:
                        state["cum_weight"] = 0.0
                    w_t = t ** r
                    state["cum_weight"] += w_t
                    x.add_(z - x, alpha=w_t / state["cum_weight"])

                # ----- Write z back to param.data -----
                p.data.copy_(z)

        return loss


class ScheduleFreeAdamW(Optimizer):
    """AdamW with schedule-free iterate averaging (Defazio et al. 2024).

    Maintains:
        z          -- the Adam-updated sequence (stored in param.data during training)
        x          -- the Polyak-Ruppert running average of z (for evaluation)
        exp_avg    -- Adam first-moment estimate (m1)
        exp_avg_sq -- Adam second-moment estimate (m2)

    Args:
        params: Iterable of parameters or param groups.
        lr: Base learning rate (default 0.0025).
        betas: (beta1, beta2) Adam momentum coefficients (default (0.9, 0.999)).
        eps: Numerical stability term (default 1e-8).
        weight_decay: Decoupled weight decay coefficient applied to z (default 0.0).
        warmup_steps: Linear LR ramp over first N steps (default 0).
        r: Averaging exponent -- 0.0 = uniform Polyak-Ruppert,
           >0 = recent-biased (default 0.0).
    """

    def __init__(
        self,
        params,
        lr: float = 0.0025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            r=r,
        )
        super().__init__(params, defaults)
        self._mode = "train"

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Switch parameters to z-sequence (training mode)."""
        if self._mode == "train":
            return
        self._swap_to("z")
        self._mode = "train"

    def switch_to_eval(self) -> None:
        """Switch parameters to x-sequence (evaluation / inference mode)."""
        if self._mode == "eval":
            return
        self._swap_to("x")
        self._mode = "eval"

    def _swap_to(self, target: str) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if not state:
                    continue
                if target == "x":
                    p.data.copy_(state["x"])
                else:
                    p.data.copy_(state["z"])

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # type: ignore[override]
        """Perform one schedule-free AdamW step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            warmup_steps = group["warmup_steps"]
            r = group["r"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # ----- Initialise state on first step -----
                if len(state) == 0:
                    state["step"] = 0
                    state["z"] = p.data.clone()
                    state["x"] = p.data.clone()
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                z = state["z"]
                x = state["x"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # ----- Warmup schedule -----
                if warmup_steps > 0:
                    warmup_factor = min(t / warmup_steps, 1.0)
                else:
                    warmup_factor = 1.0
                effective_lr = lr * warmup_factor

                # ----- Decoupled weight decay applied to z -----
                if wd != 0.0:
                    z.mul_(1.0 - effective_lr * wd)

                # ----- Adam moment updates -----
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias-corrected step size
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                step_size = effective_lr * math.sqrt(bc2) / bc1

                # ----- Update z via Adam -----
                denom = exp_avg_sq.sqrt().add_(eps)
                z.addcdiv_(exp_avg, denom, value=-step_size)

                # ----- Polyak-Ruppert averaging: advance x -----
                if r == 0.0:
                    x.add_(z - x, alpha=1.0 / t)
                else:
                    if "cum_weight" not in state:
                        state["cum_weight"] = 0.0
                    w_t = t ** r
                    state["cum_weight"] += w_t
                    x.add_(z - x, alpha=w_t / state["cum_weight"])

                # ----- Write z back to param.data -----
                p.data.copy_(z)

        return loss


def make_schedule_free(
    optimizer_class: str,
    params,
    **kwargs,
) -> ScheduleFreeSGD | ScheduleFreeAdamW:
    """Factory function for Schedule-Free optimizers.

    Args:
        optimizer_class: Either ``'sgd'`` or ``'adamw'`` (case-insensitive).
        params: Iterable of parameters or param groups passed to the optimizer.
        **kwargs: Additional keyword arguments forwarded to the optimizer
                  constructor.

    Returns:
        A :class:`ScheduleFreeSGD` or :class:`ScheduleFreeAdamW` instance.

    Raises:
        ValueError: If *optimizer_class* is not ``'sgd'`` or ``'adamw'``.
    """
    key = optimizer_class.lower()
    if key == "sgd":
        return ScheduleFreeSGD(params, **kwargs)
    if key == "adamw":
        return ScheduleFreeAdamW(params, **kwargs)
    raise ValueError(
        f"Unknown optimizer_class '{optimizer_class}'. "
        "Expected 'sgd' or 'adamw'."
    )
