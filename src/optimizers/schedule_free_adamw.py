"""
src/optimizers/schedule_free_adamw.py

Schedule-Free AdamW (Defazio 2024, arXiv:2405.15682).

Variant of AdamW that does not require a learning-rate schedule.  It maintains
two sequences (``y``, ``z``) plus a polynomial average ``x``.  The model is
trained using ``y`` (the interpolated iterate the gradient is evaluated at)
and evaluated using ``x`` (the averaged iterate).

Recurrence (per parameter, step ``k``):

    y_k = (1 - beta1) * z_k + beta1 * x_k                # interpolation
    g_k = grad f(y_k)
    v_{k+1} = beta2 * v_k + (1 - beta2) * g_k^2          # Adam 2nd moment
    hat_v = v_{k+1} / (1 - beta2^{k+1})                  # bias correction
    d_k   = sqrt(hat_v) + eps
    z_{k+1} = z_k - lr * (g_k / d_k + weight_decay * y_k)
    weight_{k+1} = lr^2 * max(k+1, 1)^r                  # polynomial avg weight
    weight_sum += weight_{k+1}
    c_{k+1} = weight_{k+1} / weight_sum
    x_{k+1} = (1 - c_{k+1}) * x_k + c_{k+1} * z_{k+1}
    y_{k+1} = (1 - beta1) * z_{k+1} + beta1 * x_{k+1}

The ``p.data`` tensor holds ``y`` during training (so that autograd uses the
correct iterate).  The ``eval_mode()`` method swaps ``p.data`` to ``x``;
``train_mode()`` swaps back to ``y``.  ``eval()`` and ``train()`` are aliases
matching the standard Schedule-Free API.

Pure-PyTorch implementation.  No external dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer


class ScheduleFreeAdamW(Optimizer):
    """Schedule-Free AdamW optimizer.

    Args:
        params:        Iterable of parameters or parameter groups.
        lr:            Base learning rate.  Default: ``1e-3``.
        betas:         ``(beta1, beta2)``.  ``beta1`` is the interpolation
                       coefficient between ``z`` and ``x`` (plays the role of
                       Polyak momentum); ``beta2`` is the Adam second-moment
                       EMA decay.  Default: ``(0.9, 0.999)``.
        eps:           Numerical stabiliser added to the denominator.
                       Default: ``1e-8``.
        weight_decay:  Decoupled weight-decay coefficient.  Default: ``0.0``.
        warmup_steps:  Linear warmup length.  ``lr`` is scaled by
                       ``min(step / warmup_steps, 1)`` for the first few
                       iterations.  ``0`` disables warmup.  Default: ``0``.
        r:             Polynomial-averaging exponent for the ``x`` sequence.
                       ``r = 0`` gives uniform weighting.  Default: ``0.0``.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
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
        if warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps: {warmup_steps}")

        defaults = dict(
            lr=lr,
            betas=(float(b1), float(b2)),
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            r=r,
            weight_sum=0.0,
            train_mode=True,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------ #
    # eval_mode / train_mode: swap p.data between y and x                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def eval_mode(self) -> None:
        """Swap ``p.data`` from ``y`` to ``x`` (the averaged iterate).

        Idempotent: calling it when already in eval-mode is a no-op.
        """
        for group in self.param_groups:
            if not group.get("train_mode", True):
                continue
            for p in group["params"]:
                state = self.state[p]
                if "x" not in state:
                    # Parameter has never been stepped; y == x == p.data.
                    continue
                x = state["x"]
                state["y_backup"] = p.data.clone()
                p.data.copy_(x)
            group["train_mode"] = False

    @torch.no_grad()
    def train_mode(self) -> None:
        """Swap ``p.data`` back from ``x`` to ``y`` (the training iterate)."""
        for group in self.param_groups:
            if group.get("train_mode", True):
                continue
            for p in group["params"]:
                state = self.state[p]
                if "y_backup" not in state:
                    continue
                p.data.copy_(state["y_backup"])
                del state["y_backup"]
            group["train_mode"] = True

    # Aliases matching the standard Schedule-Free reference API.
    def eval(self) -> None:  # noqa: A003 - intentional name
        """Alias for :meth:`eval_mode`."""
        self.eval_mode()

    def train(self) -> None:
        """Alias for :meth:`train_mode`."""
        self.train_mode()

    # ------------------------------------------------------------------ #
    # Optimization step                                                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
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
            if not group.get("train_mode", True):
                raise RuntimeError(
                    "ScheduleFreeAdamW.step() called while the optimizer "
                    "is in eval mode.  Call .train() before stepping."
                )

            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            warmup = group["warmup_steps"]
            r = group["r"]

            # Per-group effective step counter (shared across params in group)
            # so that weight_sum / ckp1 are consistent.
            group_step = group.get("_step", 0) + 1
            group["_step"] = group_step

            # Linear warmup on lr
            if warmup > 0:
                sched = min(group_step / float(warmup), 1.0)
            else:
                sched = 1.0
            lr_eff = lr * sched

            # Polynomial averaging weight
            weight = (lr_eff**2) * (max(group_step, 1) ** r)
            weight_sum = group["weight_sum"] + weight
            group["weight_sum"] = weight_sum
            if weight_sum > 0.0:
                ckp1 = weight / weight_sum
            else:
                ckp1 = 0.0

            bias_correction2 = 1.0 - beta2**group_step

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ScheduleFreeAdamW does not support sparse gradients.")

                state = self.state[p]

                # --- Lazy state initialisation -------------------------- #
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # On the very first step p.data holds y == x == z.
                    state["z"] = p.data.clone()
                    state["x"] = p.data.clone()

                state["step"] += 1

                exp_avg_sq = state["exp_avg_sq"]
                z = state["z"]
                x = state["x"]
                y = p.data  # the current train iterate lives in p.data

                # v_{k+1} = beta2 * v_k + (1-beta2) * g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)

                # z update: decoupled weight decay applied on y
                #   z <- z - lr_eff * (grad / denom + wd * y)
                z.addcdiv_(grad, denom, value=-lr_eff)
                if wd != 0.0:
                    z.add_(y, alpha=-lr_eff * wd)

                # x polynomial average: x <- (1-ckp1)*x + ckp1*z
                x.mul_(1.0 - ckp1).add_(z, alpha=ckp1)

                # y = (1-beta1)*z + beta1*x  -> write into p.data in-place
                y.copy_(x).mul_(beta1).add_(z, alpha=1.0 - beta1)

        return loss
