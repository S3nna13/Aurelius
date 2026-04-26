"""Schedule-Free optimization: eliminate LR schedulers via iterate averaging.

Maintains two sequences inside the optimizer:
- z: gradient-update sequence (used during training forward/backward)
- x: Polyak-Ruppert averaged sequence (used for evaluation)

At each step the optimizer interpolates: y = (1 - beta1)*z + beta1*x,
computes gradients at y, updates z via the underlying rule (Adam or SGD),
then advances the running average x.

Reference: Defazio et al. 2024, arXiv:2405.15682
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class ScheduleFreeAdamW(Optimizer):
    """AdamW with schedule-free iterate averaging (Defazio et al. 2024).

    The optimizer maintains:
        z  -- the "gradient" sequence updated by Adam
        x  -- the Polyak-Ruppert running average of z (used for evaluation)

    The model's .data tensors store z during training and x during inference.
    Call optimizer.train() / optimizer.switch_to_eval() before switching modes.

    Args:
        params: Iterable of parameters or param groups.
        lr: Base learning rate (default 1e-3).
        betas: (beta1, beta2) Adam momentum coefficients (default (0.9, 0.999)).
        eps: Numerical stability term (default 1e-8).
        weight_decay: Decoupled weight decay applied to z (default 0.0).
        warmup_steps: Linear LR ramp over first N steps (default 0).
        r: Averaging exponent -- 0.0 = uniform Polyak-Ruppert,
           >0 = recent-biased (default 0.0).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            r=r,
        )
        super().__init__(params, defaults)
        # Track whether param tensors currently hold z (train) or x (eval)
        self._mode = "train"

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Switch parameters to z-sequence (gradient-update mode).

        Must be called before forward/backward passes.
        """
        if self._mode == "train":
            return
        self._swap_to("z")
        self._mode = "train"

    def switch_to_eval(self) -> None:
        """Switch parameters to x-sequence (averaged weights for inference).

        Must be called before evaluation/inference.
        Public alias avoids confusion with Python built-in eval().
        """
        if self._mode == "eval":
            return
        self._swap_to("x")
        self._mode = "eval"

    # Keep the standard optimizer interface name as well.
    def eval(self):  # noqa: A003
        """Alias for switch_to_eval() -- standard optimizer interface."""
        self.switch_to_eval()

    def _swap_to(self, target: str) -> None:
        """Copy target buffer into param.data in-place."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if not state:
                    continue  # not yet initialized
                if target == "x":
                    p.data.copy_(state["x"])
                else:
                    p.data.copy_(state["z"])

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        """Perform one schedule-free AdamW step.

        Must be called while in train mode (optimizer.train()).
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
                    # z starts as current param value
                    state["z"] = p.data.clone()
                    # x (running average) also starts at z
                    state["x"] = p.data.clone()
                    # Adam first moment
                    state["exp_avg"] = torch.zeros_like(p)
                    # Adam second moment
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

                # ----- Decoupled weight decay on z -----
                if wd != 0.0:
                    z.mul_(1.0 - effective_lr * wd)

                # ----- Adam moment updates -----
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bc1 = 1.0 - beta1**t
                bc2 = 1.0 - beta2**t
                step_size = effective_lr * math.sqrt(bc2) / bc1

                # ----- Update z -----
                denom = exp_avg_sq.sqrt().add_(eps)
                z.addcdiv_(exp_avg, denom, value=-step_size)

                # ----- Polyak-Ruppert averaging: update x -----
                if r == 0.0:
                    # Uniform: x_{t+1} = x_t + (z_{t+1} - x_t) / t
                    x.add_(z - x, alpha=1.0 / t)
                else:
                    # Recent-biased weighted average
                    if "cum_weight" not in state:
                        state["cum_weight"] = 0.0
                    w_t = t**r
                    state["cum_weight"] += w_t
                    x.add_(z - x, alpha=w_t / state["cum_weight"])

                # ----- Write z back to param.data (train mode) -----
                p.data.copy_(z)

        return loss


class ScheduleFreeSGD(Optimizer):
    """SGD with schedule-free iterate averaging (Defazio et al. 2024).

    The optimizer maintains:
        z  -- the "gradient" sequence updated by SGD (with momentum)
        x  -- the Polyak-Ruppert running average of z (used for evaluation)

    Args:
        params: Iterable of parameters or param groups.
        lr: Base learning rate (default 0.01).
        momentum: SGD momentum coefficient (default 0.9).
        weight_decay: L2 regularisation applied to z (default 0.0).
        warmup_steps: Linear LR ramp over first N steps (default 0).
        r: Averaging exponent -- 0.0 = uniform, >0 = recent-biased (default 0.0).
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
        """Switch parameters to x-sequence (evaluation mode)."""
        if self._mode == "eval":
            return
        self._swap_to("x")
        self._mode = "eval"

    def eval(self):  # noqa: A003
        """Alias for switch_to_eval() -- standard optimizer interface."""
        self.switch_to_eval()

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
    def step(self, closure=None) -> float | None:
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

                # ----- Initialise on first step -----
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

                # ----- Warmup -----
                if warmup_steps > 0:
                    warmup_factor = min(t / warmup_steps, 1.0)
                else:
                    warmup_factor = 1.0
                effective_lr = lr * warmup_factor

                # ----- Weight decay on z -----
                if wd != 0.0:
                    grad = grad.add(z, alpha=wd)

                # ----- Momentum -----
                if momentum != 0.0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    update = buf
                else:
                    update = grad

                # ----- Update z -----
                z.add_(update, alpha=-effective_lr)

                # ----- Polyak-Ruppert averaging: update x -----
                if r == 0.0:
                    x.add_(z - x, alpha=1.0 / t)
                else:
                    if "cum_weight" not in state:
                        state["cum_weight"] = 0.0
                    w_t = t**r
                    state["cum_weight"] += w_t
                    x.add_(z - x, alpha=w_t / state["cum_weight"])

                # ----- Write z back -----
                p.data.copy_(z)

        return loss
