"""Weight decay scheduling utilities and AdamWR optimizer.

Implements the notation from Loshchilov and Hutter (2019):

- ``eta_t``: schedule multiplier
- ``lambda`` / ``lambda_norm``: raw and normalized weight decay
- ``T_i`` / ``T_cur``: restart length and elapsed time in the current restart

Reference: Decoupled Weight Decay Regularization, arXiv:1711.05101.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.optim import Optimizer


def normalized_weight_decay(lambda_norm: float, b: int, B: int, T: float) -> float:
    """Return ``lambda = lambda_norm * sqrt(b / (B * T))`` from the paper."""
    if b <= 0:
        raise ValueError(f"b must be positive, got {b}")
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    return lambda_norm * math.sqrt(float(b) / (float(B) * float(T)))


def cosine_eta_t(
    T_cur: float,
    T_i: float,
    eta_min: float = 0.0,
    eta_max: float = 1.0,
) -> float:
    """Return the cosine-annealed schedule multiplier ``eta_t``."""
    if T_i <= 0.0:
        raise ValueError(f"T_i must be positive, got {T_i}")
    if eta_max < eta_min:
        raise ValueError(f"eta_max must be >= eta_min, got eta_min={eta_min}, eta_max={eta_max}")
    T_cur = min(max(T_cur, 0.0), T_i)
    return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(math.pi * T_cur / T_i))


@dataclass
class AdamWRConfig:
    """Configuration for AdamW with cosine schedule and warm restarts."""

    alpha: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    lambda_norm: float = 0.0
    eta_min: float = 0.0
    eta_max: float = 1.0
    T_0: float = 1.0
    T_mult: float = 1.0
    b: int = 1
    B: int = 1


class AdamWR(Optimizer):
    """AdamW with the paper's ``eta_t`` schedule and normalized weight decay."""

    def __init__(self, params, config: AdamWRConfig | None = None, **kwargs) -> None:
        if config is None:
            config = AdamWRConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either config or keyword arguments, not both")

        if config.alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {config.alpha}")
        if not 0.0 <= config.beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {config.beta1}")
        if not 0.0 <= config.beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {config.beta2}")
        if config.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {config.epsilon}")
        if config.T_0 <= 0.0:
            raise ValueError(f"T_0 must be positive, got {config.T_0}")
        if config.T_mult < 1.0:
            raise ValueError(f"T_mult must be >= 1, got {config.T_mult}")

        defaults = dict(
            alpha=config.alpha,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
        )
        super().__init__(params, defaults)

        self.config = config
        self.steps_per_epoch = max(1, math.ceil(config.B / config.b))
        self.restart_index = 0
        self.T_i = float(config.T_0)
        self.T_cur = 0.0
        self.last_eta_t = cosine_eta_t(
            T_cur=0.0,
            T_i=self.T_i,
            eta_min=config.eta_min,
            eta_max=config.eta_max,
        )
        self.last_lambda = normalized_weight_decay(
            lambda_norm=config.lambda_norm,
            b=config.b,
            B=config.B,
            T=self.T_i,
        )

    @torch.no_grad()
    def step(self, closure=None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        eta_t = cosine_eta_t(
            T_cur=self.T_cur,
            T_i=self.T_i,
            eta_min=self.config.eta_min,
            eta_max=self.config.eta_max,
        )
        lambda_t = normalized_weight_decay(
            lambda_norm=self.config.lambda_norm,
            b=self.config.b,
            B=self.config.B,
            T=self.T_i,
        )

        self.last_eta_t = eta_t
        self.last_lambda = lambda_t

        for group in self.param_groups:
            alpha = group["alpha"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            epsilon = group["epsilon"]

            for theta_t in group["params"]:
                if theta_t.grad is None:
                    continue
                if theta_t.grad.is_sparse:
                    raise RuntimeError("AdamWR does not support sparse gradients")

                g_t = theta_t.grad.detach()
                state = self.state[theta_t]
                if len(state) == 0:
                    state["t"] = 0
                    state["m_t"] = torch.zeros_like(theta_t)
                    state["v_t"] = torch.zeros_like(theta_t)

                state["t"] += 1
                t = state["t"]
                m_t = state["m_t"]
                v_t = state["v_t"]

                theta_prev = theta_t.detach().clone()

                m_t.mul_(beta1).add_(g_t, alpha=1.0 - beta1)
                v_t.mul_(beta2).addcmul_(g_t, g_t, value=1.0 - beta2)

                m_hat_t = m_t / (1.0 - beta1**t)
                v_hat_t = v_t / (1.0 - beta2**t)
                denom = v_hat_t.sqrt().add_(epsilon)

                theta_t.addcdiv_(m_hat_t, denom, value=-eta_t * alpha)
                if lambda_t != 0.0:
                    theta_t.add_(theta_prev, alpha=-eta_t * lambda_t)

        self._advance_schedule()
        return loss

    def _advance_schedule(self) -> None:
        self.T_cur += 1.0 / float(self.steps_per_epoch)
        if self.T_cur + 1e-12 >= self.T_i:
            self.restart_index += 1
            self.T_cur = 0.0
            self.T_i *= self.config.T_mult

    def get_schedule_state(self) -> dict[str, float | int]:
        """Return the current scheduling state in paper notation."""
        return {
            "restart_index": self.restart_index,
            "T_i": self.T_i,
            "T_cur": self.T_cur,
            "eta_t": self.last_eta_t,
            "lambda": self.last_lambda,
            "steps_per_epoch": self.steps_per_epoch,
        }
