"""Lookahead optimizer wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import Optimizer


@dataclass(frozen=True)
class LookaheadConfig:
    k: int = 6
    alpha: float = 0.5


class Lookahead:
    """Wrap a base optimizer with slow weight synchronization."""

    def __init__(self, base_optimizer: Optimizer, **kwargs) -> None:
        cfg = LookaheadConfig(**kwargs)
        if cfg.k <= 0:
            raise ValueError("k must be positive")
        if not 0.0 < cfg.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.base_optimizer = base_optimizer
        self.k = cfg.k
        self.alpha = cfg.alpha
        self._step = 0
        self._params = [param for group in base_optimizer.param_groups for param in group["params"]]
        self._slow_params = [param.detach().clone() for param in self._params]

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @property
    def slow_params(self) -> list[torch.Tensor]:
        return [param.detach().clone() for param in self._slow_params]

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _sync_slow_weights(self) -> None:
        for slow, param in zip(self._slow_params, self._params, strict=True):
            slow.add_(param.data - slow, alpha=self.alpha)
            param.data.copy_(slow)

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self._step += 1
        if self._step % self.k == 0:
            self._sync_slow_weights()
        return loss

    def state_dict(self) -> dict:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "slow_params": [param.detach().clone() for param in self._slow_params],
            "step": self._step,
            "k": self.k,
            "alpha": self.alpha,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        slow_params = state_dict["slow_params"]
        if len(slow_params) != len(self._slow_params):
            raise ValueError("slow parameter count mismatch")
        for target, source in zip(self._slow_params, slow_params, strict=True):
            target.copy_(source)
        self._step = int(state_dict["step"])
        self.k = int(state_dict["k"])
        self.alpha = float(state_dict["alpha"])
