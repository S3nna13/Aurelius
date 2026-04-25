from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class DPSGDConfig:
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    batch_size: int = 256
    delta: float = 1e-5


class DPSGDOptimizer:
    """DP-SGD: per-sample gradient clipping + Gaussian noise injection."""

    def __init__(self, optimizer: torch.optim.Optimizer, config: DPSGDConfig) -> None:
        if config.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if config.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be non-negative")
        if config.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not (0 < config.delta < 1):
            raise ValueError("delta must be in (0, 1)")
        self.optimizer = optimizer
        self.config = config

    def _clip_grad(self, params: List[torch.Tensor]) -> float:
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                total_norm += p.grad.detach().norm(2).item() ** 2
        grad_norm = math.sqrt(total_norm)
        clip_coef = min(1.0, self.config.max_grad_norm / (grad_norm + 1e-8))
        for p in params:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)
        return grad_norm

    def _add_noise(self, params: List[torch.Tensor]) -> None:
        # sigma divided by batch_size because gradients are averaged over the batch
        sigma = (
            self.config.noise_multiplier
            * self.config.max_grad_norm
            / self.config.batch_size
        )
        for p in params:
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * sigma
                p.grad.detach().add_(noise)

    def step(self, params: List[torch.Tensor]) -> Dict[str, float]:
        mean_grad_norm = self._clip_grad(params)
        noise_std = (
            self.config.noise_multiplier
            * self.config.max_grad_norm
            / self.config.batch_size
        )
        self._add_noise(params)
        self.optimizer.step()
        return {"mean_grad_norm": mean_grad_norm, "noise_std": noise_std}

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def get_privacy_spent(self, steps: int) -> Dict[str, float]:
        # Simplified upper bound — not tight; real accounting uses RDP moments.
        # epsilon ≈ noise_multiplier^-2 * 2 * ln(1.25/delta) * steps / batch_size
        epsilon = (
            (self.config.noise_multiplier ** -2)
            * 2.0
            * math.log(1.25 / self.config.delta)
            * steps
            / self.config.batch_size
        )
        return {
            "epsilon": epsilon,
            "delta": self.config.delta,
            "steps": steps,
        }
