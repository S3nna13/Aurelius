"""Stochastic Weight Averaging (SWA) with Cyclical LR for Aurelius LLM."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SWAConfig:
    """Configuration for SWA training with cyclical learning rate."""

    swa_start: int = 100       # step to begin averaging
    swa_freq: int = 5          # average every N steps
    swa_lr: float = 0.05       # constant LR used during SWA phase (informational)

    # Cyclical LR params
    cycle_length: int = 20     # steps per cycle
    cycle_mult: float = 1.0    # multiply cycle length after each restart
    min_lr: float = 1e-6       # trough of cosine
    max_lr: float = 1e-3       # peak of cosine


# ---------------------------------------------------------------------------
# CyclicalLRScheduler
# ---------------------------------------------------------------------------

class CyclicalLRScheduler:
    """Cosine annealing cyclical LR scheduler.

    Each cycle of length ``cycle_length`` sweeps:
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * cycle_step / cycle_length))
    """

    def __init__(self, optimizer: Optimizer, config: SWAConfig) -> None:
        self.optimizer = optimizer
        self.config = config

        self._step: int = 0
        self._cycle: int = 0
        self._cycle_step: int = 0
        self._current_cycle_length: int = config.cycle_length

    def get_lr(self) -> float:
        """Cosine-annealed LR for current position in the cycle."""
        cfg = self.config
        return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (
            1 + math.cos(math.pi * self._cycle_step / max(1, self._current_cycle_length))
        )

    def step(self) -> float:
        """Advance one step, update optimizer LR, handle cycle reset.

        Returns the learning rate *before* advancing (i.e. the LR used this step).
        """
        lr = self.get_lr()

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self._step += 1
        self._cycle_step += 1

        if self._cycle_step >= self._current_cycle_length:
            self._cycle += 1
            self._cycle_step = 0
            self._current_cycle_length = max(
                1,
                int(self._current_cycle_length * self.config.cycle_mult),
            )

        return lr

    def cycle_count(self) -> int:
        """Return the number of completed cycles."""
        return self._cycle


# ---------------------------------------------------------------------------
# SWAModel
# ---------------------------------------------------------------------------

class SWAModel:
    """Running average of model parameters for Stochastic Weight Averaging."""

    def __init__(self, model: nn.Module) -> None:
        self._n_averaged: int = 0
        self._swa_params: Optional[dict[str, Tensor]] = None

    def update(self, model: nn.Module) -> None:
        """Update running average: swa_p = (swa_p * n + p) / (n + 1)."""
        n = self._n_averaged

        if self._swa_params is None:
            self._swa_params = {
                name: param.detach().clone().float()
                for name, param in model.named_parameters()
            }
        else:
            for name, param in model.named_parameters():
                swa_p = self._swa_params[name]
                self._swa_params[name] = (swa_p * n + param.detach().float()) / (n + 1)

        self._n_averaged += 1

    def get_averaged_model(self, base_model: nn.Module) -> None:
        """Copy averaged parameters back into base_model in-place."""
        if self._swa_params is None:
            return

        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in self._swa_params:
                    param.copy_(self._swa_params[name].to(param.dtype))

    def reset(self) -> None:
        """Reset the running average."""
        self._n_averaged = 0
        self._swa_params = None


# ---------------------------------------------------------------------------
# update_bn
# ---------------------------------------------------------------------------

def update_bn(model: nn.Module, data_loader: list[Tensor]) -> None:
    """Re-estimate BatchNorm running statistics after SWA weight averaging.

    No-op if the model contains no BatchNorm layers.
    """
    bn_layers = [
        m for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]
    if not bn_layers:
        return

    for bn in bn_layers:
        bn.reset_running_stats()
        bn.training = True
        bn.momentum = None  # cumulative moving average

    was_training = model.training
    model.train()

    with torch.no_grad():
        for batch in data_loader:
            model(batch)

    if not was_training:
        model.eval()


# ---------------------------------------------------------------------------
# SWATrainer
# ---------------------------------------------------------------------------

class SWATrainer:
    """Training wrapper combining SWA with a cyclical LR scheduler."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: SWAConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config

        self._swa_model = SWAModel(model)
        self._scheduler = CyclicalLRScheduler(optimizer, config)
        self._step: int = 0

    def train_step(self, input_ids: Tensor) -> dict:
        """One forward/backward step with optional SWA update.

        Returns dict with keys: loss, lr, swa_n_averaged, step.
        """
        self.model.train()
        self.optimizer.zero_grad()

        loss, _logits, _pkv = self.model(input_ids, labels=input_ids)

        loss.backward()
        self.optimizer.step()
        lr = self._scheduler.step()

        cfg = self.config
        if self._step >= cfg.swa_start and self._step % cfg.swa_freq == 0:
            self._swa_model.update(self.model)

        result = {
            "loss": loss.item(),
            "lr": lr,
            "swa_n_averaged": self._swa_model._n_averaged,
            "step": self._step,
        }

        self._step += 1
        return result

    def finalize(self, data: Optional[list[Tensor]] = None) -> None:
        """Apply averaged weights to model; optionally update BN statistics."""
        self._swa_model.get_averaged_model(self.model)

        if data is not None:
            update_bn(self.model, data)
