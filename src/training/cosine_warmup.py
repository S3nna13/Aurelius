"""
src/training/cosine_warmup.py

Cosine LR schedule with linear warmup. The standard schedule for
transformer pretraining: linear warmup then cosine decay to min_lr.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class CosineWarmupSchedule:
    """Configuration for cosine LR schedule with linear warmup.

    Attributes:
        warmup_steps:   Number of steps for linear warmup phase.
        total_steps:    Total number of training steps.
        min_lr_ratio:   Minimum LR as a fraction of the base LR. Default: 0.1.
    """

    warmup_steps: int
    total_steps: int
    min_lr_ratio: float = 0.1


def get_lr_multiplier(step: int, schedule: CosineWarmupSchedule) -> float:
    """Compute the LR multiplier for a given step.

    Three phases:
      - Warmup (step < warmup_steps): linear ramp from 0 to 1.
      - Decay (warmup_steps <= step < total_steps): cosine decay to min_lr_ratio.
      - After training (step >= total_steps): constant at min_lr_ratio.

    Args:
        step:     Current training step (0-indexed).
        schedule: CosineWarmupSchedule configuration.

    Returns:
        Scalar multiplier in [0, 1].
    """
    if schedule.warmup_steps > 0 and step < schedule.warmup_steps:
        return step / schedule.warmup_steps

    if step >= schedule.total_steps:
        return schedule.min_lr_ratio

    # Cosine decay phase
    progress = (step - schedule.warmup_steps) / max(
        1, schedule.total_steps - schedule.warmup_steps
    )
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule.min_lr_ratio + (1.0 - schedule.min_lr_ratio) * cosine_decay


class WarmupCosineScheduler(_LRScheduler):
    """PyTorch LR scheduler: linear warmup followed by cosine decay.

    Args:
        optimizer:      Wrapped optimizer.
        warmup_steps:   Number of warmup steps.
        total_steps:    Total number of training steps.
        min_lr_ratio:   Minimum LR ratio at the end of cosine decay. Default: 0.1.
        last_epoch:     The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self._schedule = CosineWarmupSchedule(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        multiplier = get_lr_multiplier(self.last_epoch, self._schedule)
        return [base_lr * multiplier for base_lr in self.base_lrs]


class InverseSqrtScheduler(_LRScheduler):
    """PyTorch LR scheduler: linear warmup followed by inverse-sqrt decay.

    During warmup the LR rises linearly from 0 to base_lr.
    After warmup the LR decays as base_lr * sqrt(warmup_steps) / sqrt(step).

    Args:
        optimizer:    Wrapped optimizer.
        warmup_steps: Number of warmup steps.
        last_epoch:   The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self._warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        warmup_steps = self._warmup_steps

        if warmup_steps > 0 and step < warmup_steps:
            multiplier = step / warmup_steps
        else:
            effective_step = max(step, warmup_steps)
            multiplier = math.sqrt(warmup_steps) / math.sqrt(effective_step)

        return [base_lr * multiplier for base_lr in self.base_lrs]
