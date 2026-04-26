"""Warmup LR scheduler: linear warmup, cosine decay, get_current_lr."""

import math
from enum import StrEnum


class SchedulerType(StrEnum):
    LINEAR_WARMUP = "linear_warmup"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"


class WarmupScheduler:
    """Learning-rate scheduler with linear warmup and configurable decay."""

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int = 100,
        total_steps: int = 10000,
        scheduler_type: SchedulerType = SchedulerType.COSINE,
        min_lr: float = 0.0,
        n_cycles: int = 1,
    ) -> None:
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.scheduler_type = scheduler_type
        self.min_lr = min_lr
        self.n_cycles = max(1, n_cycles)

    # ------------------------------------------------------------------
    def get_lr(self, step: int) -> float:
        """Return the learning rate for the given step."""
        # Linear warmup phase
        if step < self.warmup_steps:
            lr = self.base_lr * step / max(1, self.warmup_steps)
            return float(max(self.min_lr, min(self.base_lr, lr)))

        # Decay phase
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / decay_steps)

        if self.scheduler_type == SchedulerType.COSINE:
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        elif self.scheduler_type == SchedulerType.COSINE_WITH_RESTARTS:
            # Divide decay phase into n_cycles equal cycles
            cycle_progress = (progress * self.n_cycles) % 1.0
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * cycle_progress))

        elif self.scheduler_type == SchedulerType.POLYNOMIAL:
            lr = self.base_lr * (1.0 - progress) ** 2

        elif self.scheduler_type == SchedulerType.LINEAR_WARMUP:
            # Post-warmup: linear decay from base_lr to min_lr
            lr = self.base_lr + (self.min_lr - self.base_lr) * progress

        else:
            lr = self.base_lr

        # Clamp to [min_lr, base_lr]
        return float(max(self.min_lr, min(self.base_lr, lr)))

    def warmup_progress(self, step: int) -> float:
        """Return warmup fraction in [0.0, 1.0]."""
        return min(1.0, step / max(1, self.warmup_steps))

    def lr_schedule(self, n_steps: int) -> list[float]:
        """Return a list of learning-rate values for steps 0..n_steps-1."""
        return [self.get_lr(i) for i in range(n_steps)]
