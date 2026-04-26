"""Learning rate warmup scheduler with linear/cosine options."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class WarmupScheduler:
    """Linear or cosine LR warmup then constant/cosine decay."""

    warmup_steps: int = 1000
    total_steps: int = 10000
    min_lr_ratio: float = 0.0
    schedule: str = "cosine"  # linear or cosine

    def get_lr(self, step: int, base_lr: float) -> float:
        if step < self.warmup_steps:
            return base_lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        if self.schedule == "linear":
            return base_lr * (1.0 - progress * (1.0 - self.min_lr_ratio))
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * decay)


WARMUP_SCHEDULER = WarmupScheduler()
