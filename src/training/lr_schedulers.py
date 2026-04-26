"""
Learning rate schedulers for the Aurelius LLM project.

Pure PyTorch only — no HuggingFace, no scipy, no sklearn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SchedulerConfig:
    warmup_steps: int = 1000
    total_steps: int = 10000
    min_lr_ratio: float = 0.1
    num_cycles: float = 0.5


# ---------------------------------------------------------------------------
# Functional schedule helpers
# ---------------------------------------------------------------------------


def cosine_schedule_with_warmup(step: int, config: SchedulerConfig) -> float:
    """Linear warmup then cosine decay to min_lr_ratio.

    Returns a multiplier in [min_lr_ratio, 1.0].
    """
    warmup = config.warmup_steps
    total = config.total_steps
    min_ratio = config.min_lr_ratio
    cycles = config.num_cycles

    if step < warmup:
        # Linear warmup: 0 → 1
        return float(step) / float(max(1, warmup))

    # Cosine decay from 1.0 → min_lr_ratio
    progress = float(step - warmup) / float(max(1, total - warmup))
    cosine_val = math.cos(math.pi * cycles * 2.0 * progress)
    # cosine_val goes from 1.0 to -1.0 over one full cosine cycle when cycles=0.5
    multiplier = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + cosine_val)
    return float(max(min_ratio, min(1.0, multiplier)))


def linear_schedule_with_warmup(step: int, config: SchedulerConfig) -> float:
    """Linear warmup then linear decay to min_lr_ratio.

    Returns a multiplier in [min_lr_ratio, 1.0].
    """
    warmup = config.warmup_steps
    total = config.total_steps
    min_ratio = config.min_lr_ratio

    if step < warmup:
        return float(step) / float(max(1, warmup))

    progress = float(step - warmup) / float(max(1, total - warmup))
    multiplier = 1.0 - (1.0 - min_ratio) * progress
    return float(max(min_ratio, min(1.0, multiplier)))


def polynomial_schedule(
    step: int,
    config: SchedulerConfig,
    power: float = 1.0,
) -> float:
    """Linear warmup then polynomial decay.

    Decay formula: ((total_steps - step) / (total_steps - warmup_steps)) ** power,
    clamped to [min_lr_ratio, 1.0].
    """
    warmup = config.warmup_steps
    total = config.total_steps
    min_ratio = config.min_lr_ratio

    if step < warmup:
        return float(step) / float(max(1, warmup))

    if step >= total:
        return min_ratio

    decay_steps = total - warmup
    remaining = total - step
    multiplier = (float(remaining) / float(max(1, decay_steps))) ** power
    return float(max(min_ratio, min(1.0, multiplier)))


def wsd_schedule(
    step: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Warmup-Stable-Decay (WSD) schedule.

    Phases:
      [0, warmup_steps)                     → linear warmup 0 → 1.0
      [warmup_steps, warmup_steps + stable_steps) → constant 1.0
      [warmup_steps + stable_steps, ...)    → cosine decay 1.0 → min_lr_ratio
    """
    stable_start = warmup_steps
    decay_start = warmup_steps + stable_steps
    decay_start + decay_steps

    if step < stable_start:
        # Warmup
        return float(step) / float(max(1, warmup_steps))
    elif step < decay_start:
        # Stable
        return 1.0
    else:
        # Cosine decay
        progress = float(step - decay_start) / float(max(1, decay_steps))
        progress = min(1.0, progress)
        cosine_val = math.cos(math.pi * progress)
        multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + cosine_val)
        return float(max(min_lr_ratio, min(1.0, multiplier)))


# ---------------------------------------------------------------------------
# Scheduler classes
# ---------------------------------------------------------------------------


class WarmupCosineScheduler(LambdaLR):
    """LambdaLR wrapper that applies cosine_schedule_with_warmup."""

    def __init__(self, optimizer: optim.Optimizer, config: SchedulerConfig):
        self.config = config

        def _lambda(step: int) -> float:
            return cosine_schedule_with_warmup(step, config)

        super().__init__(optimizer, lr_lambda=_lambda)


class CyclicCosineScheduler:
    """Standalone stateful cosine annealing with restarts.

    Each cycle halves its period when num_cycles > 1 (if num_cycles == 1 the
    cycle length is constant).  get_lr(step) returns a float multiplier in
    [min_lr_ratio, 1.0].
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config

    def get_lr(self, step: int) -> float:
        config = self.config
        total = config.total_steps
        min_ratio = config.min_lr_ratio
        num_cycles = config.num_cycles

        if step <= 0:
            return 1.0  # start of first cycle

        if num_cycles <= 1:
            # Single cycle — standard cosine over total_steps
            progress = float(step) / float(max(1, total))
            progress = min(1.0, progress)
            val = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(max(min_ratio, min(1.0, val)))

        # Multi-cycle: first cycle has length = total / num_cycles,
        # each subsequent cycle is half the previous.
        # Locate which cycle we are in.
        first_cycle_steps = total / num_cycles
        cycle = 0
        cycle_start = 0.0
        cycle_len = first_cycle_steps
        while cycle_start + cycle_len <= step:
            cycle_start += cycle_len
            cycle_len = max(1.0, cycle_len / 2.0)
            cycle += 1
            if cycle_start >= total:
                break

        progress = (float(step) - cycle_start) / float(max(1.0, cycle_len))
        progress = min(1.0, progress)
        val = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(max(min_ratio, min(1.0, val)))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    config: SchedulerConfig,
) -> LambdaLR:
    """Return a LambdaLR-based scheduler by name.

    Supported names: "cosine", "linear", "polynomial".
    """
    name = name.lower().strip()

    if name == "cosine":
        return WarmupCosineScheduler(optimizer, config)

    elif name == "linear":

        def _linear_lambda(step: int) -> float:
            return linear_schedule_with_warmup(step, config)

        return LambdaLR(optimizer, lr_lambda=_linear_lambda)

    elif name == "polynomial":

        def _poly_lambda(step: int) -> float:
            return polynomial_schedule(step, config)

        return LambdaLR(optimizer, lr_lambda=_poly_lambda)

    else:
        raise ValueError(
            f"Unknown scheduler name '{name}'. Choose from: 'cosine', 'linear', 'polynomial'."
        )
