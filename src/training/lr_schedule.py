import math
from dataclasses import dataclass


@dataclass
class SGDRConfig:
    lr_max: float = 3e-4          # peak learning rate
    lr_min: float = 1e-5          # minimum LR (trough of cosine)
    warmup_steps: int = 100       # linear warmup steps
    T0: int = 1000                # steps in first cosine cycle
    T_mult: float = 2.0           # multiply cycle length after each restart
    decay_factor: float = 1.0     # multiply lr_max by this after each restart (< 1 = decay peaks)


class SGDRScheduler:
    """Cosine annealing with warm restarts and linear warmup.

    Usage:
        cfg = SGDRConfig(lr_max=3e-4, warmup_steps=100, T0=1000, T_mult=2.0)
        sched = SGDRScheduler(cfg)

        for step in range(total_steps):
            lr = sched.get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
    """

    def __init__(self, cfg: SGDRConfig):
        self.cfg = cfg

    def get_lr(self, step: int) -> float:
        """Return the learning rate for the given step.

        Phase 1 (step < warmup_steps): linear warmup from 0 to lr_max
        Phase 2 (step >= warmup_steps): cosine annealing with warm restarts
        """
        cfg = self.cfg

        if step < cfg.warmup_steps:
            return cfg.lr_max * step / cfg.warmup_steps

        adjusted = step - cfg.warmup_steps
        cycle = 0
        cycle_len = cfg.T0
        cycle_start = 0
        current_lr_max = cfg.lr_max

        while cycle_start + cycle_len <= adjusted:
            cycle_start += cycle_len
            cycle_len = int(cycle_len * cfg.T_mult)
            current_lr_max *= cfg.decay_factor
            cycle += 1

        cycle_step = adjusted - cycle_start
        t = cycle_step / cycle_len
        return cfg.lr_min + 0.5 * (current_lr_max - cfg.lr_min) * (1 + math.cos(math.pi * t))

    def get_cycle_info(self, step: int) -> dict:
        """Return cycle information for the given step."""
        cfg = self.cfg

        adjusted = step - cfg.warmup_steps
        if adjusted < 0:
            return {
                "cycle": 0,
                "cycle_step": step,
                "cycle_len": cfg.warmup_steps,
                "lr_max_this_cycle": cfg.lr_max,
            }

        cycle = 0
        cycle_len = cfg.T0
        cycle_start = 0
        current_lr_max = cfg.lr_max

        while cycle_start + cycle_len <= adjusted:
            cycle_start += cycle_len
            cycle_len = int(cycle_len * cfg.T_mult)
            current_lr_max *= cfg.decay_factor
            cycle += 1

        cycle_step = adjusted - cycle_start
        return {
            "cycle": cycle,
            "cycle_step": cycle_step,
            "cycle_len": cycle_len,
            "lr_max_this_cycle": current_lr_max,
        }
