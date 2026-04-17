"""
Advanced LR scheduling for the Aurelius LLM project.

Implements:
  - WSDScheduler        — Warmup-Stable-Decay (MiniCPM / Mistral approach)
  - CosineWithRestartsScheduler — SGDR cosine restarts
  - InverseSqrtScheduler — T5 / fairseq inverse-sqrt warmup
  - LRSchedulerComparison — utility to compare scheduler trajectories

Pure native PyTorch only — no HuggingFace, no scipy, no sklearn.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


# ---------------------------------------------------------------------------
# WSDScheduler
# ---------------------------------------------------------------------------

class WSDScheduler(LRScheduler):
    """Warmup-Stable-Decay learning rate scheduler.

    Phases
    ------
    1. Warmup  [0, warmup_steps)          : linear ramp 0 → base_lr
    2. Stable  [warmup_steps,
                warmup_steps+stable_steps): constant base_lr
    3. Decay   [warmup_steps+stable_steps,
                warmup_steps+stable_steps+decay_steps): cosine decay
                                            base_lr → min_lr_ratio * base_lr
    4. Post-decay                         : min_lr_ratio * base_lr

    Parameters
    ----------
    optimizer    : wrapped PyTorch optimizer
    warmup_steps : number of warmup steps
    stable_steps : number of stable steps
    decay_steps  : number of cosine-decay steps
    min_lr_ratio : final lr as a fraction of base_lr (default 0.1)
    last_step    : last completed step for resuming (-1 = fresh start)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr_ratio: float = 0.1,
        last_step: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = min_lr_ratio
        # LRScheduler.__init__ calls step() once, setting last_epoch to 0
        super().__init__(optimizer, last_epoch=last_step)

    def get_lr(self) -> List[float]:  # type: ignore[override]
        step = self.last_epoch  # 0-indexed current step

        warmup_end = self.warmup_steps
        stable_end = warmup_end + self.stable_steps
        decay_end = stable_end + self.decay_steps

        lrs = []
        for base_lr in self.base_lrs:
            if step < warmup_end:
                # Linear warmup: step=0 → 0, step=warmup_steps → base_lr
                scale = step / max(1, warmup_end)
                lrs.append(base_lr * scale)
            elif step < stable_end:
                lrs.append(base_lr)
            elif step < decay_end:
                # Cosine decay from base_lr to min_lr_ratio * base_lr
                progress = (step - stable_end) / max(1, self.decay_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                min_lr = self.min_lr_ratio * base_lr
                lrs.append(min_lr + (base_lr - min_lr) * cosine)
            else:
                lrs.append(self.min_lr_ratio * base_lr)
        return lrs


# ---------------------------------------------------------------------------
# CosineWithRestartsScheduler
# ---------------------------------------------------------------------------

class CosineWithRestartsScheduler(LRScheduler):
    """SGDR cosine annealing with warm restarts.

    Within each cycle the LR follows a cosine from base_lr down to
    min_lr_ratio * base_lr.  After each restart the cycle length is
    multiplied by T_mult.

    Parameters
    ----------
    optimizer    : wrapped PyTorch optimizer
    T_0          : length of the first cycle (steps)
    T_mult       : cycle-length multiplier after each restart (default 1)
    min_lr_ratio : minimum lr as fraction of base_lr (default 0.0)
    last_step    : last completed step for resuming (-1 = fresh start)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        min_lr_ratio: float = 0.0,
        last_step: int = -1,
    ) -> None:
        if T_0 <= 0:
            raise ValueError("T_0 must be a positive integer")
        self.T_0 = T_0
        self.T_mult = T_mult
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch=last_step)

    def _cycle_info(self, step: int):
        """Return (position_in_cycle, cycle_length) for a given step."""
        if self.T_mult == 1:
            cycle_len = self.T_0
            pos = step % self.T_0
        else:
            # Geometric series: cumulative length after n cycles
            # = T_0 * (T_mult^n - 1) / (T_mult - 1)
            # Find which cycle we are in.
            if step < self.T_0:
                return step, self.T_0
            # n-th cycle starts at T_0 * (T_mult^n - 1) / (T_mult - 1)
            n = math.floor(
                math.log(
                    step * (self.T_mult - 1) / self.T_0 + 1,
                    self.T_mult,
                )
            )
            t_start = int(self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1))
            cycle_len = int(self.T_0 * self.T_mult**n)
            pos = step - t_start
        return pos, cycle_len

    def get_lr(self) -> List[float]:  # type: ignore[override]
        step = self.last_epoch
        pos, cycle_len = self._cycle_info(step)
        progress = pos / max(1, cycle_len)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lrs = []
        for base_lr in self.base_lrs:
            min_lr = self.min_lr_ratio * base_lr
            lrs.append(min_lr + (base_lr - min_lr) * cosine)
        return lrs


# ---------------------------------------------------------------------------
# InverseSqrtScheduler
# ---------------------------------------------------------------------------

class InverseSqrtScheduler(LRScheduler):
    """Inverse-square-root learning rate scheduler with linear warmup.

    Schedule
    --------
    - step < warmup_steps : lr = base_lr * step / warmup_steps   (linear)
    - step >= warmup_steps: lr = base_lr * sqrt(warmup_steps / step)

    Parameters
    ----------
    optimizer    : wrapped PyTorch optimizer
    warmup_steps : number of linear-warmup steps
    last_step    : last completed step for resuming (-1 = fresh start)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        last_step: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_step)

    def get_lr(self) -> List[float]:  # type: ignore[override]
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                scale = step / max(1, self.warmup_steps)
            else:
                scale = math.sqrt(self.warmup_steps / max(1, step))
            lrs.append(base_lr * scale)
        return lrs


# ---------------------------------------------------------------------------
# LRSchedulerComparison
# ---------------------------------------------------------------------------

class LRSchedulerComparison:
    """Utility to compare scheduler lr trajectories without training.

    Parameters
    ----------
    schedulers : dict mapping a human-readable name to a scheduler object
                 that supports `step()` and exposes `get_last_lr()` or
                 `optimizer.param_groups[*]['lr']`.
    """

    def __init__(self, schedulers: Dict[str, Any]) -> None:
        self.schedulers = schedulers

    def simulate(self, n_steps: int) -> Dict[str, List[float]]:
        """Step each scheduler n_steps times and record lr at each step.

        Returns
        -------
        dict mapping name → list of lr values (length n_steps).
        The lr recorded at index t is the lr *after* calling step() for the
        (t+1)-th time, i.e. the lr that would be used at training step t.
        """
        results: Dict[str, List[float]] = {name: [] for name in self.schedulers}
        for _ in range(n_steps):
            for name, sched in self.schedulers.items():
                sched.step()
                lr = sched.optimizer.param_groups[0]["lr"]
                results[name].append(lr)
        return results

    @staticmethod
    def find_crossover(lrs_a: List[float], lrs_b: List[float]) -> int:
        """Return the first step index where lrs_a[step] > lrs_b[step].

        Returns -1 if lrs_a never exceeds lrs_b.
        """
        for i, (a, b) in enumerate(zip(lrs_a, lrs_b)):
            if a > b:
                return i
        return -1
