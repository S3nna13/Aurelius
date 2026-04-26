"""
Comprehensive learning rate schedulers for Aurelius LLM training.

Pure PyTorch / stdlib only — no transformers, einops, trl, xformers, flash_attn,
bitsandbytes, peft, diffusers, datasets, accelerate, or deepspeed.
"""

import math
from typing import Any

# ---------------------------------------------------------------------------
# WarmupScheduler — linear warmup from 0 → base_lr
# ---------------------------------------------------------------------------


class WarmupScheduler:
    """Linear warmup scheduler: lr = base_lr * (step / warmup_steps)."""

    def __init__(self, optimizer, warmup_steps: int, base_lr: float):
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be > 0")
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self._step = 0
        self._current_lr = 0.0
        # initialise optimizer LR to 0
        self._set_lr(0.0)

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self) -> None:
        self._step += 1
        if self._step <= self.warmup_steps:
            self._current_lr = self.base_lr * (self._step / self.warmup_steps)
        else:
            self._current_lr = self.base_lr
        self._set_lr(self._current_lr)

    def get_lr(self) -> float:
        return self._current_lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "base_lr": self.base_lr,
            "_step": self._step,
            "_current_lr": self._current_lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.warmup_steps = state["warmup_steps"]
        self.base_lr = state["base_lr"]
        self._step = state["_step"]
        self._current_lr = state["_current_lr"]
        self._set_lr(self._current_lr)


# ---------------------------------------------------------------------------
# CosineWithWarmup — cosine annealing with linear warmup + optional restarts
# ---------------------------------------------------------------------------


class CosineWithWarmup:
    """Cosine annealing with linear warmup.  Supports warm restarts (SGDR)."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        n_cycles: int = 1,
    ):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if total_steps <= warmup_steps:
            raise ValueError("total_steps must be > warmup_steps")
        if n_cycles < 1:
            raise ValueError("n_cycles must be >= 1")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.n_cycles = n_cycles

        # peak LR is taken from the first param group at construction time
        self.peak_lr: float = optimizer.param_groups[0]["lr"]
        self.min_lr: float = self.peak_lr * min_lr_ratio

        self._step = 0
        self._current_lr = 0.0 if warmup_steps > 0 else self.peak_lr
        self._set_lr(self._current_lr)

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self) -> None:
        self._step += 1
        s = self._step

        if s <= self.warmup_steps:
            # linear warmup
            if self.warmup_steps == 0:
                self._current_lr = self.peak_lr
            else:
                self._current_lr = self.peak_lr * (s / self.warmup_steps)
        else:
            # cosine decay over (total_steps - warmup_steps), split into n_cycles.
            # Within each cycle, progress goes 0.0 → 1.0 so that the last step of
            # every cycle lands exactly at min_lr and the first step of every new
            # cycle starts back at peak_lr (warm restart / SGDR behaviour).
            decay_steps = self.total_steps - self.warmup_steps
            cycle_steps = decay_steps / self.n_cycles
            pos = s - self.warmup_steps  # 1-based position in decay phase
            # 0-indexed position within the current cycle
            cycle_idx = int((pos - 1) / cycle_steps)
            cycle_pos = (pos - 1) - cycle_idx * cycle_steps
            if cycle_steps > 1:
                progress = cycle_pos / (cycle_steps - 1)
            else:
                progress = 1.0
            progress = min(progress, 1.0)
            cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            self._current_lr = self.min_lr + (self.peak_lr - self.min_lr) * cos_val

        self._set_lr(self._current_lr)

    def get_lr(self) -> float:
        return self._current_lr


# ---------------------------------------------------------------------------
# WSDScheduler — Warmup-Stable-Decay  (MiniCPM / OLMo style)
# ---------------------------------------------------------------------------


class WSDScheduler:
    """Warmup → Stable → Decay (cosine) scheduler."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        peak_lr: float,
        min_lr: float = 0.0,
    ):
        if warmup_steps < 0 or stable_steps < 0 or decay_steps < 0:
            raise ValueError("all step counts must be >= 0")
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr

        self._step = 0
        self._current_lr = 0.0
        self._set_lr(0.0)

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def current_phase(self) -> str:
        s = self._step
        if s <= self.warmup_steps:
            return "warmup"
        elif s <= self.warmup_steps + self.stable_steps:
            return "stable"
        else:
            return "decay"

    def step(self) -> None:
        self._step += 1
        s = self._step

        if s <= self.warmup_steps:
            if self.warmup_steps == 0:
                self._current_lr = self.peak_lr
            else:
                self._current_lr = self.peak_lr * (s / self.warmup_steps)
        elif s <= self.warmup_steps + self.stable_steps:
            self._current_lr = self.peak_lr
        else:
            # cosine decay
            decay_pos = s - self.warmup_steps - self.stable_steps
            if self.decay_steps == 0:
                self._current_lr = self.min_lr
            else:
                progress = min(decay_pos / self.decay_steps, 1.0)
                cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
                self._current_lr = self.min_lr + (self.peak_lr - self.min_lr) * cos_val

        self._set_lr(self._current_lr)


# ---------------------------------------------------------------------------
# InverseSquareRootScheduler — Vaswani 2017 original transformer schedule
# ---------------------------------------------------------------------------


class InverseSquareRootScheduler:
    """lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))."""

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be > 0")
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0
        self._current_lr = 0.0
        self._set_lr(0.0)

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self) -> None:
        self._step += 1
        s = self._step
        scale = (self.d_model**-0.5) * min(
            s**-0.5,
            s * (self.warmup_steps**-1.5),
        )
        self._current_lr = scale
        self._set_lr(self._current_lr)

    def get_lr(self) -> float:
        return self._current_lr


# ---------------------------------------------------------------------------
# CyclicLRScheduler — triangular / triangular2 / exp_range  (Smith 2017)
# ---------------------------------------------------------------------------


class CyclicLRScheduler:
    """Triangular cyclic LR.

    modes
    -----
    'triangular'  — constant amplitude each cycle
    'triangular2' — amplitude halved each cycle
    'exp_range'   — amplitude scaled by gamma^iteration
    """

    def __init__(
        self,
        optimizer,
        base_lr: float,
        max_lr: float,
        step_size: int = 2000,
        mode: str = "triangular",
        gamma: float = 0.99994,
    ):
        if base_lr >= max_lr:
            raise ValueError("base_lr must be < max_lr")
        if mode not in ("triangular", "triangular2", "exp_range"):
            raise ValueError(f"Unknown mode: {mode}")
        if step_size <= 0:
            raise ValueError("step_size must be > 0")

        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

        self._step = 0
        self._current_lr = base_lr
        self._set_lr(base_lr)

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def cycle_progress(self) -> float:
        """Position within current half-cycle [0, 1]."""
        cycle_len = 2 * self.step_size
        pos = self._step % cycle_len
        return pos / cycle_len

    def step(self) -> None:
        self._step += 1
        cycle_len = 2 * self.step_size
        cycle = math.floor(1 + self._step / cycle_len)
        x = abs(self._step / self.step_size - 2 * cycle + 1)
        # x goes 0→1→0 over one full cycle (two half-cycles)
        x = max(0.0, 1.0 - x)

        if self.mode == "triangular":
            scale = 1.0
        elif self.mode == "triangular2":
            scale = 1.0 / (2.0 ** (cycle - 1))
        else:  # exp_range
            scale = self.gamma ** (self._step)

        self._current_lr = self.base_lr + (self.max_lr - self.base_lr) * x * scale
        self._set_lr(self._current_lr)

    def get_lr(self) -> float:
        return self._current_lr


# ---------------------------------------------------------------------------
# LRSchedulerFactory
# ---------------------------------------------------------------------------


class LRSchedulerFactory:
    """Create schedulers by name and produce LR traces for inspection."""

    _REGISTRY = {
        "warmup": WarmupScheduler,
        "cosine_warmup": CosineWithWarmup,
        "wsd": WSDScheduler,
        "inv_sqrt": InverseSquareRootScheduler,
        "cyclic": CyclicLRScheduler,
    }

    def __init__(self):
        pass

    def create(self, name: str, optimizer, **kwargs):
        """Return a scheduler instance for *name*."""
        if name not in self._REGISTRY:
            raise ValueError(f"Unknown scheduler '{name}'. Available: {sorted(self._REGISTRY)}")
        return self._REGISTRY[name](optimizer, **kwargs)

    def plot_schedule(self, scheduler, n_steps: int) -> list:
        """Step *scheduler* n_steps times and return list of LR values."""
        lrs = []
        for _ in range(n_steps):
            scheduler.step()
            # retrieve LR from scheduler if available, else from optimizer
            if hasattr(scheduler, "get_lr"):
                lrs.append(scheduler.get_lr())
            else:
                lrs.append(scheduler.optimizer.param_groups[0]["lr"])
        return lrs
