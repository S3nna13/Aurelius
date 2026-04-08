"""Early stopping and reduce-on-plateau utilities for training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Mode(Enum):
    MIN = "min"  # lower is better (loss, perplexity)
    MAX = "max"  # higher is better (accuracy)


@dataclass
class EarlyStoppingConfig:
    patience: int = 5  # steps without improvement before triggering
    min_delta: float = 1e-4  # minimum change to count as improvement
    mode: Mode = Mode.MIN
    reduce_factor: float = 0.5  # LR scale when plateau detected
    reduce_patience: int = 3  # separate patience for LR reduction


class EarlyStopping:
    """Tracks a metric and signals when training should stop.

    Usage:
        es = EarlyStopping(EarlyStoppingConfig(patience=5))
        for step in range(total_steps):
            val_loss = trainer.validate()
            if es.step(val_loss):
                print("Early stopping triggered")
                break
    """

    def __init__(self, cfg: EarlyStoppingConfig | None = None) -> None:
        self.cfg = cfg or EarlyStoppingConfig()
        self.best: float | None = None
        self.steps_without_improvement: int = 0
        self.history: list[float] = []
        self.stopped: bool = False

    def step(self, metric: float) -> bool:
        """Record metric. Returns True if should stop."""
        self.history.append(metric)

        if self.best is None or self.is_improvement(metric):
            self.best = metric
            self.steps_without_improvement = 0
            return False

        self.steps_without_improvement += 1

        if self.steps_without_improvement >= self.cfg.patience:
            self.stopped = True
            return True

        return False

    def is_improvement(self, metric: float) -> bool:
        """Return True if metric is an improvement over best."""
        if self.best is None:
            return True
        if self.cfg.mode is Mode.MIN:
            return metric < self.best - self.cfg.min_delta
        return metric > self.best + self.cfg.min_delta

    @property
    def should_stop(self) -> bool:
        return self.stopped


class ReduceOnPlateau:
    """Suggests LR scale factor when metric plateaus.

    Usage:
        rop = ReduceOnPlateau(EarlyStoppingConfig(reduce_patience=3, reduce_factor=0.5))
        lr_scale = rop.step(val_loss)  # returns 1.0 normally, 0.5 on plateau
    """

    def __init__(self, cfg: EarlyStoppingConfig | None = None) -> None:
        self.cfg = cfg or EarlyStoppingConfig()
        self.best: float | None = None
        self.steps_without_improvement: int = 0
        self.history: list[float] = []

    def _is_improvement(self, metric: float) -> bool:
        if self.best is None:
            return True
        if self.cfg.mode is Mode.MIN:
            return metric < self.best - self.cfg.min_delta
        return metric > self.best + self.cfg.min_delta

    def step(self, metric: float) -> float:
        """Record metric. Returns lr_scale (1.0 = no change, reduce_factor = reduce LR)."""
        self.history.append(metric)

        if self.best is None or self._is_improvement(metric):
            self.best = metric
            self.steps_without_improvement = 0
            return 1.0

        self.steps_without_improvement += 1

        if self.steps_without_improvement >= self.cfg.reduce_patience:
            self.steps_without_improvement = 0
            return self.cfg.reduce_factor

        return 1.0
