"""LR range test (Leslie Smith, 2015).

Exponentially sweeps the learning rate from ``lr_start`` to ``lr_end``
over a fixed number of training steps, records the loss at each step,
and identifies the best LR region (steepest descent of the smoothed
loss curve) along with an optional divergence point.

Pure PyTorch + stdlib.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class LRRangeTestResult:
    lrs: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    best_lr: float = 0.0
    best_lr_div_10: float = 0.0
    divergence_lr: float | None = None


class LRRangeTest:
    """LR range test over an exponential LR schedule.

    Parameters
    ----------
    train_step_fn:
        Callable taking the current learning rate and returning the
        scalar training loss at that step.
    lr_start:
        Lower bound of the LR sweep (must be > 0).
    lr_end:
        Upper bound of the LR sweep (must be > ``lr_start``).
    num_steps:
        Number of sweep steps (must be > 1).
    smooth_factor:
        EMA factor in ``[0, 1)`` used to smooth the loss curve.
    divergence_threshold:
        Early-stop threshold: if the smoothed loss exceeds
        ``smooth_factor * min_loss + divergence_threshold`` the sweep
        stops. A value of ``0`` disables the check.
    """

    def __init__(
        self,
        train_step_fn: Callable[[float], float],
        lr_start: float = 1e-7,
        lr_end: float = 1.0,
        num_steps: int = 100,
        smooth_factor: float = 0.98,
        divergence_threshold: float = 4.0,
    ) -> None:
        if not callable(train_step_fn):
            raise TypeError("train_step_fn must be callable")
        if lr_start <= 0:
            raise ValueError("lr_start must be > 0")
        if lr_end <= lr_start:
            raise ValueError("lr_end must be > lr_start")
        if num_steps <= 1:
            raise ValueError("num_steps must be > 1")
        if not (0.0 <= smooth_factor < 1.0):
            raise ValueError("smooth_factor must be in [0, 1)")
        if divergence_threshold < 0:
            raise ValueError("divergence_threshold must be >= 0")

        self.train_step_fn = train_step_fn
        self.lr_start = float(lr_start)
        self.lr_end = float(lr_end)
        self.num_steps = int(num_steps)
        self.smooth_factor = float(smooth_factor)
        self.divergence_threshold = float(divergence_threshold)

    def _lr_schedule(self) -> list[float]:
        # Geometric (exponential) spacing between lr_start and lr_end.
        log_start = math.log(self.lr_start)
        log_end = math.log(self.lr_end)
        step = (log_end - log_start) / (self.num_steps - 1)
        return [math.exp(log_start + i * step) for i in range(self.num_steps)]

    def run(self) -> LRRangeTestResult:
        lrs_schedule = self._lr_schedule()

        lrs: list[float] = []
        losses: list[float] = []
        smoothed_losses: list[float] = []

        avg = 0.0
        min_smoothed = math.inf
        divergence_lr: float | None = None

        for i, lr in enumerate(lrs_schedule):
            loss = float(self.train_step_fn(lr))

            # Exponential moving average on the raw loss, bias-corrected
            # so early steps aren't biased toward zero.
            if i == 0:
                avg = loss
            else:
                avg = self.smooth_factor * avg + (1.0 - self.smooth_factor) * loss
            smoothed = avg

            lrs.append(lr)
            losses.append(loss)
            smoothed_losses.append(smoothed)

            if smoothed < min_smoothed:
                min_smoothed = smoothed

            # Divergence check: only trigger if threshold > 0.
            if self.divergence_threshold > 0 and math.isfinite(min_smoothed):
                limit = self.smooth_factor * min_smoothed + self.divergence_threshold
                if smoothed > limit:
                    divergence_lr = lr
                    break
            # Also handle NaN/Inf loss as divergence.
            if not math.isfinite(loss) and self.divergence_threshold > 0:
                divergence_lr = lr
                break

        best_lr = self._find_best_lr(lrs, smoothed_losses)
        return LRRangeTestResult(
            lrs=lrs,
            losses=losses,
            best_lr=best_lr,
            best_lr_div_10=best_lr / 10.0,
            divergence_lr=divergence_lr,
        )

    @staticmethod
    def _find_best_lr(lrs: list[float], smoothed: list[float]) -> float:
        """LR at steepest descent (minimum gradient of smoothed loss).

        The gradient is taken in log-LR space, which matches Smith's
        original visualization and is less noisy than wall-clock LR.
        The first few samples are skipped because the EMA is still
        warming up and can produce a spurious steep descent.
        """
        n = len(smoothed)
        if n == 0:
            return 0.0
        if n == 1:
            return lrs[0]

        # Skip the EMA warm-up region: ~10% of samples or >=5 points,
        # whichever is smaller. Never skip so much that nothing remains.
        skip = min(max(5, n // 10), n - 2)
        if skip < 1:
            skip = 1

        best_idx = skip
        best_grad = math.inf
        for i in range(skip, n):
            dlog = math.log(lrs[i]) - math.log(lrs[i - 1])
            if dlog <= 0:
                continue
            grad = (smoothed[i] - smoothed[i - 1]) / dlog
            if grad < best_grad:
                best_grad = grad
                best_idx = i
        return lrs[best_idx]


__all__ = ["LRRangeTest", "LRRangeTestResult"]
