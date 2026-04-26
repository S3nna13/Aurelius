"""Pruning scheduler: ramps sparsity levels across training steps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SparsitySchedule(StrEnum):
    CONSTANT = "constant"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    CUBIC = "cubic"


@dataclass(frozen=True)
class PruningConfig:
    """Configuration for a pruning schedule."""

    initial_sparsity: float = 0.0
    target_sparsity: float = 0.9
    begin_step: int = 0
    end_step: int = 1000
    frequency: int = 100


class PruningScheduler:
    """Compute per-step sparsity targets and pruning triggers."""

    def __init__(
        self,
        config: PruningConfig | None = None,
        schedule: SparsitySchedule = SparsitySchedule.CUBIC,
    ) -> None:
        self.config = config or PruningConfig()
        self.schedule = schedule

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sparsity_at(self, step: int) -> float:
        """Return the target sparsity to apply at *step*.

        * Before ``begin_step`` → ``initial_sparsity``
        * After  ``end_step``   → ``target_sparsity``
        * CONSTANT  → ``initial_sparsity`` throughout
        * LINEAR    → linear interpolation between initial and target
        * POLYNOMIAL / CUBIC → cubic ramp  ``s = target - (target - initial) * (1 - pct)^3``
        """
        cfg = self.config
        initial = cfg.initial_sparsity
        final = cfg.target_sparsity

        if self.schedule == SparsitySchedule.CONSTANT:
            return initial

        if step <= cfg.begin_step:
            return initial
        if step >= cfg.end_step:
            return final

        pct = (step - cfg.begin_step) / (cfg.end_step - cfg.begin_step)

        if self.schedule == SparsitySchedule.LINEAR:
            return initial + (final - initial) * pct

        # POLYNOMIAL and CUBIC both use power=3
        # s = final - (final - initial) * (1 - pct)^3
        return final - (final - initial) * (1.0 - pct) ** 3

    def should_prune(self, step: int) -> bool:
        """True when a pruning mask should be applied at *step*."""
        cfg = self.config
        return step >= cfg.begin_step and (step - cfg.begin_step) % cfg.frequency == 0

    def schedule_steps(self, total_steps: int) -> list[tuple[int, float]]:
        """Return ``(step, sparsity)`` pairs for every step where pruning occurs."""
        cfg = self.config
        result: list[tuple[int, float]] = []
        step = cfg.begin_step
        while step <= total_steps:
            if self.should_prune(step):
                result.append((step, self.sparsity_at(step)))
            step += cfg.frequency
        return result


# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

PRUNING_SCHEDULER_REGISTRY: dict[str, type[PruningScheduler]] = {
    "default": PruningScheduler,
}
