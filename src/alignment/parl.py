"""PARL reward — Kimi K2.5 §3.3 (arXiv:2602.02276).

r_PARL(x, y) = λ₁·r_parallel + λ₂·r_finish + r_perf(x, y)

r_parallel  = normalized count of active sub-agents spawned (incentivizes parallelism)
r_finish    = sub-agent completion rate (prevents spurious parallelism with no results)
r_perf      = task-level outcome reward (binary or graded)
λ₁, λ₂     = annealed to 0 over training (start 1.0 → 0.0 linearly by total_steps)

Key insight: λ annealing means early training focuses on learning to parallelize,
late training focuses purely on task performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class AnnealedLambda:
    """Linearly anneals a scalar weight from `start` → 0 over `total_steps`.

    After step >= total_steps the value clamps to 0 (never goes negative).
    """

    def __init__(self, start: float = 1.0, total_steps: int = 10_000) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        self.start = start
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        """Return the annealed λ value at `step`."""
        return self.start * max(0.0, 1.0 - step / self.total_steps)


@dataclass
class PARLReward:
    """PARL composite reward (Kimi K2.5 §3.3).

    Computes:
        r_PARL = r_perf + ann(λ₁, step) * r_parallel + ann(λ₂, step) * r_finish

    All tensor inputs must be 1-D with the same batch size B.

    Attributes:
        lambda1:      Initial weight for r_parallel term (anneals to 0).
        lambda2:      Initial weight for r_finish term (anneals to 0).
        total_steps:  Number of steps over which λ annealing is applied.
    """

    lambda1: float = 1.0
    lambda2: float = 1.0
    total_steps: int = 10_000

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {self.total_steps}")

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def __call__(
        self,
        r_perf: torch.Tensor,
        r_parallel: torch.Tensor,
        r_finish: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor:
        """Compute the PARL reward for a batch.

        Args:
            r_perf:     (B,) task-level outcome reward (binary or graded).
            r_parallel: (B,) normalized count of active sub-agents spawned.
            r_finish:   (B,) sub-agent completion rate.
            step:       Current training step (used for λ annealing).

        Returns:
            (B,) combined PARL reward tensor (same dtype/device as r_perf).
        """
        ann1 = self.lambda1 * max(0.0, 1.0 - step / self.total_steps)
        ann2 = self.lambda2 * max(0.0, 1.0 - step / self.total_steps)
        return r_perf + ann1 * r_parallel + ann2 * r_finish

    # ------------------------------------------------------------------
    # Convenience: annealed λ values at a given step
    # ------------------------------------------------------------------

    def annealed_lambda1(self, step: int) -> float:
        """Return the annealed λ₁ at `step`."""
        return self.lambda1 * max(0.0, 1.0 - step / self.total_steps)

    def annealed_lambda2(self, step: int) -> float:
        """Return the annealed λ₂ at `step`."""
        return self.lambda2 * max(0.0, 1.0 - step / self.total_steps)
