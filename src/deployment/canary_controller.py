"""Progressive canary deployment controller with staged traffic promotion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class CanaryStage:
    traffic_pct: float
    min_duration_s: float
    success_threshold: float = 0.99


class CanaryState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


DEFAULT_STAGES: list[CanaryStage] = [
    CanaryStage(5.0, 60.0),
    CanaryStage(25.0, 120.0),
    CanaryStage(50.0, 120.0),
    CanaryStage(100.0, 0.0),
]


class CanaryController:
    """Drives a multi-stage canary deployment based on observed success rates."""

    def __init__(self, stages: list[CanaryStage] | None = None) -> None:
        self._stages: list[CanaryStage] = stages if stages is not None else list(DEFAULT_STAGES)
        self.current_stage_idx: int = 0
        self.state: CanaryState = CanaryState.RUNNING

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def advance(self, success_rate: float) -> CanaryState:
        """Evaluate *success_rate* for the current stage and advance or fail.

        If the success rate meets the current stage's threshold the controller
        moves to the next stage (transitioning to SUCCEEDED when the last stage
        is passed).  A rate below the threshold transitions to FAILED.
        """
        if self.state not in (CanaryState.RUNNING, CanaryState.PENDING):
            return self.state

        current = self._stages[self.current_stage_idx]
        if success_rate < current.success_threshold:
            self.state = CanaryState.FAILED
            return self.state

        # Success — move forward
        if self.current_stage_idx >= len(self._stages) - 1:
            self.state = CanaryState.SUCCEEDED
        else:
            self.current_stage_idx += 1
            self.state = CanaryState.RUNNING

        return self.state

    def rollback(self) -> CanaryState:
        """Immediately mark the deployment as rolled back."""
        self.state = CanaryState.ROLLED_BACK
        return self.state

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------

    def traffic_pct(self) -> float:
        """Return the traffic percentage for the current stage.

        Returns 0.0 when the deployment has failed or been rolled back.
        """
        if self.state in (CanaryState.FAILED, CanaryState.ROLLED_BACK):
            return 0.0
        if self.current_stage_idx >= len(self._stages):
            return 0.0
        return self._stages[self.current_stage_idx].traffic_pct

    def summary(self) -> dict:
        return {
            "state": self.state.value,
            "stage": self.current_stage_idx,
            "traffic_pct": self.traffic_pct(),
            "total_stages": len(self._stages),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CANARY_CONTROLLER_REGISTRY: dict[str, type[CanaryController]] = {
    "default": CanaryController,
}
