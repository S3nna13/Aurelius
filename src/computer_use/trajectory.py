"""Aurelius v2 CUA Trajectory — recording, replay, and evaluation of CUA action sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.computer_use.driver_base import CUAAction, CUAResult


@dataclass
class CUATrajectory:
    """A recorded CUA trajectory."""
    task_id: str
    task_description: str
    actions: list[CUAAction]
    results: list[CUAResult]
    total_steps: int = 0
    success: bool = False
    duration_seconds: float = 0.0

    @property
    def failed_steps(self) -> list[int]:
        return [i for i, r in enumerate(self.results) if not r.success]

    @property
    def blocked_steps(self) -> list[int]:
        return [i for i, r in enumerate(self.results) if r.blocked]


class CUATrajectoryRecorder:
    """Records CUA action trajectories for replay and evaluation."""

    def __init__(self) -> None:
        self._trajectories: list[CUATrajectory] = []

    def record(self, trajectory: CUATrajectory) -> None:
        trajectory.total_steps = len(trajectory.actions)
        trajectory.success = all(r.success for r in trajectory.results)
        self._trajectories.append(trajectory)

    def get(self, task_id: str) -> CUATrajectory | None:
        for t in self._trajectories:
            if t.task_id == task_id:
                return t
        return None

    def list_trajectories(self) -> list[CUATrajectory]:
        return list(self._trajectories)


class CUATrajectoryReplay:
    """Replays recorded CUA trajectories for evaluation."""

    def replay(self, trajectory: CUATrajectory) -> list[dict[str, Any]]:
        """Replay a trajectory step-by-step, returning comparison results."""
        results = []
        for i, (action, result) in enumerate(zip(trajectory.actions, trajectory.results)):
            results.append({
                "step": i,
                "action_type": action.action_type,
                "expected_success": result.success,
                "was_blocked": result.blocked,
                "block_reason": result.block_reason,
            })
        return results
