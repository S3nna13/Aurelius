"""Action trajectory recording and replay for the Aurelius computer_use surface.

Inspired by OpenDevin/OpenDevin (Apache-2.0, browser tool), MoonshotAI/Kimi-Dev
(Apache-2.0, patch synthesis), WebArena trajectory replay, clean-room reimplementation.

No playwright, pyautogui, selenium, or OS accessibility API imports anywhere.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.computer_use.gui_action import ActionType, GUIAction
from src.computer_use.screen_parser import ScreenSnapshot

if TYPE_CHECKING:
    from src.computer_use.browser_driver import BrowserDriver


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TrajectoryError(Exception):
    """Raised when a trajectory operation is invalid."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryStep:
    """A single recorded step in a GUI trajectory."""

    step_id: int
    action: GUIAction
    snapshot_before: ScreenSnapshot | None = None
    snapshot_after: ScreenSnapshot | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A complete sequence of GUI actions toward a goal."""

    trajectory_id: str
    goal: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    success: bool | None = None
    total_duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class TrajectoryRecorder:
    """Records GUI actions into a Trajectory.

    Usage::

        recorder = TrajectoryRecorder(goal="Open settings")
        step = recorder.record_step(action, snapshot_before, snapshot_after)
        trajectory = recorder.finalize(success=True)
    """

    def __init__(self, goal: str = "", trajectory_id: str | None = None) -> None:
        self._goal: str = goal
        self._trajectory_id: str = trajectory_id or str(uuid.uuid4())
        self._steps: list[TrajectoryStep] = []
        self._start_time: float = time.time()

    def record_step(
        self,
        action: GUIAction,
        snapshot_before: ScreenSnapshot | None = None,
        snapshot_after: ScreenSnapshot | None = None,
    ) -> TrajectoryStep:
        """Append a step to the current recording.

        Parameters
        ----------
        action:
            The GUI action that was (or will be) executed.
        snapshot_before:
            Screen state immediately before the action.
        snapshot_after:
            Screen state immediately after the action.

        Returns
        -------
        TrajectoryStep
            The newly created step (step_id auto-incremented from 0).
        """
        step = TrajectoryStep(
            step_id=len(self._steps),
            action=action,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )
        self._steps.append(step)
        return step

    def finalize(self, success: bool) -> Trajectory:
        """Close the recording and return the completed Trajectory.

        Parameters
        ----------
        success:
            Whether the trajectory achieved its goal.

        Returns
        -------
        Trajectory
        """
        duration = time.time() - self._start_time
        return Trajectory(
            trajectory_id=self._trajectory_id,
            goal=self._goal,
            steps=list(self._steps),
            success=success,
            total_duration_s=duration,
        )

    def reset(self) -> None:
        """Clear all recorded steps and restart the timer."""
        self._steps = []
        self._start_time = time.time()
        self._trajectory_id = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Replayer
# ---------------------------------------------------------------------------


class TrajectoryReplayer:
    """Replays and validates recorded trajectories.

    Replay dispatches each step's action to a :class:`BrowserDriver`.

    - ``ActionType.CLICK`` → ``driver.click(step.action.target_selector)``
    - ``ActionType.TYPE``  → ``driver.type_text(step.action.target_selector, step.action.value)``
    - ``ActionType.SCROLL`` / ``ActionType.KEY`` → no-op in stub

    All other action types are treated as no-ops to remain forward compatible.
    """

    def replay(
        self,
        trajectory: Trajectory,
        driver: BrowserDriver,
    ) -> list[str]:
        """Replay *trajectory* against *driver*.

        Parameters
        ----------
        trajectory:
            The trajectory to replay.
        driver:
            A :class:`BrowserDriver` instance to dispatch actions to.

        Returns
        -------
        list[str]
            Status strings, one per step, e.g. ``["step-0: ok", "step-1: ok"]``.

        Raises
        ------
        TrajectoryError
            If an unexpected driver error occurs during replay.
        """
        statuses: list[str] = []

        for step in trajectory.steps:
            label = f"step-{step.step_id}"
            action = step.action

            try:
                if action.action_type == ActionType.CLICK:
                    selector = action.target_selector or ""
                    driver.click(selector)
                elif action.action_type == ActionType.TYPE:
                    selector = action.target_selector or ""
                    text = action.value or ""
                    driver.type_text(selector, text)
                elif action.action_type in (ActionType.SCROLL, ActionType.KEY):
                    # No-op in stub — real drivers would handle these.
                    pass
                elif action.action_type == ActionType.WAIT:
                    # No-op — respect real timing in a live driver.
                    pass
                else:
                    # Forward-compatible: unknown types are no-ops.
                    pass

                statuses.append(f"{label}: ok")

            except Exception as exc:  # noqa: BLE001
                statuses.append(f"{label}: error — {exc}")

        return statuses

    def verify(self, trajectory: Trajectory) -> tuple[bool, list[str]]:
        """Validate a trajectory for structural correctness.

        Checks:

        1. The trajectory has at least one step.
        2. No step has both ``action`` and ``snapshot_before`` as ``None``
           (a step with no action and no context is meaningless).

        Parameters
        ----------
        trajectory:
            The trajectory to validate.

        Returns
        -------
        tuple[bool, list[str]]
            ``(all_valid, issues)`` where ``issues`` is empty when ``all_valid``
            is ``True``.
        """
        issues: list[str] = []

        if len(trajectory.steps) < 1:
            issues.append("Trajectory has no steps.")
            return False, issues

        for step in trajectory.steps:
            # action is always set via the dataclass, but guard against None
            # in case a caller manually constructs a step.
            if step.action is None and step.snapshot_before is None:
                issues.append(f"Step {step.step_id}: both action and snapshot_before are None.")

        return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRAJECTORY_REGISTRY: dict[str, Trajectory] = {}
