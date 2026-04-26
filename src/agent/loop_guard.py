"""Loop-guard module for the Aurelius agent surface.

Detects pathological agent behaviours:
  - Exceeding a maximum step budget (MAX_STEPS)
  - Making no measurable progress over a window of steps (NO_PROGRESS)
  - Repeating the same action too many times (REPEATED_ACTION)
  - Cycling through a small set of actions (CYCLE_DETECTED)
"""

from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Enumerations & data classes
# ---------------------------------------------------------------------------


class StallReason(StrEnum):
    NO_PROGRESS = "NO_PROGRESS"
    REPEATED_ACTION = "REPEATED_ACTION"
    MAX_STEPS = "MAX_STEPS"
    CYCLE_DETECTED = "CYCLE_DETECTED"


@dataclass
class LoopGuardConfig:
    max_no_progress: int = 5
    max_steps: int = 50
    max_action_repeats: int = 3
    history_window: int = 10


@dataclass
class LoopGuardResult:
    should_terminate: bool
    reason: StallReason | None
    steps_taken: int
    message: str


# ---------------------------------------------------------------------------
# Guard implementation
# ---------------------------------------------------------------------------


class AgentLoopGuard:
    """Stateful guard that detects agent loops and stalls."""

    def __init__(self, config: LoopGuardConfig | None = None) -> None:
        self._config = config or LoopGuardConfig()
        self._steps: int = 0
        self._progress_signals: collections.deque[float] = collections.deque(
            maxlen=self._config.max_no_progress
        )
        self._action_hashes: collections.deque[str] = collections.deque(
            maxlen=self._config.history_window
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _action_hash(self, action: dict) -> str:
        """Return a 16-char hex digest of the JSON-serialised action."""
        payload = json.dumps(action, sort_keys=True).encode()
        return hashlib.sha256(payload, usedforsecurity=False).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        action: dict,
        result: dict,  # noqa: ARG002  – reserved for future use
        progress_signal: float = 0.0,
    ) -> LoopGuardResult:
        """Record one agent step and return a termination verdict."""
        self._steps += 1

        # 1. Hard step limit
        if self._steps >= self._config.max_steps:
            return LoopGuardResult(
                should_terminate=True,
                reason=StallReason.MAX_STEPS,
                steps_taken=self._steps,
                message=f"Maximum step limit ({self._config.max_steps}) reached.",
            )

        # 2. No-progress detection
        self._progress_signals.append(progress_signal)
        if len(self._progress_signals) == self._config.max_no_progress and all(
            s < 0.01 for s in self._progress_signals
        ):
            return LoopGuardResult(
                should_terminate=True,
                reason=StallReason.NO_PROGRESS,
                steps_taken=self._steps,
                message=(f"No progress detected over last {self._config.max_no_progress} steps."),
            )

        # 3. Repeated-action / cycle detection
        h = self._action_hash(action)
        repeat_count = self._action_hashes.count(h)

        if repeat_count >= self._config.max_action_repeats:
            return LoopGuardResult(
                should_terminate=True,
                reason=StallReason.REPEATED_ACTION,
                steps_taken=self._steps,
                message=(
                    f"Action repeated {repeat_count + 1} times "
                    f"within the last {self._config.history_window} steps."
                ),
            )

        if repeat_count > 0:
            # Hash appears in the window but not enough for REPEATED_ACTION
            self._action_hashes.append(h)
            return LoopGuardResult(
                should_terminate=True,
                reason=StallReason.CYCLE_DETECTED,
                steps_taken=self._steps,
                message="Cycle detected: action hash seen again within history window.",
            )

        self._action_hashes.append(h)

        return LoopGuardResult(
            should_terminate=False,
            reason=None,
            steps_taken=self._steps,
            message="OK",
        )

    def reset(self) -> None:
        """Reset all state so the guard can be reused for a new episode."""
        self._steps = 0
        self._progress_signals.clear()
        self._action_hashes.clear()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AGENT_REGISTRY: dict[str, Any] = {
    "loop_guard": AgentLoopGuard,
}
