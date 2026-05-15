"""Aurelius v2 CUA Driver Base — abstract interface for all CUA drivers.

Modes:
- verifier_only: validates actions without executing
- remote_driver: sends action plan to remote desktop
- browser_only: browser automation via CDP
- local_basic: click/type/scroll/key with strict verifier
- local_full: screenshot + AX tree + background driver
- multimodal_full: full CUA with vision/document/GUI reasoning
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any


class CUAMode(enum.StrEnum):
    VERIFIER_ONLY = "verifier_only"
    REMOTE_DRIVER = "remote_driver"
    BROWSER_ONLY = "browser_only"
    LOCAL_BASIC = "local_basic"
    LOCAL_FULL = "local_full"
    MULTIMODAL_FULL = "multimodal_full"


@dataclass
class CUAAction:
    """A single CUA action."""
    action_type: str  # click, type, scroll, key, drag, set_value
    element_ref: str = ""
    coordinate: tuple[int, int] | None = None
    text: str = ""
    keys: str = ""
    direction: str = ""
    amount: int = 0
    app: str = ""
    modifiers: list[str] = field(default_factory=list)
    button: str = "left"


@dataclass
class CUAObservation:
    """An observation from the desktop/browser."""
    screenshot_path: str = ""
    ax_tree: str = ""
    ocr_text: str = ""
    focused_element: str = ""
    app_name: str = ""
    window_title: str = ""
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CUAResult:
    """Result of executing a CUA action."""
    success: bool
    action: CUAAction
    before_observation: CUAObservation | None = None
    after_observation: CUAObservation | None = None
    error: str = ""
    verification: dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    block_reason: str = ""


class ComputerUseDriver(abc.ABC):
    """Abstract base for all CUA drivers.

    Every driver must:
    1. Capture an observation (screenshot + AX tree)
    2. Execute an action
    3. Verify the action was safe and successful
    4. Log the action to the audit trail
    """

    def __init__(self, mode: CUAMode) -> None:
        self.mode = mode

    @abc.abstractmethod
    def capture(self) -> CUAObservation:
        """Capture the current desktop/browser state."""
        ...

    @abc.abstractmethod
    def execute(self, action: CUAAction) -> CUAResult:
        """Execute a CUA action."""
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        ...

    def run_task(self, actions: list[CUAAction]) -> list[CUAResult]:
        """Run a sequence of actions, verifying each one."""
        results = []
        for action in actions:
            before = self.capture()
            result = self.execute(action)
            result.before_observation = before
            result.after_observation = self.capture()
            results.append(result)
        return results
