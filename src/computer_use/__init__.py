"""Aurelius computer_use surface — screen parsing, GUI action prediction, and action verification.

Inspired by OpenDevin/OpenDevin (browser tool), MoonshotAI/Kimi-Dev (coding agent loop),
Apache-2.0, clean-room reimplementation.

No playwright, pyautogui, or OS accessibility API imports anywhere in this package.
All external integrations are behind abstract interfaces.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.computer_use.screen_parser import SCREEN_PARSER_REGISTRY  # noqa: F401
    from src.computer_use.gui_action import GUI_ACTION_REGISTRY  # noqa: F401
    from src.computer_use.browser_driver import (  # noqa: F401
        BrowserDriver,
        StubBrowserDriver,
        BrowserDriverError,
        BROWSER_DRIVER_REGISTRY,
    )
    from src.computer_use.trajectory_replay import (  # noqa: F401
        TrajectoryRecorder,
        TrajectoryReplayer,
        Trajectory,
        TRAJECTORY_REGISTRY,
    )
    from src.computer_use.webarena_eval import (  # noqa: F401
        WebArenaHarness,
        WebTask,
        TaskResult,
        SuccessEvaluator,
        WebArenaError,
        WEBARENA_HARNESS_REGISTRY,
        WEBARENA_DEFAULT_TASKS,
    )
    from src.computer_use.action_planner import (  # noqa: F401
        ActionPlanner,
        ActionPlan,
        PlannedAction,
        ActionType,
        COMPUTER_USE_REGISTRY as _ACTION_PLANNER_REGISTRY,
    )
    from src.computer_use.screen_state_tracker import (  # noqa: F401
        ScreenStateTracker,
        ScreenState,
        ScreenRegion,
        COMPUTER_USE_REGISTRY as _SCREEN_STATE_REGISTRY,
    )

__all__ = [
    "SCREEN_PARSER_REGISTRY",
    "GUI_ACTION_REGISTRY",
    "COMPUTER_USE_REGISTRY",
    "screen_parser",
    "gui_action",
    "action_verifier",
    "browser_driver",
    "trajectory_replay",
    "webarena_eval",
    "action_planner",
    "screen_state_tracker",
    # action_planner exports
    "ActionPlanner",
    "ActionPlan",
    "PlannedAction",
    "ActionType",
    # screen_state_tracker exports
    "ScreenStateTracker",
    "ScreenState",
    "ScreenRegion",
    # browser_driver exports
    "BrowserDriver",
    "StubBrowserDriver",
    "BrowserDriverError",
    "BROWSER_DRIVER_REGISTRY",
    # trajectory_replay exports
    "TrajectoryRecorder",
    "TrajectoryReplayer",
    "Trajectory",
    "TRAJECTORY_REGISTRY",
    # webarena_eval exports
    "WebArenaHarness",
    "WebTask",
    "TaskResult",
    "SuccessEvaluator",
    "WebArenaError",
    "WEBARENA_HARNESS_REGISTRY",
    "WEBARENA_DEFAULT_TASKS",
]

_SUBMODULES = ("screen_parser", "gui_action", "action_verifier", "browser_driver", "trajectory_replay", "webarena_eval", "action_planner", "screen_state_tracker")

_REGISTRY_ATTRS: dict[str, tuple[str, str]] = {
    "SCREEN_PARSER_REGISTRY": ("screen_parser", "SCREEN_PARSER_REGISTRY"),
    "GUI_ACTION_REGISTRY": ("gui_action", "GUI_ACTION_REGISTRY"),
    "BROWSER_DRIVER_REGISTRY": ("browser_driver", "BROWSER_DRIVER_REGISTRY"),
    "TRAJECTORY_REGISTRY": ("trajectory_replay", "TRAJECTORY_REGISTRY"),
    "WEBARENA_HARNESS_REGISTRY": ("webarena_eval", "WEBARENA_HARNESS_REGISTRY"),
    "WEBARENA_DEFAULT_TASKS": ("webarena_eval", "WEBARENA_DEFAULT_TASKS"),
    "COMPUTER_USE_REGISTRY": ("action_planner", "COMPUTER_USE_REGISTRY"),
}

_CLASS_ATTRS: dict[str, tuple[str, str]] = {
    "BrowserDriver": ("browser_driver", "BrowserDriver"),
    "StubBrowserDriver": ("browser_driver", "StubBrowserDriver"),
    "BrowserDriverError": ("browser_driver", "BrowserDriverError"),
    "TrajectoryRecorder": ("trajectory_replay", "TrajectoryRecorder"),
    "TrajectoryReplayer": ("trajectory_replay", "TrajectoryReplayer"),
    "Trajectory": ("trajectory_replay", "Trajectory"),
    "WebArenaHarness": ("webarena_eval", "WebArenaHarness"),
    "WebTask": ("webarena_eval", "WebTask"),
    "TaskResult": ("webarena_eval", "TaskResult"),
    "SuccessEvaluator": ("webarena_eval", "SuccessEvaluator"),
    "WebArenaError": ("webarena_eval", "WebArenaError"),
    # action_planner
    "ActionPlanner": ("action_planner", "ActionPlanner"),
    "ActionPlan": ("action_planner", "ActionPlan"),
    "PlannedAction": ("action_planner", "PlannedAction"),
    "ActionType": ("action_planner", "ActionType"),
    # screen_state_tracker
    "ScreenStateTracker": ("screen_state_tracker", "ScreenStateTracker"),
    "ScreenState": ("screen_state_tracker", "ScreenState"),
    "ScreenRegion": ("screen_state_tracker", "ScreenRegion"),
}


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f"src.computer_use.{name}")
        globals()[name] = module
        return module
    if name in _REGISTRY_ATTRS:
        mod_name, attr = _REGISTRY_ATTRS[name]
        mod = import_module(f"src.computer_use.{mod_name}")
        return getattr(mod, attr)
    if name in _CLASS_ATTRS:
        mod_name, attr = _CLASS_ATTRS[name]
        mod = import_module(f"src.computer_use.{mod_name}")
        return getattr(mod, attr)
    raise AttributeError(f"module 'src.computer_use' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | set(_SUBMODULES)
        | set(_REGISTRY_ATTRS)
        | set(_CLASS_ATTRS)
    )
