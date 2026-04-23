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

__all__ = [
    "SCREEN_PARSER_REGISTRY",
    "GUI_ACTION_REGISTRY",
    "screen_parser",
    "gui_action",
    "action_verifier",
]

_SUBMODULES = ("screen_parser", "gui_action", "action_verifier")


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f"src.computer_use.{name}")
        globals()[name] = module
        return module
    if name == "SCREEN_PARSER_REGISTRY":
        mod = import_module("src.computer_use.screen_parser")
        return mod.SCREEN_PARSER_REGISTRY
    if name == "GUI_ACTION_REGISTRY":
        mod = import_module("src.computer_use.gui_action")
        return mod.GUI_ACTION_REGISTRY
    raise AttributeError(f"module 'src.computer_use' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULES) | {"SCREEN_PARSER_REGISTRY", "GUI_ACTION_REGISTRY"})
