"""Aurelius toolset definitions."""

from __future__ import annotations

from typing import Any

TOOLSETS: dict[str, dict[str, Any]] = {
    "default": {"tools": ["file", "shell", "grep", "search"]},
    "code": {"tools": ["file", "shell", "grep", "edit", "diff", "lint"]},
    "web": {"tools": ["search", "web", "browse"]},
}


def get_toolset(name: str) -> dict[str, Any]:
    """Get a toolset configuration by name."""
    if name not in TOOLSETS:
        return TOOLSETS["default"]
    return TOOLSETS[name]
