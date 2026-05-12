"""Plugin system shim for the aurelius namespace."""

from __future__ import annotations

from typing import Any


class PluginManager:
    """Minimal stub — real implementation lives in agent.plugin_loader."""


BUILTIN_PLUGINS: dict[str, Any] = {}
PLUGIN_MANAGER = PluginManager()

__all__ = ["PluginManager", "BUILTIN_PLUGINS", "PLUGIN_MANAGER"]
