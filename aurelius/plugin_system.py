from typing import Any

from src.agent.plugin_system import Plugin, PluginManager

BUILTIN_PLUGINS: dict[str, Any] = {}
PLUGIN_MANAGER = PluginManager()

__all__ = ["Plugin", "PluginManager", "BUILTIN_PLUGINS", "PLUGIN_MANAGER"]
