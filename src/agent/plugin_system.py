"""Plugin system — plugin lifecycle management with 12 built-in plugins.

Ported from Aurelius's aurelius/plugin_system.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Plugin:
    name: str
    version: str
    description: str
    enabled: bool = True
    hooks: dict[str, list[Callable]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PluginManager:
    """Plugin lifecycle management.

    Supports:
    - Registration and deregistration
    - Enable/disable
    - Hook-based extension points
    - Dependency management
    """

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        builtins: list[tuple[str, str, str]] = [
            ("filesystem", "1.0", "File system operations"),
            ("web", "1.0", "Web fetching and API calls"),
            ("database", "1.0", "Database queries and management"),
            ("search", "1.0", "Search engine integration"),
            ("shell", "1.0", "Shell command execution"),
            ("python", "1.0", "Python code execution"),
            ("git", "1.0", "Git repository operations"),
            ("docker", "1.0", "Docker container management"),
            ("notifications", "1.0", "Notification delivery"),
            ("analytics", "1.0", "Usage analytics and metrics"),
            ("caching", "1.0", "Response and computation caching"),
            ("auth", "1.0", "Authentication and authorization"),
        ]
        for name, version, desc in builtins:
            self._plugins[name] = Plugin(name=name, version=version, description=desc)

    def register(
        self,
        name: str,
        version: str,
        description: str,
        hooks: dict[str, list[Callable]] | None = None,
    ) -> Plugin:
        plugin = Plugin(name=name, version=version, description=description, hooks=hooks or {})
        self._plugins[name] = plugin
        return plugin

    def deregister(self, name: str) -> None:
        self._plugins.pop(name, None)

    def enable(self, name: str) -> None:
        if name in self._plugins:
            self._plugins[name].enabled = True

    def disable(self, name: str) -> None:
        if name in self._plugins:
            self._plugins[name].enabled = False

    def get(self, name: str) -> Plugin | None:
        return self._plugins.get(name)

    def get_enabled(self) -> list[Plugin]:
        return [p for p in self._plugins.values() if p.enabled]

    def get_tools_for_agent(self, agent_id: str) -> list[str]:
        return []

    def execute_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> list[Any]:
        results: list[Any] = []
        for plugin in self._plugins.values():
            if not plugin.enabled:
                continue
            hooks = plugin.hooks.get(hook_name, [])
            for hook_fn in hooks:
                try:
                    result = hook_fn(*args, **kwargs)
                    results.append(result)
                except Exception:
                    import logging

                    logging.getLogger("ark.plugin").exception("Hook execution failed")
        return results

    def list_plugins(self) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "enabled": p.enabled,
            }
            for p in self._plugins.values()
        ]
