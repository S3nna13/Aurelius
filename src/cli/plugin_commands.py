"""CLI commands for MCP plugin management: list, enable, disable, info."""

from dataclasses import dataclass, field


@dataclass
class PluginCommandResult:
    success: bool
    message: str
    data: dict = field(default_factory=dict)


class PluginCommands:
    def __init__(self) -> None:
        self._plugins: dict[str, dict] = {}

    def register_plugin(
        self, name: str, version: str, description: str = ""
    ) -> PluginCommandResult:
        self._plugins[name] = {
            "name": name,
            "version": version,
            "description": description,
            "enabled": False,
        }
        return PluginCommandResult(
            success=True,
            message=f"Plugin '{name}' registered.",
            data=dict(self._plugins[name]),
        )

    def enable(self, name: str) -> PluginCommandResult:
        if name not in self._plugins:
            return PluginCommandResult(
                success=False, message=f"Plugin '{name}' is not registered."
            )
        self._plugins[name]["enabled"] = True
        return PluginCommandResult(
            success=True,
            message=f"Plugin '{name}' enabled.",
            data=dict(self._plugins[name]),
        )

    def disable(self, name: str) -> PluginCommandResult:
        if name not in self._plugins:
            return PluginCommandResult(
                success=False, message=f"Plugin '{name}' is not registered."
            )
        self._plugins[name]["enabled"] = False
        return PluginCommandResult(
            success=True,
            message=f"Plugin '{name}' disabled.",
            data=dict(self._plugins[name]),
        )

    def list_plugins(self, enabled_only: bool = False) -> list[dict]:
        plugins = list(self._plugins.values())
        if enabled_only:
            plugins = [p for p in plugins if p["enabled"]]
        return [dict(p) for p in plugins]

    def info(self, name: str) -> PluginCommandResult:
        if name not in self._plugins:
            return PluginCommandResult(
                success=False, message=f"Plugin '{name}' not found."
            )
        return PluginCommandResult(
            success=True,
            message=f"Plugin '{name}' info.",
            data=dict(self._plugins[name]),
        )

    def unregister(self, name: str) -> PluginCommandResult:
        if name not in self._plugins:
            return PluginCommandResult(
                success=False, message=f"Plugin '{name}' not found."
            )
        self._plugins.pop(name)
        return PluginCommandResult(
            success=True, message=f"Plugin '{name}' unregistered."
        )


PLUGIN_COMMANDS = PluginCommands()

# Pre-register default plugins
for _plugin in [
    {"name": "mcp-core", "version": "1.0.0", "description": "Core MCP server"},
    {"name": "eval-runner", "version": "0.3.0", "description": "Eval harness runner"},
]:
    PLUGIN_COMMANDS.register_plugin(
        _plugin["name"], _plugin["version"], _plugin["description"]
    )
    PLUGIN_COMMANDS.enable(_plugin["name"])
