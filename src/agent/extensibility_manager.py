"""Extensibility subsystem — MCP, plugins, skills, and hooks.

Four extension mechanisms from Claude Code:
1. MCP servers — Model Context Protocol for external tools/data
2. Plugins — self-contained extension packages
3. Skills — reusable instruction templates
4. Hooks — lifecycle event interception (27 event types, 5 safety-related)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HookEvent(Enum):
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    PRE_MODEL_CALL = "pre_model_call"
    POST_MODEL_CALL = "post_model_call"
    PERMISSION_REQUEST = "permission_request"
    TOOL_ERROR = "tool_error"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    COMPACTION = "compaction"
    SUBAGENT_SPAWN = "subagent_spawn"
    SUBAGENT_RETURN = "subagent_return"
    CONTEXT_ASSEMBLY = "context_assembly"


@dataclass
class Hook:
    event: HookEvent
    handler: Callable[[dict[str, Any]], dict[str, Any] | None]
    name: str = ""
    priority: int = 0
    blocking: bool = False

    def __call__(self, context: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return self.handler(context)
        except Exception:
            return None


class HookRegistry:
    """Registry for lifecycle hooks with 27 event types."""

    def __init__(self):
        self._hooks: dict[HookEvent, list[Hook]] = {event: [] for event in HookEvent}

    def register(self, hook: Hook) -> None:
        self._hooks[hook.event].append(hook)
        self._hooks[hook.event].sort(key=lambda h: h.priority, reverse=True)

    def unregister(self, name: str) -> None:
        for event in self._hooks:
            self._hooks[event] = [h for h in self._hooks[event] if h.name != name]

    def dispatch(self, event: HookEvent, context: dict[str, Any]) -> list[dict[str, Any] | None]:
        results: list[dict[str, Any] | None] = []
        for hook in self._hooks[event]:
            result = hook(context)
            results.append(result)
            if hook.blocking and result is not None and result.get("block", False):
                break
        return results

    @property
    def total_hooks(self) -> int:
        return sum(len(h) for h in self._hooks.values())


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    server: str = ""


@dataclass
class MCPServerConfig:
    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""


class MCPServer:
    """Model Context Protocol server connection."""

    def __init__(self, config: MCPServerConfig):
        self.cfg = config
        self.tools: list[MCPTool] = []
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def list_tools(self) -> list[MCPTool]:
        return self.tools

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"result": f"executed {name}", "status": "ok"}

    @property
    def connected(self) -> bool:
        return self._connected


class Plugin:
    """Self-contained extension package."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self._hooks: list[Hook] = []
        self._tools: list[MCPTool] = []

    def activate(self, registry: HookRegistry) -> None:
        for hook in self._hooks:
            registry.register(hook)

    def deactivate(self, registry: HookRegistry) -> None:
        for hook in self._hooks:
            registry.unregister(hook.name)

    def add_hook(self, hook: Hook) -> None:
        self._hooks.append(hook)

    def add_tool(self, tool: MCPTool) -> None:
        self._tools.append(tool)


class Skill:
    """Reusable instruction template with triggers."""

    def __init__(self, name: str, trigger: str, instructions: str):
        self.name = name
        self.trigger = trigger
        self.instructions = instructions
        self._enabled: bool = True

    def matches(self, query: str) -> bool:
        return self._enabled and self.trigger.lower() in query.lower()

    def apply(self, prompt: str) -> str:
        return f"{self.instructions}\n\n{prompt}" if self.matches(prompt) else prompt

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False


class ExtensibilityManager:
    """Central manager for MCP, plugins, skills, and hooks."""

    def __init__(self):
        self.hooks = HookRegistry()
        self.mcp_servers: dict[str, MCPServer] = {}
        self.plugins: dict[str, Plugin] = {}
        self.skills: dict[str, Skill] = {}
        self._tool_pool: list[MCPTool] = []

    def register_mcp(self, config: MCPServerConfig) -> MCPServer:
        server = MCPServer(config)
        self.mcp_servers[config.name] = server
        return server

    def register_plugin(self, plugin: Plugin) -> None:
        self.plugins[plugin.name] = plugin
        plugin.activate(self.hooks)

    def register_skill(self, skill: Skill) -> None:
        self.skills[skill.name] = skill

    def register_hook(self, hook: Hook) -> None:
        self.hooks.register(hook)

    def assemble_tool_pool(self) -> list[MCPTool]:
        self._tool_pool.clear()
        for server in self.mcp_servers.values():
            if server.connected:
                self._tool_pool.extend(server.list_tools())
        for plugin in self.plugins.values():
            self._tool_pool.extend(plugin._tools)
        return self._tool_pool

    def apply_skills(self, prompt: str) -> str:
        result = prompt
        for skill in self.skills.values():
            result = skill.apply(result)
        return result

    def dispatch_hook(
        self,
        event: HookEvent,
        context: dict[str, Any],
    ) -> list[dict[str, Any] | None]:
        return self.hooks.dispatch(event, context)

    @property
    def n_tools_available(self) -> int:
        total = sum(len(server.list_tools()) for server in self.mcp_servers.values())
        total += sum(len(plugin._tools) for plugin in self.plugins.values())
        return total
