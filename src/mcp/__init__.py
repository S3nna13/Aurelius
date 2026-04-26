"""Aurelius MCP (Model Context Protocol) surface.

Exposes the server registry, client registry, and tool schema registry for
the Aurelius MCP integration layer.  All submodules are lazily imported to
avoid circular dependencies and keep startup cost near zero.

Inspired by cline/cline (MCP integration), continuedev/continue (context providers),
Apache-2.0, clean-room reimplementation.
"""

from __future__ import annotations

from importlib import import_module

# ---------------------------------------------------------------------------
# Lazy submodule loader
# ---------------------------------------------------------------------------

_MCP_SUBMODULES = (
    "mcp_server",
    "mcp_client",
    "tool_schema_registry",
    "tool_schema_validator",
    "sse_mcp_server",
    "plugin_host",
    "skill_catalog",
    "extension_manifest",
    "tool_result_formatter",
    "session_manager",
)

__all__ = [
    "MCP_SERVER_REGISTRY",
    "MCP_CLIENT_REGISTRY",
    "MCP_TOOL_SCHEMA_REGISTRY",
    "MCP_REGISTRY",
    # tool_result_formatter
    "ToolResultFormatter",
    "ResultFormat",
    # session_manager
    "MCPSessionManager",
    "MCPSession",
    "SessionState",
    "SSEMCPServer",
    "SSEMCPServerConfig",
    "SSE_SERVER_REGISTRY",
    "PluginHost",
    "PluginManifest",
    "PLUGIN_HOST_REGISTRY",
    "DEFAULT_PLUGIN_HOST",
    "SkillCatalog",
    "SkillMetadata",
    "DEFAULT_SKILL_CATALOG",
    "SKILL_CATALOG_REGISTRY",
    "ExtensionManifestValidator",
    "ExtensionManifest",
    "ManifestValidationResult",
    "ToolSchemaValidator",
    "DEFAULT_TOOL_SCHEMA_VALIDATOR",
    "TOOL_SCHEMA_VALIDATOR_REGISTRY",
    *_MCP_SUBMODULES,
]


def __getattr__(name: str):
    # Expose top-level registry aliases.
    if name == "MCP_SERVER_REGISTRY":
        from .mcp_server import MCP_SERVER_REGISTRY  # noqa: PLC0415
        globals()["MCP_SERVER_REGISTRY"] = MCP_SERVER_REGISTRY
        return MCP_SERVER_REGISTRY
    if name == "MCP_CLIENT_REGISTRY":
        from .mcp_client import MCP_CLIENT_REGISTRY  # noqa: PLC0415
        globals()["MCP_CLIENT_REGISTRY"] = MCP_CLIENT_REGISTRY
        return MCP_CLIENT_REGISTRY
    if name == "MCP_TOOL_SCHEMA_REGISTRY":
        from .tool_schema_registry import MCP_TOOL_SCHEMA_REGISTRY  # noqa: PLC0415
        globals()["MCP_TOOL_SCHEMA_REGISTRY"] = MCP_TOOL_SCHEMA_REGISTRY
        return MCP_TOOL_SCHEMA_REGISTRY
    if name == "SSEMCPServer":
        from .sse_mcp_server import SSEMCPServer  # noqa: PLC0415
        globals()["SSEMCPServer"] = SSEMCPServer
        return SSEMCPServer
    if name == "SSEMCPServerConfig":
        from .sse_mcp_server import SSEMCPServerConfig  # noqa: PLC0415
        globals()["SSEMCPServerConfig"] = SSEMCPServerConfig
        return SSEMCPServerConfig
    if name == "SSE_SERVER_REGISTRY":
        from .sse_mcp_server import SSE_SERVER_REGISTRY  # noqa: PLC0415
        globals()["SSE_SERVER_REGISTRY"] = SSE_SERVER_REGISTRY
        return SSE_SERVER_REGISTRY
    if name == "PluginHost":
        from .plugin_host import PluginHost  # noqa: PLC0415
        globals()["PluginHost"] = PluginHost
        return PluginHost
    if name == "PluginManifest":
        from .plugin_host import PluginManifest  # noqa: PLC0415
        globals()["PluginManifest"] = PluginManifest
        return PluginManifest
    if name == "PLUGIN_HOST_REGISTRY":
        from .plugin_host import PLUGIN_HOST_REGISTRY  # noqa: PLC0415
        globals()["PLUGIN_HOST_REGISTRY"] = PLUGIN_HOST_REGISTRY
        return PLUGIN_HOST_REGISTRY
    if name == "DEFAULT_PLUGIN_HOST":
        from .plugin_host import DEFAULT_PLUGIN_HOST  # noqa: PLC0415
        globals()["DEFAULT_PLUGIN_HOST"] = DEFAULT_PLUGIN_HOST
        return DEFAULT_PLUGIN_HOST
    if name == "SkillCatalog":
        from .skill_catalog import SkillCatalog  # noqa: PLC0415
        globals()["SkillCatalog"] = SkillCatalog
        return SkillCatalog
    if name == "SkillMetadata":
        from .skill_catalog import SkillMetadata  # noqa: PLC0415
        globals()["SkillMetadata"] = SkillMetadata
        return SkillMetadata
    if name == "DEFAULT_SKILL_CATALOG":
        from .skill_catalog import DEFAULT_SKILL_CATALOG  # noqa: PLC0415
        globals()["DEFAULT_SKILL_CATALOG"] = DEFAULT_SKILL_CATALOG
        return DEFAULT_SKILL_CATALOG
    if name == "SKILL_CATALOG_REGISTRY":
        from .skill_catalog import SKILL_CATALOG_REGISTRY  # noqa: PLC0415
        globals()["SKILL_CATALOG_REGISTRY"] = SKILL_CATALOG_REGISTRY
        return SKILL_CATALOG_REGISTRY
    if name == "ExtensionManifestValidator":
        from .extension_manifest import ExtensionManifestValidator  # noqa: PLC0415
        globals()["ExtensionManifestValidator"] = ExtensionManifestValidator
        return ExtensionManifestValidator
    if name == "ExtensionManifest":
        from .extension_manifest import ExtensionManifest  # noqa: PLC0415
        globals()["ExtensionManifest"] = ExtensionManifest
        return ExtensionManifest
    if name == "ManifestValidationResult":
        from .extension_manifest import ManifestValidationResult  # noqa: PLC0415
        globals()["ManifestValidationResult"] = ManifestValidationResult
        return ManifestValidationResult
    if name == "MCP_REGISTRY":
        from .tool_result_formatter import MCP_REGISTRY  # noqa: PLC0415
        globals()["MCP_REGISTRY"] = MCP_REGISTRY
        return MCP_REGISTRY
    if name == "ToolResultFormatter":
        from .tool_result_formatter import ToolResultFormatter  # noqa: PLC0415
        globals()["ToolResultFormatter"] = ToolResultFormatter
        return ToolResultFormatter
    if name == "ResultFormat":
        from .tool_result_formatter import ResultFormat  # noqa: PLC0415
        globals()["ResultFormat"] = ResultFormat
        return ResultFormat
    if name == "MCPSessionManager":
        from .session_manager import MCPSessionManager  # noqa: PLC0415
        globals()["MCPSessionManager"] = MCPSessionManager
        return MCPSessionManager
    if name == "MCPSession":
        from .session_manager import MCPSession  # noqa: PLC0415
        globals()["MCPSession"] = MCPSession
        return MCPSession
    if name == "SessionState":
        from .session_manager import SessionState  # noqa: PLC0415
        globals()["SessionState"] = SessionState
        return SessionState
    if name == "ToolSchemaValidator":
        from .tool_schema_validator import ToolSchemaValidator  # noqa: PLC0415
        globals()["ToolSchemaValidator"] = ToolSchemaValidator
        return ToolSchemaValidator
    if name == "DEFAULT_TOOL_SCHEMA_VALIDATOR":
        from .tool_schema_validator import DEFAULT_TOOL_SCHEMA_VALIDATOR  # noqa: PLC0415
        globals()["DEFAULT_TOOL_SCHEMA_VALIDATOR"] = DEFAULT_TOOL_SCHEMA_VALIDATOR
        return DEFAULT_TOOL_SCHEMA_VALIDATOR
    if name == "TOOL_SCHEMA_VALIDATOR_REGISTRY":
        from .tool_schema_validator import TOOL_SCHEMA_VALIDATOR_REGISTRY  # noqa: PLC0415
        globals()["TOOL_SCHEMA_VALIDATOR_REGISTRY"] = TOOL_SCHEMA_VALIDATOR_REGISTRY
        return TOOL_SCHEMA_VALIDATOR_REGISTRY
    if name in _MCP_SUBMODULES:
        module = import_module(f"src.mcp.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'src.mcp' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
