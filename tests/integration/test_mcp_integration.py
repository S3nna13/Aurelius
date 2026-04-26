"""End-to-end integration tests for the Aurelius MCP surface (src.mcp).

Covers:
- Register a ToolSchema, create a LocalMCPClient backed by StdioMCPServer,
  call list_tools() and validate the schema is present in the registry.
"""

from __future__ import annotations

import pytest

from src.mcp.mcp_client import LocalMCPClient, MCPClientConfig
from src.mcp.mcp_server import MCPServerConfig, StdioMCPServer
from src.mcp.tool_schema_registry import (
    MCP_TOOL_SCHEMA_REGISTRY,
    ToolSchema,
    get_tool_schema,
    register_tool_schema,
    validate_tool_call,
)

# ---------------------------------------------------------------------------
# Fixture: isolated registry
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_registry():
    """Snapshot and restore MCP_TOOL_SCHEMA_REGISTRY around each test."""
    snapshot = dict(MCP_TOOL_SCHEMA_REGISTRY)
    yield
    MCP_TOOL_SCHEMA_REGISTRY.clear()
    MCP_TOOL_SCHEMA_REGISTRY.update(snapshot)


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------


def test_end_to_end_schema_register_client_list_tools(clean_registry):
    """Register a ToolSchema, wire up LocalMCPClient to StdioMCPServer,
    call list_tools(), and confirm the schema is present in the registry."""

    # 1. Register a ToolSchema.
    schema = ToolSchema(
        name="greet",
        description="Greet a user by name.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        output_schema={"type": "object"},
        version="1.0",
        tags=["integration", "greeting"],
    )
    register_tool_schema(schema)

    # 2. Confirm schema is in the registry.
    retrieved = get_tool_schema("greet")
    assert retrieved.name == "greet"
    assert retrieved.version == "1.0"

    # 3. Create a StdioMCPServer and start it.
    server = StdioMCPServer(MCPServerConfig(transport="stdio"))
    server.start()

    # 4. Wrap in a LocalMCPClient.
    client = LocalMCPClient(server, config=MCPClientConfig(server_name="test", transport="local"))

    # 5. list_tools() returns a list (server built-in returns empty list,
    #    but the call must succeed and be a list type).
    tools = client.list_tools()
    assert isinstance(tools, list)

    # 6. Validate the schema against correct args.
    assert validate_tool_call("greet", {"name": "Aurelius"}) is True

    # 7. Clean up server.
    server.stop()
    assert not server._running


def test_end_to_end_schema_version_mismatch_detected(clean_registry):
    """Register v1 schema then attempt to register an incompatible v2; confirm error."""
    from src.mcp.tool_schema_registry import ToolSchemaError

    schema_v1 = ToolSchema(
        name="search",
        description="Search tool v1.",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        output_schema={},
        version="1.0",
    )
    register_tool_schema(schema_v1)

    schema_v2 = ToolSchema(
        name="search",
        description="Search tool v2 — breaking change.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        output_schema={},
        version="2.0",
    )
    with pytest.raises(ToolSchemaError, match="Version mismatch"):
        register_tool_schema(schema_v2)


def test_end_to_end_mcp_surface_lazy_import():
    """src.mcp exposes MCP_SERVER_REGISTRY and MCP_TOOL_SCHEMA_REGISTRY lazily."""
    import src.mcp as mcp_surface

    assert hasattr(mcp_surface, "MCP_SERVER_REGISTRY") or "MCP_SERVER_REGISTRY" in dir(mcp_surface)
    # Force lazy load
    reg = mcp_surface.MCP_SERVER_REGISTRY
    assert "stdio" in reg

    tool_reg = mcp_surface.MCP_TOOL_SCHEMA_REGISTRY
    assert isinstance(tool_reg, dict)


# ---------------------------------------------------------------------------
# SSEMCPServer integration
# ---------------------------------------------------------------------------


def test_sse_server_handle():
    """Instantiate SSEMCPServer, register a handler, call handle_request, verify."""
    from src.mcp.sse_mcp_server import SSEMCPServer

    server = SSEMCPServer()

    def greet(payload: dict) -> dict:
        return {"hello": payload.get("name", "world")}

    server.register_tool_handler("greet", greet)

    response = server.handle_request({"tool": "greet", "name": "Aurelius"})
    assert response == {"hello": "Aurelius"}

    # Unknown tool returns error, not a crash
    error_resp = server.handle_request({"tool": "unknown_xyz"})
    assert "error" in error_resp


# ---------------------------------------------------------------------------
# PluginHost lifecycle integration
# ---------------------------------------------------------------------------


def test_skill_catalog_find_and_invoke():
    """Register a skill, find by trigger, record an invocation, verify history."""
    import time

    from src.mcp.skill_catalog import (
        SkillCatalog,
        SkillInvocationRecord,
        SkillMetadata,
        SkillTrigger,
    )

    catalog = SkillCatalog()
    skill = SkillMetadata(
        skill_id="integration-skill",
        name="Integration Test Skill",
        version="2.0.0",
        description="Used in integration test.",
        triggers=[
            SkillTrigger(
                pattern="integration",
                description="Fires on integration keyword.",
                examples=["run integration"],
            )
        ],
        tags=["integration"],
    )
    catalog.register(skill)

    # find by trigger
    found = catalog.find_by_trigger("run integration test now")
    assert any(s.skill_id == "integration-skill" for s in found)

    # record invocation
    record = SkillInvocationRecord(
        skill_id="integration-skill",
        invoked_at=time.time(),
        args={"mode": "test"},
        result={"status": "ok"},
        duration_ms=42.0,
    )
    catalog.record_invocation(record)

    history = catalog.invocation_history(skill_id="integration-skill")
    assert len(history) == 1
    assert history[0].result == {"status": "ok"}


def test_extension_manifest_validate():
    """Validate a valid and an invalid extension manifest via the validator."""
    from src.mcp.extension_manifest import ExtensionManifest, ExtensionManifestValidator

    validator = ExtensionManifestValidator()

    # Valid manifest
    valid_m = ExtensionManifest(
        extension_id="my-ext",
        display_name="My Extension",
        version="1.0.0",
        description="A sufficiently long description for this extension.",
        mcp_version="1.0",
        permissions=["file_read"],
    )
    result = validator.validate(valid_m)
    assert result.valid is True
    assert result.errors == []

    # Invalid manifest: empty id, bad version, short description
    invalid_m = ExtensionManifest(
        extension_id="",
        display_name="X",
        version="bad",
        description="short",
    )
    bad_result = validator.validate(invalid_m)
    assert bad_result.valid is False
    assert len(bad_result.errors) >= 2


def test_plugin_host_lifecycle():
    """Register plugin, enable, list, disable, unregister — full lifecycle."""
    from src.mcp.plugin_host import PluginHost, PluginHostError, PluginManifest

    host = PluginHost()

    manifest = PluginManifest(
        plugin_id="integration.test",
        name="Integration Test Plugin",
        version="2.3.4",
        description="Used in integration lifecycle test.",
        tool_schemas=["greet", "ping"],
        entry_point="integration_test:main",
    )

    # Register
    host.register(manifest)
    retrieved = host.get("integration.test")
    assert retrieved.version == "2.3.4"

    # Enable (already enabled by default — re-enable is idempotent)
    host.enable("integration.test")
    assert host.get("integration.test").enabled is True

    # list_plugins returns it
    enabled = host.list_plugins(enabled_only=True)
    assert any(p.plugin_id == "integration.test" for p in enabled)

    # Disable
    host.disable("integration.test")
    assert host.get("integration.test").enabled is False

    # Not in enabled-only list
    enabled_after = host.list_plugins(enabled_only=True)
    assert not any(p.plugin_id == "integration.test" for p in enabled_after)

    # Unregister
    host.unregister("integration.test")
    with pytest.raises(PluginHostError):
        host.get("integration.test")
