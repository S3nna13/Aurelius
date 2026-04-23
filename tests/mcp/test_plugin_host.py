"""Tests for src.mcp.plugin_host — plugin/extension hot-reload registry.

Covers PluginManifest, PluginHost CRUD, lifecycle operations, serialization,
and the module-level singletons.
"""

from __future__ import annotations

import pytest

from src.mcp.plugin_host import (
    DEFAULT_PLUGIN_HOST,
    PLUGIN_HOST_REGISTRY,
    PluginHost,
    PluginHostError,
    PluginManifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manifest(
    plugin_id: str = "test.plugin",
    name: str = "Test Plugin",
    version: str = "1.0.0",
    description: str = "A test plugin.",
    tool_schemas: list[str] | None = None,
    entry_point: str = "test_plugin:main",
    enabled: bool = True,
) -> PluginManifest:
    return PluginManifest(
        plugin_id=plugin_id,
        name=name,
        version=version,
        description=description,
        tool_schemas=tool_schemas or [],
        entry_point=entry_point,
        enabled=enabled,
    )


@pytest.fixture()
def host() -> PluginHost:
    """Fresh PluginHost for each test."""
    return PluginHost()


# ---------------------------------------------------------------------------
# PluginManifest
# ---------------------------------------------------------------------------


class TestPluginManifest:
    def test_default_enabled(self):
        m = PluginManifest(
            plugin_id="x", name="X", version="0.1.0", description="desc"
        )
        assert m.enabled is True

    def test_default_tool_schemas_empty(self):
        m = PluginManifest(
            plugin_id="x", name="X", version="0.1.0", description="desc"
        )
        assert m.tool_schemas == []

    def test_fields_stored_correctly(self):
        m = _manifest()
        assert m.plugin_id == "test.plugin"
        assert m.version == "1.0.0"


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------


class TestRegisterGet:
    def test_register_valid_manifest(self, host: PluginHost):
        m = _manifest()
        host.register(m)
        assert host.get("test.plugin") is m

    def test_register_empty_plugin_id_raises(self, host: PluginHost):
        with pytest.raises(PluginHostError, match="plugin_id"):
            host.register(_manifest(plugin_id=""))

    def test_register_whitespace_plugin_id_raises(self, host: PluginHost):
        with pytest.raises(PluginHostError, match="plugin_id"):
            host.register(_manifest(plugin_id="   "))

    def test_register_invalid_version_raises(self, host: PluginHost):
        with pytest.raises(PluginHostError, match="version"):
            host.register(_manifest(version="bad"))

    def test_register_version_with_prerelease_raises(self, host: PluginHost):
        with pytest.raises(PluginHostError, match="version"):
            host.register(_manifest(version="1.0.0-alpha"))

    def test_register_semver_two_part_raises(self, host: PluginHost):
        with pytest.raises(PluginHostError, match="version"):
            host.register(_manifest(version="1.0"))

    def test_get_unknown_raises(self, host: PluginHost):
        with pytest.raises(PluginHostError, match="not found"):
            host.get("nope")

    def test_register_overwrites_existing(self, host: PluginHost):
        host.register(_manifest(name="v1"))
        host.register(_manifest(name="v2"))
        assert host.get("test.plugin").name == "v2"


# ---------------------------------------------------------------------------
# unregister
# ---------------------------------------------------------------------------


class TestUnregister:
    def test_unregister_removes_plugin(self, host: PluginHost):
        host.register(_manifest())
        host.unregister("test.plugin")
        with pytest.raises(PluginHostError):
            host.get("test.plugin")

    def test_unregister_unknown_raises(self, host: PluginHost):
        with pytest.raises(PluginHostError, match="not found"):
            host.unregister("ghost")


# ---------------------------------------------------------------------------
# list_plugins
# ---------------------------------------------------------------------------


class TestListPlugins:
    def test_list_enabled_only(self, host: PluginHost):
        host.register(_manifest("a", enabled=True))
        host.register(_manifest("b", enabled=False))
        result = host.list_plugins(enabled_only=True)
        ids = {m.plugin_id for m in result}
        assert "a" in ids
        assert "b" not in ids

    def test_list_all(self, host: PluginHost):
        host.register(_manifest("a", enabled=True))
        host.register(_manifest("b", enabled=False))
        result = host.list_plugins(enabled_only=False)
        assert len(result) == 2

    def test_list_empty_host(self, host: PluginHost):
        assert host.list_plugins() == []


# ---------------------------------------------------------------------------
# enable / disable
# ---------------------------------------------------------------------------


class TestEnableDisable:
    def test_disable_sets_enabled_false(self, host: PluginHost):
        host.register(_manifest())
        host.disable("test.plugin")
        assert host.get("test.plugin").enabled is False

    def test_enable_sets_enabled_true(self, host: PluginHost):
        host.register(_manifest(enabled=False))
        host.enable("test.plugin")
        assert host.get("test.plugin").enabled is True

    def test_enable_unknown_raises(self, host: PluginHost):
        with pytest.raises(PluginHostError):
            host.enable("ghost")

    def test_disable_unknown_raises(self, host: PluginHost):
        with pytest.raises(PluginHostError):
            host.disable("ghost")

    def test_toggle_cycle(self, host: PluginHost):
        host.register(_manifest())
        host.disable("test.plugin")
        host.enable("test.plugin")
        assert host.get("test.plugin").enabled is True


# ---------------------------------------------------------------------------
# reload
# ---------------------------------------------------------------------------


class TestReload:
    def test_reload_replaces_manifest(self, host: PluginHost):
        host.register(_manifest(name="old name"))
        new = _manifest(name="new name")
        host.reload("test.plugin", new)
        assert host.get("test.plugin").name == "new name"

    def test_reload_unknown_raises(self, host: PluginHost):
        with pytest.raises(PluginHostError, match="not found"):
            host.reload("ghost", _manifest())

    def test_reload_invalid_manifest_raises(self, host: PluginHost):
        host.register(_manifest())
        with pytest.raises(PluginHostError, match="version"):
            host.reload("test.plugin", _manifest(version="bad-ver"))


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    def test_to_dict_returns_dict(self, host: PluginHost):
        host.register(_manifest())
        result = host.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_contains_plugin_id_key(self, host: PluginHost):
        host.register(_manifest())
        result = host.to_dict()
        assert "test.plugin" in result

    def test_to_dict_has_version_field(self, host: PluginHost):
        host.register(_manifest())
        entry = host.to_dict()["test.plugin"]
        assert entry["version"] == "1.0.0"

    def test_to_dict_empty_host(self, host: PluginHost):
        assert host.to_dict() == {}


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------


class TestSingletons:
    def test_default_plugin_host_is_plugin_host(self):
        assert isinstance(DEFAULT_PLUGIN_HOST, PluginHost)

    def test_plugin_host_registry_is_dict(self):
        assert isinstance(PLUGIN_HOST_REGISTRY, dict)

    def test_plugin_host_registry_has_default(self):
        assert "default" in PLUGIN_HOST_REGISTRY

    def test_plugin_host_registry_default_is_default_host(self):
        assert PLUGIN_HOST_REGISTRY["default"] is DEFAULT_PLUGIN_HOST
