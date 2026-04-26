"""Tests for src/cli/plugin_commands.py — ~50 tests."""

import pytest

from src.cli.plugin_commands import (
    PLUGIN_COMMANDS,
    PluginCommandResult,
    PluginCommands,
)

# ---------------------------------------------------------------------------
# PluginCommandResult
# ---------------------------------------------------------------------------


class TestPluginCommandResult:
    def test_success_field_true(self):
        r = PluginCommandResult(success=True, message="ok")
        assert r.success is True

    def test_success_field_false(self):
        r = PluginCommandResult(success=False, message="err")
        assert r.success is False

    def test_message_field(self):
        r = PluginCommandResult(success=True, message="hello")
        assert r.message == "hello"

    def test_data_default_empty_dict(self):
        r = PluginCommandResult(success=True, message="ok")
        assert r.data == {}

    def test_data_field_custom(self):
        r = PluginCommandResult(success=True, message="ok", data={"k": "v"})
        assert r.data == {"k": "v"}

    def test_data_default_is_independent(self):
        r1 = PluginCommandResult(success=True, message="a")
        r2 = PluginCommandResult(success=True, message="b")
        r1.data["x"] = 1
        assert r2.data == {}


# ---------------------------------------------------------------------------
# PluginCommands — fresh instance per test
# ---------------------------------------------------------------------------


@pytest.fixture
def pc():
    return PluginCommands()


class TestRegisterPlugin:
    def test_returns_success_true(self, pc):
        r = pc.register_plugin("foo", "1.0.0")
        assert r.success is True

    def test_returns_plugin_in_data(self, pc):
        r = pc.register_plugin("foo", "1.0.0", "desc")
        assert r.data["name"] == "foo"

    def test_version_in_data(self, pc):
        r = pc.register_plugin("foo", "2.1.3")
        assert r.data["version"] == "2.1.3"

    def test_description_in_data(self, pc):
        r = pc.register_plugin("foo", "1.0", "my desc")
        assert r.data["description"] == "my desc"

    def test_description_defaults_empty(self, pc):
        r = pc.register_plugin("bar", "0.1")
        assert r.data["description"] == ""

    def test_plugin_appears_in_list(self, pc):
        pc.register_plugin("baz", "1.0")
        names = [p["name"] for p in pc.list_plugins()]
        assert "baz" in names

    def test_initial_enabled_false(self, pc):
        pc.register_plugin("foo", "1.0")
        plugins = pc.list_plugins()
        foo = next(p for p in plugins if p["name"] == "foo")
        assert foo["enabled"] is False


class TestEnable:
    def test_enable_registered_success(self, pc):
        pc.register_plugin("foo", "1.0")
        r = pc.enable("foo")
        assert r.success is True

    def test_enable_sets_enabled_true(self, pc):
        pc.register_plugin("foo", "1.0")
        pc.enable("foo")
        plugins = pc.list_plugins()
        foo = next(p for p in plugins if p["name"] == "foo")
        assert foo["enabled"] is True

    def test_enable_unknown_returns_false(self, pc):
        r = pc.enable("nonexistent")
        assert r.success is False

    def test_enable_unknown_message_not_empty(self, pc):
        r = pc.enable("ghost")
        assert r.message != ""

    def test_enable_twice_still_success(self, pc):
        pc.register_plugin("foo", "1.0")
        pc.enable("foo")
        r = pc.enable("foo")
        assert r.success is True


class TestDisable:
    def test_disable_registered_success(self, pc):
        pc.register_plugin("foo", "1.0")
        pc.enable("foo")
        r = pc.disable("foo")
        assert r.success is True

    def test_disable_sets_enabled_false(self, pc):
        pc.register_plugin("foo", "1.0")
        pc.enable("foo")
        pc.disable("foo")
        plugins = pc.list_plugins()
        foo = next(p for p in plugins if p["name"] == "foo")
        assert foo["enabled"] is False

    def test_disable_unknown_returns_false(self, pc):
        r = pc.disable("ghost")
        assert r.success is False

    def test_disable_unknown_message_not_empty(self, pc):
        r = pc.disable("ghost")
        assert r.message != ""


class TestListPlugins:
    def test_returns_list(self, pc):
        assert isinstance(pc.list_plugins(), list)

    def test_empty_when_no_plugins(self, pc):
        assert pc.list_plugins() == []

    def test_returns_all_by_default(self, pc):
        pc.register_plugin("a", "1")
        pc.register_plugin("b", "2")
        assert len(pc.list_plugins()) == 2

    def test_enabled_only_filters_disabled(self, pc):
        pc.register_plugin("a", "1")
        pc.register_plugin("b", "2")
        pc.enable("a")
        result = pc.list_plugins(enabled_only=True)
        names = [p["name"] for p in result]
        assert "a" in names
        assert "b" not in names

    def test_enabled_only_false_returns_all(self, pc):
        pc.register_plugin("a", "1")
        pc.register_plugin("b", "2")
        pc.enable("a")
        assert len(pc.list_plugins(enabled_only=False)) == 2

    def test_enabled_only_empty_when_all_disabled(self, pc):
        pc.register_plugin("a", "1")
        assert pc.list_plugins(enabled_only=True) == []

    def test_each_dict_has_name_version_description_enabled(self, pc):
        pc.register_plugin("a", "1.0", "desc")
        p = pc.list_plugins()[0]
        assert set(p.keys()) >= {"name", "version", "description", "enabled"}


class TestInfo:
    def test_info_found_success_true(self, pc):
        pc.register_plugin("foo", "1.0")
        r = pc.info("foo")
        assert r.success is True

    def test_info_data_contains_name(self, pc):
        pc.register_plugin("foo", "1.0")
        r = pc.info("foo")
        assert r.data["name"] == "foo"

    def test_info_not_found_success_false(self, pc):
        r = pc.info("ghost")
        assert r.success is False

    def test_info_not_found_data_empty(self, pc):
        r = pc.info("ghost")
        assert r.data == {}

    def test_info_returns_correct_version(self, pc):
        pc.register_plugin("foo", "3.2.1")
        r = pc.info("foo")
        assert r.data["version"] == "3.2.1"


class TestUnregister:
    def test_unregister_success(self, pc):
        pc.register_plugin("foo", "1.0")
        r = pc.unregister("foo")
        assert r.success is True

    def test_unregister_removes_from_list(self, pc):
        pc.register_plugin("foo", "1.0")
        pc.unregister("foo")
        names = [p["name"] for p in pc.list_plugins()]
        assert "foo" not in names

    def test_unregister_unknown_returns_false(self, pc):
        r = pc.unregister("ghost")
        assert r.success is False

    def test_unregister_unknown_message_not_empty(self, pc):
        r = pc.unregister("ghost")
        assert r.message != ""

    def test_unregister_then_info_returns_false(self, pc):
        pc.register_plugin("foo", "1.0")
        pc.unregister("foo")
        r = pc.info("foo")
        assert r.success is False


# ---------------------------------------------------------------------------
# PLUGIN_COMMANDS singleton — pre-registered defaults
# ---------------------------------------------------------------------------


class TestPluginCommandsSingleton:
    def test_mcp_core_registered(self):
        names = [p["name"] for p in PLUGIN_COMMANDS.list_plugins()]
        assert "mcp-core" in names

    def test_eval_runner_registered(self):
        names = [p["name"] for p in PLUGIN_COMMANDS.list_plugins()]
        assert "eval-runner" in names

    def test_mcp_core_enabled(self):
        plugins = PLUGIN_COMMANDS.list_plugins()
        p = next(p for p in plugins if p["name"] == "mcp-core")
        assert p["enabled"] is True

    def test_eval_runner_enabled(self):
        plugins = PLUGIN_COMMANDS.list_plugins()
        p = next(p for p in plugins if p["name"] == "eval-runner")
        assert p["enabled"] is True

    def test_mcp_core_version(self):
        plugins = PLUGIN_COMMANDS.list_plugins()
        p = next(p for p in plugins if p["name"] == "mcp-core")
        assert p["version"] == "1.0.0"

    def test_eval_runner_version(self):
        plugins = PLUGIN_COMMANDS.list_plugins()
        p = next(p for p in plugins if p["name"] == "eval-runner")
        assert p["version"] == "0.3.0"

    def test_mcp_core_description(self):
        plugins = PLUGIN_COMMANDS.list_plugins()
        p = next(p for p in plugins if p["name"] == "mcp-core")
        assert p["description"] == "Core MCP server"

    def test_eval_runner_description(self):
        plugins = PLUGIN_COMMANDS.list_plugins()
        p = next(p for p in plugins if p["name"] == "eval-runner")
        assert p["description"] == "Eval harness runner"

    def test_is_plugin_commands_instance(self):
        assert isinstance(PLUGIN_COMMANDS, PluginCommands)
