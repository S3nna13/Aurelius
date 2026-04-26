"""Tests for src.agent.plugin_sandbox."""

from __future__ import annotations

import types

import pytest

from src.agent.plugin_hook import PluginHookRegistry
from src.agent.plugin_sandbox import (
    PLUGIN_SANDBOX_REGISTRY,
    PluginSandbox,
    SandboxConfig,
    SandboxViolationError,
)


def _safe_add(a: int, b: int) -> int:
    return a + b


def _raises_error() -> None:
    raise ValueError("intentional")


def _make_uses_os():
    """Return a function whose __globals__ contains 'os'."""
    mod = types.ModuleType("tmp_os_mod")
    mod.__dict__["os"] = __import__("os")
    exec(  # noqa: S102
        "def fn():\n    return os.getcwd()\n",
        mod.__dict__,
    )
    return mod.__dict__["fn"]


def _make_uses_json():
    """Return a function whose __globals__ contains 'json'."""
    mod = types.ModuleType("tmp_json_mod")
    mod.__dict__["json"] = __import__("json")
    exec(  # noqa: S102
        "def fn():\n    return json.dumps({})\n",
        mod.__dict__,
    )
    return mod.__dict__["fn"]


class TestSandboxViolationError:
    def test_is_exception(self) -> None:
        with pytest.raises(SandboxViolationError):
            raise SandboxViolationError("test")


class TestSandboxConfig:
    def test_defaults(self) -> None:
        cfg = SandboxConfig()
        assert cfg.timeout_seconds == 5.0
        assert cfg.max_memory_mb is None
        assert "os" in cfg.denied_imports
        assert cfg.allow_network is False

    def test_custom_denied_imports(self) -> None:
        cfg = SandboxConfig(denied_imports=["foo"])
        assert cfg.denied_imports == ["foo"]

    def test_negative_memory_raises(self) -> None:
        with pytest.raises(ValueError, match="max_memory_mb"):
            SandboxConfig(max_memory_mb=-1)

    def test_zero_memory_ok(self) -> None:
        cfg = SandboxConfig(max_memory_mb=0)
        assert cfg.max_memory_mb == 0


class TestPluginSandboxRun:
    def test_safe_callable(self) -> None:
        sandbox = PluginSandbox()
        result = sandbox.run(_safe_add, 2, b=3)
        assert result.success is True
        assert result.output == 5
        assert result.duration_ms >= 0

    def test_callable_with_denied_import_blocked(self) -> None:
        sandbox = PluginSandbox()
        result = sandbox.run(_make_uses_os())
        assert result.success is False
        assert result.violation is not None
        assert "denied import" in result.violation

    def test_exception_caught(self) -> None:
        sandbox = PluginSandbox()
        result = sandbox.run(_raises_error)
        assert result.success is False
        assert "intentional" in result.violation
        assert result.duration_ms >= 0

    def test_non_callable_raises(self) -> None:
        sandbox = PluginSandbox()
        with pytest.raises(TypeError, match="callable"):
            sandbox.run("not callable")  # type: ignore[arg-type]

    def test_empty_denied_list_allows_all(self) -> None:
        sandbox = PluginSandbox(SandboxConfig(denied_imports=[]))
        result = sandbox.run(_make_uses_os())
        assert result.success is True
        assert result.output is not None

    def test_custom_denied_import(self) -> None:
        sandbox = PluginSandbox(SandboxConfig(denied_imports=["json"]))
        result = sandbox.run(_make_uses_json())
        assert result.success is False
        assert "denied import: json" in result.violation


class TestPluginSandboxRunHook:
    def test_safe_hooks(self) -> None:
        registry = PluginHookRegistry()
        registry.register("pre_tool_call", lambda **kwargs: None)
        sandbox = PluginSandbox()
        result = sandbox.run_hook(registry, "pre_tool_call")
        assert result.success is True

    def test_violating_hook_fails(self) -> None:
        registry = PluginHookRegistry()
        registry.register("pre_tool_call", _make_uses_os())
        sandbox = PluginSandbox()
        result = sandbox.run_hook(registry, "pre_tool_call")
        assert result.success is False
        assert "denied import" in result.violation

    def test_unknown_hook_point(self) -> None:
        registry = PluginHookRegistry()
        sandbox = PluginSandbox()
        result = sandbox.run_hook(registry, "nonexistent")
        assert result.success is False
        assert "unknown hook point" in result.violation


class TestPluginSandboxCheckModule:
    def test_finds_denied_imports(self) -> None:
        sandbox = PluginSandbox()
        mod = types.ModuleType("test_mod")
        mod.os = object()
        violations = sandbox.check_module(mod)
        assert "denied import: os" in violations

    def test_returns_empty_for_safe_module(self) -> None:
        sandbox = PluginSandbox()
        mod = types.ModuleType("safe_mod")
        violations = sandbox.check_module(mod)
        assert violations == []

    def test_multiple_violations(self) -> None:
        sandbox = PluginSandbox()
        mod = types.ModuleType("test_mod")
        mod.os = object()
        mod.socket = object()
        violations = sandbox.check_module(mod)
        assert len(violations) == 2


class TestRegistry:
    def test_default_singleton_exists(self) -> None:
        assert "default" in PLUGIN_SANDBOX_REGISTRY

    def test_custom_sandbox_in_registry(self) -> None:
        custom = PluginSandbox(SandboxConfig(timeout_seconds=10.0))
        PLUGIN_SANDBOX_REGISTRY["custom"] = custom
        assert PLUGIN_SANDBOX_REGISTRY["custom"] is custom
        del PLUGIN_SANDBOX_REGISTRY["custom"]
