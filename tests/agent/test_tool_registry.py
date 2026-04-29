"""Tests for src.agent.tool_registry."""

from __future__ import annotations

from src.agent.tool_registry import ToolRegistry, ToolSpec


def _echo(message: str) -> dict[str, str]:
    return {"message": message}


def test_execute_requires_authentication() -> None:
    registry = ToolRegistry()
    spec = ToolSpec(
        name="private_echo",
        description="Sensitive echo tool",
        requires_auth=True,
    )
    registry.register(spec, handler=_echo)

    result = registry.execute(spec.id, authenticated=False, message="hello")

    assert result == {"error": f"Tool {spec.id} requires authentication"}


def test_execute_respects_requested_sandbox_level() -> None:
    registry = ToolRegistry()
    spec = ToolSpec(
        name="file_writer",
        description="Writes files",
        sandbox_level="read",
    )
    registry.register(spec, handler=_echo)

    result = registry.execute(
        spec.id,
        authenticated=True,
        requested_level="write",
        message="hello",
    )

    assert result == {"error": f"Tool {spec.id} denied for sandbox level write"}


def test_execute_succeeds_when_controls_allow_it() -> None:
    registry = ToolRegistry()
    spec = ToolSpec(
        name="public_echo",
        description="Safe echo tool",
        requires_auth=True,
        sandbox_level="write",
    )
    registry.register(spec, handler=_echo)

    result = registry.execute(
        spec.id,
        authenticated=True,
        requested_level="read",
        message="hello",
    )

    assert result == {"message": "hello"}
