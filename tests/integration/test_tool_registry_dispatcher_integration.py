"""Integration tests for the safe tool dispatcher surface."""

from __future__ import annotations

import src.agent as agent_surface
from src.agent import (
    AGENT_LOOP_REGISTRY,
    SessionBudget,
    ToolInvocationResult,
    ToolRegistryDispatcher,
    ToolSpec,
)


def test_symbols_exposed_on_surface() -> None:
    for name in (
        "ToolRegistryDispatcher",
        "ToolSpec",
        "ToolInvocationResult",
        "SessionBudget",
    ):
        assert hasattr(agent_surface, name), f"missing export: {name}"


def test_safe_dispatch_registered_in_registry() -> None:
    assert "safe_dispatch" in AGENT_LOOP_REGISTRY
    assert AGENT_LOOP_REGISTRY["safe_dispatch"] is ToolRegistryDispatcher


def test_react_still_constructible_regression() -> None:
    # Regression: adding safe_dispatch must not break the existing ReAct entry.
    assert "react" in AGENT_LOOP_REGISTRY
    react_cls = AGENT_LOOP_REGISTRY["react"]
    # It must still be a class/callable (we don't instantiate because ReAct
    # needs a model + tools; constructibility means "is-a-callable-class").
    assert callable(react_cls)


def test_end_to_end_echo_and_add() -> None:
    d = ToolRegistryDispatcher(budget=SessionBudget(total_calls=8, total_wall_seconds=10.0))

    def echo(message: str) -> str:
        return message

    def add(a: int, b: int) -> int:
        return a + b

    d.register(
        ToolSpec(
            name="echo",
            fn=echo,
            description="echo back",
            schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "additionalProperties": False,
            },
        )
    )
    d.register(
        ToolSpec(
            name="add",
            fn=add,
            description="integer addition",
            schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        )
    )

    r1 = d.dispatch("echo", {"message": "hello"})
    r2 = d.dispatch("add", {"a": 2, "b": 3})

    for r in (r1, r2):
        assert isinstance(r, ToolInvocationResult)
        assert r.ok is True
        assert r.error is None
        assert r.duration_ms >= 0.0
        assert isinstance(r.truncated, bool)

    assert r1.value == "hello"
    assert r2.value == 5

    tools = d.list_tools()
    assert [t["name"] for t in tools] == ["echo", "add"]

    log = d.audit_log()
    assert len(log) == 2
    assert all(entry["ok"] for entry in log)
