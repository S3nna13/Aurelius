"""Integration tests: :class:`RecoveringDispatcher` over a real
:class:`ToolRegistryDispatcher`.
"""

from __future__ import annotations

import pytest

from src.agent import (
    RecoveringDispatcher,
    RecoveryPolicy,
    RecoveryStrategy,
    ToolRegistryDispatcher,
    ToolSpec,
)


def test_public_exports_present():
    # Symbols flow through the agent surface's __init__.
    import src.agent as agent

    for name in (
        "RecoveringDispatcher",
        "RecoveryPolicy",
        "RecoveryStrategy",
    ):
        assert hasattr(agent, name), name


def test_flaky_tool_recovers_after_two_failures():
    """Tool fails twice, succeeds on third attempt."""

    state = {"count": 0}

    def flaky(x: int) -> int:
        state["count"] += 1
        if state["count"] < 3:
            raise TimeoutError("simulated timeout")
        return x * 2

    inner = ToolRegistryDispatcher()
    inner.register(
        ToolSpec(
            name="flaky",
            fn=flaky,
            schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            },
            per_call_timeout=2.0,
        )
    )

    sleeps: list[float] = []
    disp = RecoveringDispatcher(
        inner,
        RecoveryPolicy(max_retries=5, backoff_base=0.01, backoff_factor=2.0),
        sleep=sleeps.append,
    )

    result = disp.dispatch("flaky", {"x": 21})
    assert result.ok is True
    assert result.value == 42
    assert state["count"] == 3  # attempt_count == 3
    # Two retry-with-backoff decisions were taken.
    history = disp.recovery_history()
    assert len(history) == 2
    assert all(d.strategy == RecoveryStrategy.RETRY_BACKOFF for _, d in history)
    assert len(sleeps) == 2


def test_validation_error_routes_to_modified_args():
    """Schema violation should invoke the args_mutator and recover."""

    def echo(msg: str) -> str:
        return f"echo:{msg}"

    inner = ToolRegistryDispatcher()
    inner.register(
        ToolSpec(
            name="echo",
            fn=echo,
            schema={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
                "additionalProperties": False,
            },
        )
    )

    def mutate(name, args, result):
        return {"msg": "fallback"}

    disp = RecoveringDispatcher(
        inner,
        RecoveryPolicy(max_retries=3),
        args_mutator=mutate,
        sleep=lambda s: None,
    )
    # Missing required 'msg' -> validation error -> mutator supplies it.
    result = disp.dispatch("echo", {})
    assert result.ok is True
    assert result.value == "echo:fallback"


def test_unknown_tool_falls_back_to_registered_sibling():
    def newsearch(q: str) -> str:
        return f"hits:{q}"

    inner = ToolRegistryDispatcher()
    inner.register(
        ToolSpec(
            name="search_v2",
            fn=newsearch,
            schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
                "additionalProperties": False,
            },
        )
    )

    disp = RecoveringDispatcher(
        inner,
        RecoveryPolicy(fallback_map={"search_v1": "search_v2"}),
        sleep=lambda s: None,
    )
    result = disp.dispatch("search_v1", {"q": "aurelius"})
    assert result.ok is True
    assert result.value == "hits:aurelius"
