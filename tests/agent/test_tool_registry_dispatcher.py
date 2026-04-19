"""Unit tests for :mod:`src.agent.tool_registry_dispatcher`."""

from __future__ import annotations

import time

import pytest

from src.agent.tool_registry_dispatcher import (
    SessionBudget,
    ToolInvocationResult,
    ToolRegistryDispatcher,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _echo(message: str) -> str:
    return message


def _add(a: int, b: int) -> int:
    return a + b


_ECHO_SCHEMA = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
    "additionalProperties": False,
}

_ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
    "additionalProperties": False,
}


def _make_echo_spec(**overrides) -> ToolSpec:
    defaults = dict(
        name="echo",
        fn=_echo,
        schema=_ECHO_SCHEMA,
        description="Echo the input",
        per_call_timeout=2.0,
    )
    defaults.update(overrides)
    return ToolSpec(**defaults)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_register_and_dispatch_happy_path() -> None:
    d = ToolRegistryDispatcher()
    d.register(_make_echo_spec())
    result = d.dispatch("echo", {"message": "hello"})
    assert isinstance(result, ToolInvocationResult)
    assert result.ok is True
    assert result.value == "hello"
    assert result.error is None
    assert result.duration_ms >= 0.0
    assert result.truncated is False


def test_schema_validation_missing_required() -> None:
    d = ToolRegistryDispatcher()
    d.register(_make_echo_spec())
    result = d.dispatch("echo", {})
    assert result.ok is False
    assert result.error is not None
    assert "missing_required" in result.error


def test_schema_type_mismatch_rejected() -> None:
    d = ToolRegistryDispatcher()
    d.register(ToolSpec(name="add", fn=_add, schema=_ADD_SCHEMA))
    result = d.dispatch("add", {"a": "not-an-int", "b": 3})
    assert result.ok is False
    assert "type_mismatch" in (result.error or "")


def test_additional_properties_false_rejects_extras() -> None:
    d = ToolRegistryDispatcher()
    d.register(_make_echo_spec())
    result = d.dispatch("echo", {"message": "hi", "extra": 1})
    assert result.ok is False
    assert "unexpected_property" in (result.error or "")


def test_unknown_tool_name() -> None:
    d = ToolRegistryDispatcher()
    result = d.dispatch("does-not-exist", {})
    assert result.ok is False
    assert result.error == "unknown_tool"


def test_tool_raises_captured() -> None:
    def kaboom() -> None:
        raise RuntimeError("boom")

    d = ToolRegistryDispatcher()
    d.register(
        ToolSpec(
            name="kaboom",
            fn=kaboom,
            schema={"type": "object", "properties": {}, "additionalProperties": False},
        )
    )
    result = d.dispatch("kaboom", {})
    assert result.ok is False
    assert "tool_raised" in (result.error or "")
    assert "boom" in (result.error or "")


def test_timeout_returns_timeout_error() -> None:
    def slow() -> str:
        time.sleep(0.4)
        return "done"

    d = ToolRegistryDispatcher()
    d.register(
        ToolSpec(
            name="slow",
            fn=slow,
            schema={"type": "object", "properties": {}, "additionalProperties": False},
            per_call_timeout=0.05,
        )
    )
    result = d.dispatch("slow", {})
    assert result.ok is False
    assert "timeout" in (result.error or "")


def test_session_budget_total_calls_exhausted() -> None:
    d = ToolRegistryDispatcher(budget=SessionBudget(total_calls=2, total_wall_seconds=60.0))
    d.register(_make_echo_spec())
    assert d.dispatch("echo", {"message": "a"}).ok
    assert d.dispatch("echo", {"message": "b"}).ok
    blocked = d.dispatch("echo", {"message": "c"})
    assert blocked.ok is False
    assert "budget" in (blocked.error or "")


def test_session_budget_wall_seconds_exhausted() -> None:
    fake_now = [1000.0]

    def clock() -> float:
        return fake_now[0]

    d = ToolRegistryDispatcher(
        budget=SessionBudget(total_calls=100, total_wall_seconds=0.5),
        clock=clock,
    )
    d.register(_make_echo_spec())
    assert d.dispatch("echo", {"message": "a"}).ok
    fake_now[0] += 5.0  # advance past wall budget
    blocked = d.dispatch("echo", {"message": "b"})
    assert blocked.ok is False
    assert "budget" in (blocked.error or "")
    assert "wall_seconds" in (blocked.error or "")


def test_rate_limit_kicks_in() -> None:
    fake_now = [0.0]

    def clock() -> float:
        return fake_now[0]

    d = ToolRegistryDispatcher(clock=clock)
    d.register(_make_echo_spec(rate_limit_per_minute=3))
    # 3 rapid calls inside same instant: fine. 4th: rate limited.
    for i in range(3):
        assert d.dispatch("echo", {"message": str(i)}).ok
    blocked = d.dispatch("echo", {"message": "x"})
    assert blocked.ok is False
    assert blocked.error == "rate_limited"


def test_audit_log_records_every_attempt() -> None:
    d = ToolRegistryDispatcher()
    d.register(_make_echo_spec())
    d.dispatch("echo", {"message": "ok"})
    d.dispatch("echo", {})  # schema failure
    d.dispatch("missing", {})  # unknown tool
    log = d.audit_log()
    assert len(log) == 3
    names = [entry["name"] for entry in log]
    assert names == ["echo", "echo", "missing"]
    statuses = [entry["ok"] for entry in log]
    assert statuses == [True, False, False]


def test_redactor_applied_to_error_and_value() -> None:
    def redactor(text: str) -> str:
        return text.replace("secret", "[REDACTED]")

    def leak(message: str) -> str:
        return f"value={message}"

    d = ToolRegistryDispatcher(redactor=redactor)
    d.register(
        ToolSpec(
            name="leak",
            fn=leak,
            schema=_ECHO_SCHEMA,
        )
    )
    result = d.dispatch("leak", {"message": "my secret data"})
    assert result.ok is True
    assert "secret" not in result.value
    assert "[REDACTED]" in result.value

    # Now force an error that contains the word 'secret'.
    def raiser(message: str) -> str:
        raise ValueError("oh no, the secret leaked")

    d2 = ToolRegistryDispatcher(redactor=redactor)
    d2.register(ToolSpec(name="raiser", fn=raiser, schema=_ECHO_SCHEMA))
    r2 = d2.dispatch("raiser", {"message": "x"})
    assert r2.ok is False
    assert "secret" not in (r2.error or "")
    assert "[REDACTED]" in (r2.error or "")


def test_max_result_chars_truncates() -> None:
    def big(message: str) -> str:
        return "x" * 10000

    d = ToolRegistryDispatcher()
    d.register(
        ToolSpec(
            name="big",
            fn=big,
            schema=_ECHO_SCHEMA,
            max_result_chars=64,
        )
    )
    result = d.dispatch("big", {"message": "ignored"})
    assert result.ok is True
    assert result.truncated is True
    assert len(result.value) == 64


def test_list_tools_deterministic_order() -> None:
    d = ToolRegistryDispatcher()
    d.register(_make_echo_spec())
    d.register(ToolSpec(name="add", fn=_add, schema=_ADD_SCHEMA, description="add two"))
    tools = d.list_tools()
    names = [t["name"] for t in tools]
    assert names == ["echo", "add"]
    assert tools[0]["description"] == "Echo the input"
    assert tools[1]["schema"]["required"] == ["a", "b"]


def test_register_duplicate_rejected() -> None:
    d = ToolRegistryDispatcher()
    d.register(_make_echo_spec())
    with pytest.raises(ValueError):
        d.register(_make_echo_spec())


def test_reset_clears_state() -> None:
    d = ToolRegistryDispatcher(budget=SessionBudget(total_calls=1, total_wall_seconds=60.0))
    d.register(_make_echo_spec())
    assert d.dispatch("echo", {"message": "a"}).ok
    assert d.dispatch("echo", {"message": "b"}).ok is False  # budget hit
    d.reset()
    assert d.dispatch("echo", {"message": "c"}).ok is True
    assert len(d.audit_log()) == 1


def test_schema_depth_limit() -> None:
    # Construct deeply nested schema matching deeply nested args; should
    # reject with schema_depth_exceeded rather than blow the stack.
    schema: dict = {"type": "object", "properties": {}, "additionalProperties": False}
    cursor = schema
    for _ in range(30):
        child = {"type": "object", "properties": {}, "additionalProperties": False}
        cursor["properties"]["x"] = child
        cursor["required"] = ["x"]
        cursor = child

    args: dict = {}
    c = args
    for _ in range(30):
        c["x"] = {}
        c = c["x"]

    def noop(**_k) -> str:
        return "ok"

    d = ToolRegistryDispatcher()
    d.register(ToolSpec(name="deep", fn=noop, schema=schema))
    result = d.dispatch("deep", args)
    assert result.ok is False
    assert "schema" in (result.error or "")
