"""Unit tests for budget_bounded_loop."""

from __future__ import annotations

import pytest

from src.agent.budget_bounded_loop import (
    BudgetBoundedLoop,
    BudgetState,
    ToolInvocationBudgetError,
    _wrap_tool_registry,
)


def test_wrap_tool_registry_invokes_and_counts():
    state = BudgetState()
    calls: list[int] = []

    def add(x: int = 0) -> int:
        calls.append(1)
        return x + 1

    wrapped = _wrap_tool_registry({"add": add}, state, max_tool_invocations=3)
    assert wrapped["add"](x=1) == 2
    assert wrapped["add"](x=2) == 3
    assert wrapped["add"](x=3) == 4
    assert state.tool_invocations_used == 3
    with pytest.raises(ToolInvocationBudgetError):
        wrapped["add"](x=0)


def test_wrap_zero_budget_blocks_immediately():
    state = BudgetState()

    def echo() -> str:
        return "ok"

    wrapped = _wrap_tool_registry({"echo": echo}, state, max_tool_invocations=0)
    with pytest.raises(ToolInvocationBudgetError):
        wrapped["echo"]()


def test_wrap_rejects_bad_registry():
    with pytest.raises(TypeError):
        _wrap_tool_registry("not-a-dict", BudgetState(), 1)  # type: ignore[arg-type]


def test_wrap_rejects_negative_cap():
    with pytest.raises(ValueError):
        _wrap_tool_registry({}, BudgetState(), -1)


def _scripted(lines: list[str]):
    it = iter(lines)

    def gen(messages: list[dict]) -> str:
        return next(it)

    return gen


def test_budget_bounded_loop_respects_tool_cap_with_json_tools():
    script = [
        '{"tool_calls":[{"name":"t","arguments":{}}]}',
        '{"tool_calls":[{"name":"t","arguments":{}}]}',
        "<final_answer>done</final_answer>",
    ]

    def t() -> str:
        return "obs"

    loop = BudgetBoundedLoop(
        _scripted(script),
        {"t": t},
        max_llm_turns=8,
        max_tool_invocations=2,
        max_tool_seconds=2.0,
    )
    trace = loop.run("task")
    assert trace.status == "success"
    assert trace.final_answer == "done"
    assert loop.tool_invocations_used == 2


def test_budget_bounded_loop_tool_error_after_cap():
    script = [
        '{"tool_calls":[{"name":"t","arguments":{}}]}',
        '{"tool_calls":[{"name":"t","arguments":{}}]}',
        "<final_answer>z</final_answer>",
    ]

    def t() -> str:
        return "obs"

    loop = BudgetBoundedLoop(
        _scripted(script),
        {"t": t},
        max_llm_turns=6,
        max_tool_invocations=1,
        max_tool_seconds=2.0,
    )
    trace = loop.run("task")
    assert loop.tool_invocations_used == 1
    tool_errors = [s for s in trace.steps if s.role == "tool" and s.error]
    assert any("tool_budget_exhausted" in (e.error or "") for e in tool_errors)
    assert trace.status == "success"
    assert trace.final_answer == "z"


def test_budget_bounded_loop_final_answer_no_tools():
    def gen(messages: list[dict]) -> str:
        return "<final_answer>done</final_answer>"

    loop = BudgetBoundedLoop(gen, {}, max_llm_turns=3, max_tool_invocations=0)
    trace = loop.run("hello")
    assert trace.status == "success"
    assert trace.final_answer == "done"
    assert loop.tool_invocations_used == 0


def test_budget_bounded_loop_type_errors():
    with pytest.raises(TypeError):
        BudgetBoundedLoop("not-callable", {}, max_llm_turns=2, max_tool_invocations=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        BudgetBoundedLoop(lambda m: "x", "bad", max_llm_turns=2, max_tool_invocations=1)  # type: ignore[arg-type]


def test_budget_bounded_loop_bad_max_llm():
    with pytest.raises(ValueError):
        BudgetBoundedLoop(lambda m: "x", {}, max_llm_turns=0, max_tool_invocations=1)


def test_malicious_tool_name_still_wrapped():
    state = BudgetState()

    def weird(**kwargs):
        return "ok"

    wrapped = _wrap_tool_registry({"a;b": weird}, state, max_tool_invocations=1)
    assert wrapped["a;b"]() == "ok"
    with pytest.raises(ToolInvocationBudgetError):
        wrapped["a;b"]()
