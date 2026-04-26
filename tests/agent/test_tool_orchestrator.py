"""Tests for src/agent/tool_orchestrator.py."""

from src.agent.tool_orchestrator import (
    TOOL_ORCHESTRATOR,
    ToolCall,
    ToolCallOutcome,
    ToolOrchestrator,
)

# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


def test_toolcall_auto_id_is_8_chars():
    tc = ToolCall(tool_name="my_tool", arguments={})
    assert len(tc.id) == 8


def test_toolcall_auto_id_is_hex():
    tc = ToolCall(tool_name="my_tool", arguments={})
    int(tc.id, 16)  # raises ValueError if not valid hex


def test_toolcall_ids_are_unique():
    ids = {ToolCall(tool_name="t", arguments={}).id for _ in range(20)}
    assert len(ids) > 1  # overwhelmingly likely all unique


def test_toolcall_default_max_retries():
    tc = ToolCall(tool_name="t", arguments={})
    assert tc.max_retries == 2


def test_toolcall_stores_tool_name():
    tc = ToolCall(tool_name="search", arguments={"q": "hello"})
    assert tc.tool_name == "search"


def test_toolcall_stores_arguments():
    args = {"x": 1, "y": 2}
    tc = ToolCall(tool_name="t", arguments=args)
    assert tc.arguments == args


def test_toolcall_custom_max_retries():
    tc = ToolCall(tool_name="t", arguments={}, max_retries=5)
    assert tc.max_retries == 5


def test_toolcall_custom_id():
    tc = ToolCall(tool_name="t", arguments={}, id="abcdef12")
    assert tc.id == "abcdef12"


# ---------------------------------------------------------------------------
# ToolCallOutcome
# ---------------------------------------------------------------------------


def test_outcome_fields_stored():
    o = ToolCallOutcome(call_id="abc", tool_name="t", success=True, output="ok")
    assert o.call_id == "abc"
    assert o.tool_name == "t"
    assert o.success is True
    assert o.output == "ok"


def test_outcome_default_attempts():
    o = ToolCallOutcome(call_id="x", tool_name="t", success=True, output="")
    assert o.attempts == 1


def test_outcome_default_error():
    o = ToolCallOutcome(call_id="x", tool_name="t", success=True, output="")
    assert o.error == ""


def test_outcome_custom_error():
    o = ToolCallOutcome(call_id="x", tool_name="t", success=False, output="", error="oops")
    assert o.error == "oops"


# ---------------------------------------------------------------------------
# ToolOrchestrator.build_call
# ---------------------------------------------------------------------------


def test_build_call_returns_tool_call():
    orch = ToolOrchestrator()
    tc = orch.build_call("my_tool", x=1, y=2)
    assert isinstance(tc, ToolCall)


def test_build_call_sets_tool_name():
    orch = ToolOrchestrator()
    tc = orch.build_call("search", query="foo")
    assert tc.tool_name == "search"


def test_build_call_sets_arguments():
    orch = ToolOrchestrator()
    tc = orch.build_call("calc", a=3, b=4)
    assert tc.arguments == {"a": 3, "b": 4}


def test_build_call_generates_id():
    orch = ToolOrchestrator()
    tc = orch.build_call("t")
    assert len(tc.id) == 8


# ---------------------------------------------------------------------------
# dispatch: success
# ---------------------------------------------------------------------------


def test_dispatch_success():
    orch = ToolOrchestrator()
    tc = ToolCall(tool_name="add", arguments={"a": 1, "b": 2})
    outcome = orch.dispatch(tc, lambda a, b: a + b)
    assert outcome.success is True


def test_dispatch_success_output():
    orch = ToolOrchestrator()
    tc = ToolCall(tool_name="add", arguments={"a": 1, "b": 2})
    outcome = orch.dispatch(tc, lambda a, b: a + b)
    assert outcome.output == "3"


def test_dispatch_success_attempts_one():
    orch = ToolOrchestrator()
    tc = ToolCall(tool_name="t", arguments={})
    outcome = orch.dispatch(tc, lambda: "ok")
    assert outcome.attempts == 1


def test_dispatch_call_id_matches():
    orch = ToolOrchestrator()
    tc = ToolCall(tool_name="t", arguments={}, id="deadbeef")
    outcome = orch.dispatch(tc, lambda: "x")
    assert outcome.call_id == "deadbeef"


def test_dispatch_tool_name_matches():
    orch = ToolOrchestrator()
    tc = ToolCall(tool_name="my_func", arguments={})
    outcome = orch.dispatch(tc, lambda: None)
    assert outcome.tool_name == "my_func"


# ---------------------------------------------------------------------------
# dispatch: retry on transient failure
# ---------------------------------------------------------------------------


def test_dispatch_retries_once_and_succeeds():
    orch = ToolOrchestrator()
    counter = {"n": 0}

    def flaky():
        counter["n"] += 1
        if counter["n"] < 2:
            raise RuntimeError("transient")
        return "ok"

    tc = ToolCall(tool_name="t", arguments={}, max_retries=2)
    outcome = orch.dispatch(tc, flaky)
    assert outcome.success is True
    assert outcome.attempts == 2


def test_dispatch_all_failures():
    orch = ToolOrchestrator()

    def always_fail():
        raise ValueError("boom")

    tc = ToolCall(tool_name="t", arguments={}, max_retries=2)
    outcome = orch.dispatch(tc, always_fail)
    assert outcome.success is False


def test_dispatch_all_failures_attempts():
    orch = ToolOrchestrator()

    def always_fail():
        raise ValueError("boom")

    tc = ToolCall(tool_name="t", arguments={}, max_retries=2)
    outcome = orch.dispatch(tc, always_fail)
    assert outcome.attempts == 3  # max_retries+1


def test_dispatch_all_failures_error_stored():
    orch = ToolOrchestrator()

    def always_fail():
        raise ValueError("specific error")

    tc = ToolCall(tool_name="t", arguments={}, max_retries=1)
    outcome = orch.dispatch(tc, always_fail)
    assert "specific error" in outcome.error


def test_dispatch_zero_retries_one_attempt_on_fail():
    orch = ToolOrchestrator()

    def always_fail():
        raise RuntimeError("x")

    tc = ToolCall(tool_name="t", arguments={}, max_retries=0)
    outcome = orch.dispatch(tc, always_fail)
    assert outcome.attempts == 1


# ---------------------------------------------------------------------------
# dispatch_batch
# ---------------------------------------------------------------------------


def test_dispatch_batch_returns_list():
    orch = ToolOrchestrator()
    calls = [
        ToolCall(tool_name="add", arguments={"a": 1, "b": 2}),
        ToolCall(tool_name="add", arguments={"a": 3, "b": 4}),
    ]
    outcomes = orch.dispatch_batch(calls, {"add": lambda a, b: a + b})
    assert isinstance(outcomes, list)
    assert len(outcomes) == 2


def test_dispatch_batch_all_success():
    orch = ToolOrchestrator()
    calls = [ToolCall(tool_name="echo", arguments={"msg": "hi"})]
    outcomes = orch.dispatch_batch(calls, {"echo": lambda msg: msg})
    assert outcomes[0].success is True


def test_dispatch_batch_empty():
    orch = ToolOrchestrator()
    outcomes = orch.dispatch_batch([], {})
    assert outcomes == []


def test_dispatch_batch_mixed_tools():
    orch = ToolOrchestrator()
    calls = [
        ToolCall(tool_name="double", arguments={"x": 3}),
        ToolCall(tool_name="negate", arguments={"x": 5}),
    ]
    handlers = {
        "double": lambda x: x * 2,
        "negate": lambda x: -x,
    }
    outcomes = orch.dispatch_batch(calls, handlers)
    assert outcomes[0].output == "6"
    assert outcomes[1].output == "-5"


# ---------------------------------------------------------------------------
# success_rate
# ---------------------------------------------------------------------------


def test_success_rate_all_success():
    outcomes = [
        ToolCallOutcome(call_id="a", tool_name="t", success=True, output=""),
        ToolCallOutcome(call_id="b", tool_name="t", success=True, output=""),
    ]
    assert ToolOrchestrator.success_rate(outcomes) == 1.0


def test_success_rate_empty():
    assert ToolOrchestrator.success_rate([]) == 1.0


def test_success_rate_half():
    outcomes = [
        ToolCallOutcome(call_id="a", tool_name="t", success=True, output=""),
        ToolCallOutcome(call_id="b", tool_name="t", success=False, output=""),
    ]
    assert ToolOrchestrator.success_rate(outcomes) == 0.5


def test_success_rate_all_failed():
    outcomes = [
        ToolCallOutcome(call_id="a", tool_name="t", success=False, output=""),
        ToolCallOutcome(call_id="b", tool_name="t", success=False, output=""),
    ]
    assert ToolOrchestrator.success_rate(outcomes) == 0.0


# ---------------------------------------------------------------------------
# failed_outcomes
# ---------------------------------------------------------------------------


def test_failed_outcomes_returns_only_failed():
    outcomes = [
        ToolCallOutcome(call_id="a", tool_name="t", success=True, output=""),
        ToolCallOutcome(call_id="b", tool_name="t", success=False, output=""),
        ToolCallOutcome(call_id="c", tool_name="t", success=False, output=""),
    ]
    failed = ToolOrchestrator.failed_outcomes(outcomes)
    assert len(failed) == 2
    assert all(not o.success for o in failed)


def test_failed_outcomes_empty_if_all_success():
    outcomes = [
        ToolCallOutcome(call_id="a", tool_name="t", success=True, output=""),
    ]
    assert ToolOrchestrator.failed_outcomes(outcomes) == []


def test_failed_outcomes_empty_input():
    assert ToolOrchestrator.failed_outcomes([]) == []


# ---------------------------------------------------------------------------
# TOOL_ORCHESTRATOR singleton
# ---------------------------------------------------------------------------


def test_tool_orchestrator_singleton_exists():
    assert TOOL_ORCHESTRATOR is not None


def test_tool_orchestrator_singleton_is_instance():
    assert isinstance(TOOL_ORCHESTRATOR, ToolOrchestrator)


def test_tool_orchestrator_singleton_build_call():
    tc = TOOL_ORCHESTRATOR.build_call("ping")
    assert isinstance(tc, ToolCall)
