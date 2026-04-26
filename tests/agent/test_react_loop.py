"""Unit tests for src.agent.react_loop.ReActLoop."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

from src.agent.react_loop import ReActLoop


def _scripted(responses: list[str]):
    """Return a generate_fn that yields one scripted response per call."""
    it: Iterator[str] = iter(responses)
    calls: list[list[dict]] = []

    def fn(messages: list[dict]) -> str:
        calls.append(list(messages))
        try:
            return next(it)
        except StopIteration:
            return "...stuck..."

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_one_step_final_answer_xml():
    gen = _scripted(["<final_answer>hi</final_answer>"])
    loop = ReActLoop(gen, tool_registry={})
    trace = loop.run("say hi")
    assert trace.status == "success"
    assert trace.final_answer == "hi"
    assert trace.steps_used == 1


def test_one_step_final_answer_prefix():
    gen = _scripted(["thinking...\nFinal Answer: 42"])
    loop = ReActLoop(gen, tool_registry={})
    trace = loop.run("compute")
    assert trace.status == "success"
    assert trace.final_answer == "42"
    assert trace.steps_used == 1


def test_tool_call_flow_json_then_final():
    def echo(msg: str) -> str:
        return f"ECHO:{msg}"

    script = [
        '{"tool_calls":[{"name":"echo","arguments":{"msg":"hello"}}]}',
        "<final_answer>done</final_answer>",
    ]
    gen = _scripted(script)
    loop = ReActLoop(gen, tool_registry={"echo": echo})
    trace = loop.run("use the echo tool")
    assert trace.status == "success"
    assert trace.final_answer == "done"
    # assistant, tool(observation), assistant
    assert len(trace.steps) == 3
    assert trace.steps[0].role == "assistant"
    assert trace.steps[0].tool_name == "echo"
    assert trace.steps[1].role == "tool"
    assert trace.steps[1].tool_output == "ECHO:hello"
    assert trace.steps[1].error is None
    assert trace.steps_used == 2


def test_unknown_tool_error_does_not_crash():
    script = [
        '{"tool_calls":[{"name":"nope","arguments":{}}]}',
        "<final_answer>ok</final_answer>",
    ]
    gen = _scripted(script)
    loop = ReActLoop(gen, tool_registry={})
    trace = loop.run("try missing tool")
    assert trace.status == "success"
    obs = next(s for s in trace.steps if s.role == "tool")
    assert obs.error is not None
    assert "unknown_tool" in obs.error


def test_tool_exception_captured():
    def boom():
        raise RuntimeError("kaboom")

    script = [
        '{"tool_calls":[{"name":"boom","arguments":{}}]}',
        "<final_answer>after</final_answer>",
    ]
    gen = _scripted(script)
    loop = ReActLoop(gen, tool_registry={"boom": boom})
    trace = loop.run("call boom")
    obs = next(s for s in trace.steps if s.role == "tool")
    assert obs.error is not None
    assert "RuntimeError" in obs.error and "kaboom" in obs.error


def test_tool_timeout_captured():
    def slow():
        time.sleep(10.0)
        return "never"

    script = [
        '{"tool_calls":[{"name":"slow","arguments":{}}]}',
        "<final_answer>moved on</final_answer>",
    ]
    gen = _scripted(script)
    loop = ReActLoop(gen, tool_registry={"slow": slow}, max_tool_seconds=0.1)
    trace = loop.run("call slow")
    obs = next(s for s in trace.steps if s.role == "tool")
    assert obs.error is not None
    assert "timeout" in obs.error.lower()
    assert trace.status == "success"  # loop kept going


def test_budget_exhausted_status():
    # Model never finalises or calls tools: every turn is bare reasoning.
    gen = _scripted(["thinking 1", "thinking 2", "thinking 3"])
    loop = ReActLoop(gen, tool_registry={}, max_steps=2)
    trace = loop.run("forever")
    assert trace.status == "budget"
    assert trace.steps_used == 2
    assert trace.final_answer is None


def test_malformed_tool_call_json_captured():
    # Leads with { and mentions tool_calls so JSON dispatch fires, but the
    # envelope is truncated garbage -> parse error.
    script = [
        '{"tool_calls": [{"name": "x", "arguments": {not json here}}]}',
        "<final_answer>recovered</final_answer>",
    ]
    gen = _scripted(script)
    loop = ReActLoop(gen, tool_registry={})
    trace = loop.run("attempt malformed")
    # First assistant step should carry a parse error annotation.
    assert trace.steps[0].role == "assistant"
    assert trace.steps[0].error is not None
    assert "tool_call_parse_error" in trace.steps[0].error
    assert trace.status == "success"
    assert trace.final_answer == "recovered"


def test_empty_task_returns_no_answer():
    gen = _scripted(["<final_answer>should not be asked</final_answer>"])
    loop = ReActLoop(gen, tool_registry={})
    trace = loop.run("")
    assert trace.status == "no_answer"
    assert trace.final_answer is None
    assert trace.steps_used == 0


def test_determinism_same_input_same_trace():
    def make():
        return _scripted(
            [
                '{"tool_calls":[{"name":"echo","arguments":{"msg":"x"}}]}',
                "<final_answer>ok</final_answer>",
            ]
        )

    def echo(msg: str) -> str:
        return f"E:{msg}"

    t1 = ReActLoop(make(), tool_registry={"echo": echo}).run("go")
    t2 = ReActLoop(make(), tool_registry={"echo": echo}).run("go")
    assert t1.status == t2.status
    assert t1.final_answer == t2.final_answer
    assert [(s.role, s.content, s.error) for s in t1.steps] == [
        (s.role, s.content, s.error) for s in t2.steps
    ]


def test_system_prompt_reaches_generate_fn():
    gen = _scripted(["<final_answer>yes</final_answer>"])
    loop = ReActLoop(gen, tool_registry={})
    loop.run("task", system_prompt="you are helpful")
    first_call_messages = gen.calls[0]  # type: ignore[attr-defined]
    assert first_call_messages[0] == {
        "role": "system",
        "content": "you are helpful",
    }
    assert first_call_messages[1] == {"role": "user", "content": "task"}


def test_empty_registry_allows_pure_answer():
    gen = _scripted(["<final_answer>only reasoning needed</final_answer>"])
    loop = ReActLoop(gen, tool_registry={})
    trace = loop.run("no tools")
    assert trace.status == "success"
    assert trace.final_answer == "only reasoning needed"


def test_xml_tool_call_format_parsed():
    def add(a: int, b: int) -> str:
        return str(a + b)

    script = [
        '<tool_use name="add"><input>{"a":2,"b":3}</input></tool_use>',
        "<final_answer>5</final_answer>",
    ]
    gen = _scripted(script)
    loop = ReActLoop(gen, tool_registry={"add": add})
    trace = loop.run("add two numbers")
    assert trace.status == "success"
    obs = next(s for s in trace.steps if s.role == "tool")
    assert obs.tool_output == "5"
    assert obs.error is None


def test_invalid_arguments_rejected_by_signature():
    def only_a(a: int) -> str:
        return str(a)

    script = [
        '{"tool_calls":[{"name":"only_a","arguments":{"a":1,"evil":"x"}}]}',
        "<final_answer>eh</final_answer>",
    ]
    gen = _scripted(script)
    loop = ReActLoop(gen, tool_registry={"only_a": only_a})
    trace = loop.run("bad args")
    obs = next(s for s in trace.steps if s.role == "tool")
    assert obs.error is not None
    assert "invalid_arguments" in obs.error


def test_generate_fn_exception_surfaces_as_error_status():
    def boom(messages):
        raise ValueError("model down")

    loop = ReActLoop(boom, tool_registry={})
    trace = loop.run("hi")
    assert trace.status == "error"
    assert trace.steps[-1].error is not None
    assert "model down" in trace.steps[-1].error


def test_constructor_validation():
    with pytest.raises(TypeError):
        ReActLoop("not callable", tool_registry={})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ReActLoop(lambda m: "", tool_registry=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ReActLoop(lambda m: "", tool_registry={}, max_steps=0)
    with pytest.raises(ValueError):
        ReActLoop(lambda m: "", tool_registry={}, max_tool_seconds=0)
