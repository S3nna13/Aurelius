"""Integration tests for the ReAct loop registry wiring."""

from __future__ import annotations

import src.agent as agent_pkg
from src.agent import AGENT_LOOP_REGISTRY, TOOL_CALL_PARSER_REGISTRY


def test_react_registered():
    assert "react" in AGENT_LOOP_REGISTRY
    # Constructible with sane defaults.
    cls = AGENT_LOOP_REGISTRY["react"]
    assert callable(cls)


def test_tool_call_parser_registry_unchanged():
    # react_loop must not have mutated the parser registry.
    assert "xml" in TOOL_CALL_PARSER_REGISTRY
    assert "json" in TOOL_CALL_PARSER_REGISTRY
    # No new keys introduced by this surface.
    assert set(TOOL_CALL_PARSER_REGISTRY.keys()) == {"xml", "json"}


def test_end_to_end_with_echo_tool():
    """Drive a live loop through the registry with a trivial echo tool."""

    ReActLoop = AGENT_LOOP_REGISTRY["react"]

    def echo(text: str) -> str:
        return f"echoed:{text}"

    script = iter(
        [
            '{"tool_calls":[{"name":"echo","arguments":{"text":"ping"}}]}',
            "<final_answer>pong</final_answer>",
        ]
    )

    def gen(messages):
        return next(script)

    loop = ReActLoop(gen, tool_registry={"echo": echo}, max_steps=4)
    trace = loop.run("please echo ping then finalize")

    assert trace.final_answer == "pong"
    assert trace.status == "success"
    # Observation step is present with the echoed output.
    obs = [s for s in trace.steps if s.role == "tool"]
    assert len(obs) == 1
    assert obs[0].tool_output == "echoed:ping"
    assert obs[0].error is None


def test_agent_step_and_trace_exported():
    # AgentStep / AgentTrace should be importable from the package root
    # so downstream harnesses need not reach into the submodule.
    assert hasattr(agent_pkg, "AgentStep")
    assert hasattr(agent_pkg, "AgentTrace")
    assert hasattr(agent_pkg, "ReActLoop")
