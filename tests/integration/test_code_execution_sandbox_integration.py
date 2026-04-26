"""Integration tests for the code execution sandbox.

Confirm public exposure via :mod:`src.agent` and basic end-to-end
``execute`` / ``execute_file`` behaviour. Also sanity-checks that prior
agent-surface registry entries remain intact.
"""

from __future__ import annotations

import pytest

import src.agent as agent_pkg
from src.agent import CodeExecutionSandbox, ExecutionResult


def test_exposed_via_src_agent() -> None:
    assert hasattr(agent_pkg, "CodeExecutionSandbox")
    assert hasattr(agent_pkg, "ExecutionResult")
    assert agent_pkg.CodeExecutionSandbox is CodeExecutionSandbox
    assert agent_pkg.ExecutionResult is ExecutionResult
    assert "CodeExecutionSandbox" in agent_pkg.__all__
    assert "ExecutionResult" in agent_pkg.__all__


def test_prior_agent_registry_intact() -> None:
    # Tool-call parsers: xml + json.
    assert "xml" in agent_pkg.TOOL_CALL_PARSER_REGISTRY
    assert "json" in agent_pkg.TOOL_CALL_PARSER_REGISTRY
    # Agent loop registries populated by earlier modules.
    assert "react" in agent_pkg.AGENT_LOOP_REGISTRY
    assert "safe_dispatch" in agent_pkg.AGENT_LOOP_REGISTRY
    assert "beam_plan" in agent_pkg.AGENT_LOOP_REGISTRY
    # Other classes still surfaced.
    for name in (
        "ReActLoop",
        "ToolRegistryDispatcher",
        "RepoContextPacker",
        "UnifiedDiffGenerator",
        "ShellCommandPlanner",
    ):
        assert hasattr(agent_pkg, name), name


def test_execute_smoke() -> None:
    sbx = CodeExecutionSandbox(timeout=5.0, max_memory_mb=128)
    res = sbx.execute("print(2 + 2)")
    assert res.exit_code == 0
    assert "4" in res.stdout
    assert res.timed_out is False
    assert res.duration_ms > 0


def test_execute_file_smoke(tmp_path) -> None:
    script = tmp_path / "hello.py"
    script.write_text(
        "import sys\nprint('argv', sys.argv[1])\nprint('isolated', 'src' not in sys.modules)\n"
    )
    sbx = CodeExecutionSandbox(timeout=5.0)
    res = sbx.execute_file(str(script), args=["world"])
    assert res.exit_code == 0
    assert "argv world" in res.stdout


@pytest.mark.parametrize(
    "snippet,expected",
    [
        ("print('ok')", "ok"),
        ("import json; print(json.dumps({'a':1}))", '"a": 1'),
    ],
)
def test_execute_various_snippets(snippet: str, expected: str) -> None:
    sbx = CodeExecutionSandbox(timeout=5.0)
    res = sbx.execute(snippet)
    assert res.exit_code == 0
    assert expected in res.stdout
