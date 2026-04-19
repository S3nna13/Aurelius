"""Integration tests for the code test runner.

Confirm public exposure via :mod:`src.agent`, that prior registry
entries remain intact, and that an end-to-end pytest invocation
against a temporary directory produces a parsed :class:`TestResult`.
"""

from __future__ import annotations

import sys
import textwrap

import src.agent as agent_pkg
from src.agent import CodeTestRunner, TestResult


def test_exposed_via_src_agent() -> None:
    assert hasattr(agent_pkg, "CodeTestRunner")
    assert hasattr(agent_pkg, "TestResult")
    assert agent_pkg.CodeTestRunner is CodeTestRunner
    assert agent_pkg.TestResult is TestResult
    assert "CodeTestRunner" in agent_pkg.__all__
    assert "TestResult" in agent_pkg.__all__


def test_prior_agent_registry_intact() -> None:
    assert "xml" in agent_pkg.TOOL_CALL_PARSER_REGISTRY
    assert "json" in agent_pkg.TOOL_CALL_PARSER_REGISTRY
    assert "react" in agent_pkg.AGENT_LOOP_REGISTRY
    assert "safe_dispatch" in agent_pkg.AGENT_LOOP_REGISTRY
    assert "beam_plan" in agent_pkg.AGENT_LOOP_REGISTRY
    assert "task_decompose" in agent_pkg.AGENT_LOOP_REGISTRY
    for name in (
        "ReActLoop",
        "ToolRegistryDispatcher",
        "RepoContextPacker",
        "UnifiedDiffGenerator",
        "ShellCommandPlanner",
        "CodeExecutionSandbox",
        "TaskDecomposer",
    ):
        assert hasattr(agent_pkg, name), name


def test_end_to_end_tmp_test(tmp_path) -> None:
    (tmp_path / "test_integration_ok.py").write_text(
        textwrap.dedent(
            """
            def test_one():
                assert 1 == 1

            def test_two():
                assert sum([1, 2, 3]) == 6
            """
        ).lstrip("\n")
    )
    runner = CodeTestRunner(
        test_command=[
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            str(tmp_path),
        ],
        timeout=60.0,
        working_dir=str(tmp_path),
    )
    result = runner.run()
    assert isinstance(result, TestResult)
    assert result.passed == 2
    assert result.failed == 0
    assert result.total == 2
    assert result.duration_ms > 0.0
