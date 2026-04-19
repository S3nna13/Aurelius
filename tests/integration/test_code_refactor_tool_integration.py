"""Integration tests for the code refactor tool surface."""

from __future__ import annotations

import pathlib

import src.agent as agent_pkg
from src.agent import CodeRefactorTool, RefactorResult


def test_exposed_via_src_agent() -> None:
    # Additive export present.
    assert "CodeRefactorTool" in agent_pkg.__all__
    assert "RefactorResult" in agent_pkg.__all__
    assert agent_pkg.CodeRefactorTool is CodeRefactorTool


def test_prior_entries_intact() -> None:
    # Smoke-check that the pre-existing registrations are still there.
    for name in (
        "ReActLoop",
        "ToolRegistryDispatcher",
        "BeamPlanner",
        "UnifiedDiffGenerator",
        "CodeExecutionSandbox",
        "CodeTestRunner",
        "TaskDecomposer",
        "MCPClient",
    ):
        assert name in agent_pkg.__all__, name
        assert hasattr(agent_pkg, name), name


def test_rename_and_remove_unused_roundtrip(tmp_path: pathlib.Path) -> None:
    source = (
        "import os\n"
        "import sys\n"
        "\n"
        "def greet(name):\n"
        "    return 'hello ' + name\n"
        "\n"
        "print(greet('world'), sys.argv)\n"
    )
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")

    tool = CodeRefactorTool()

    # Rename `greet` -> `salute` across the module.
    renamed = tool.rename_symbol(path.read_text(encoding="utf-8"), "greet", "salute")
    assert isinstance(renamed, RefactorResult)
    assert renamed.changes >= 2
    path.write_text(renamed.new_code, encoding="utf-8")

    # Then drop the unused `os` import.
    pruned = tool.remove_unused_imports(path.read_text(encoding="utf-8"))
    assert pruned.changes == 1
    path.write_text(pruned.new_code, encoding="utf-8")

    final = path.read_text(encoding="utf-8")
    assert "salute" in final
    assert "greet" not in final
    assert "import os" not in final
    assert "import sys" in final
