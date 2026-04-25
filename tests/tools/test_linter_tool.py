import tempfile
from pathlib import Path
import pytest
from src.tools.linter_tool import LinterTool, LintIssue, LintResult, LINTER_REGISTRY


def test_clean_code_passes():
    tool = LinterTool()
    result = tool.lint("x = 1\ny = x + 2\n")
    assert result.passed is True
    assert result.issues == ()


def test_syntax_error_detected():
    tool = LinterTool()
    result = tool.lint("def foo(:\n    pass\n")
    assert result.passed is False
    assert any(i.code == "E001" for i in result.issues)


def test_bare_except_detected():
    tool = LinterTool()
    src = "try:\n    pass\nexcept:\n    pass\n"
    result = tool.lint(src)
    assert any(i.code == "W001" for i in result.issues)


def test_assert_detected():
    tool = LinterTool()
    result = tool.lint("assert x == 1\n")
    assert any(i.code == "W002" for i in result.issues)


def test_print_detected():
    tool = LinterTool()
    result = tool.lint('print("hello")\n')
    assert any(i.code == "W003" for i in result.issues)


def test_multiple_issues():
    tool = LinterTool()
    src = 'print("x")\nassert True\ntry:\n    pass\nexcept:\n    pass\n'
    result = tool.lint(src)
    codes = {i.code for i in result.issues}
    assert "W001" in codes
    assert "W002" in codes
    assert "W003" in codes
    assert result.passed is False


def test_lint_file_reads_path():
    tool = LinterTool()
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write('print("hi")\n')
        name = f.name
    result = tool.lint_file(name)
    assert any(i.code == "W003" for i in result.issues)


def test_lint_file_clean():
    tool = LinterTool()
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("x = 1\n")
        name = f.name
    result = tool.lint_file(name)
    assert result.passed is True


def test_registry_key():
    assert "default" in LINTER_REGISTRY
    assert LINTER_REGISTRY["default"] is LinterTool
