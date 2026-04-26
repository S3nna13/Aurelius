"""Unit tests for :mod:`src.agent.code_refactor_tool`."""

from __future__ import annotations

import ast

import pytest

from src.agent.code_refactor_tool import CodeRefactorTool, RefactorResult


@pytest.fixture()
def tool() -> CodeRefactorTool:
    return CodeRefactorTool()


def test_rename_symbol_module_scope(tool: CodeRefactorTool) -> None:
    result = tool.rename_symbol("x = 1\nprint(x)\n", "x", "y")
    assert isinstance(result, RefactorResult)
    assert "y = 1" in result.new_code
    assert "print(y)" in result.new_code
    assert "x" not in result.new_code.replace("print(y)", "")
    assert result.changes == 2
    assert result.operation == "rename_symbol"


def test_rename_symbol_function_scope(tool: CodeRefactorTool) -> None:
    src = "def a():\n    foo = 1\n    return foo\ndef b():\n    foo = 2\n    return foo\n"
    result = tool.rename_symbol(src, "foo", "bar", scope="function")
    # Only the first function that defines/uses foo is renamed.
    tree = ast.parse(result.new_code)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    src_a = ast.unparse(fns["a"])
    src_b = ast.unparse(fns["b"])
    assert "bar" in src_a and "foo" not in src_a
    assert "foo" in src_b and "bar" not in src_b


def test_inline_variable_replaces_refs(tool: CodeRefactorTool) -> None:
    src = "x = 1 + 2\nprint(x)\ny = x + 3\n"
    result = tool.inline_variable(src, "x")
    assert "x =" not in result.new_code  # binding gone
    assert result.new_code.count("1 + 2") == 2
    assert result.changes >= 2


def test_inline_variable_multiple_assignments_warns(tool: CodeRefactorTool) -> None:
    src = "x = 1\nx = 2\nprint(x)\n"
    result = tool.inline_variable(src, "x")
    assert result.new_code == src
    assert result.changes == 0
    assert result.warnings
    assert any("assignments" in w for w in result.warnings)


def test_remove_unused_imports_drops_unused(tool: CodeRefactorTool) -> None:
    src = "import os\nimport sys\nprint(sys.argv)\n"
    result = tool.remove_unused_imports(src)
    assert "import os" not in result.new_code
    assert "import sys" in result.new_code
    assert result.changes == 1


def test_remove_unused_imports_preserves_used(tool: CodeRefactorTool) -> None:
    src = "import json\nprint(json.dumps({}))\n"
    result = tool.remove_unused_imports(src)
    assert "import json" in result.new_code
    assert result.changes == 0


def test_extract_function_creates_named_fn(tool: CodeRefactorTool) -> None:
    src = "a = 1\nb = 2\nc = a + b\nprint(c)\n"
    result = tool.extract_function(src, 1, 3, "build")
    assert "def build" in result.new_code
    assert "build()" in result.new_code
    # Original trailing print(c) must still be present.
    assert "print(c)" in result.new_code
    assert result.changes == 3


def test_add_type_hint_annotates(tool: CodeRefactorTool) -> None:
    result = tool.add_type_hint("def f(x):\n    return x\n", "f", "x", "int")
    assert "def f(x: int)" in result.new_code
    assert result.changes == 1


def test_malformed_syntax_raises(tool: CodeRefactorTool) -> None:
    with pytest.raises(SyntaxError):
        tool.rename_symbol("def :", "a", "b")


def test_rename_nothing_to_rename_returns_original(tool: CodeRefactorTool) -> None:
    src = "x = 1\nprint(x)\n"
    result = tool.rename_symbol(src, "zzz", "yyy")
    assert result.new_code == src
    assert result.changes == 0
    assert result.warnings


def test_rename_does_not_touch_strings_or_comments(tool: CodeRefactorTool) -> None:
    src = 'x = 1\n# comment mentions x\ns = "x in a string"\nprint(x)\n'
    result = tool.rename_symbol(src, "x", "y")
    # String literal contents unchanged (ast.unparse may re-quote).
    assert "x in a string" in result.new_code
    # Comments are stripped by ast.unparse, but we at least must not
    # have rewritten the string contents into "y in a string".
    assert "y in a string" not in result.new_code


def test_determinism(tool: CodeRefactorTool) -> None:
    src = "a = 1\nb = a + 2\nprint(a, b)\n"
    r1 = tool.rename_symbol(src, "a", "q")
    r2 = tool.rename_symbol(src, "a", "q")
    assert r1.new_code == r2.new_code
    assert r1.changes == r2.changes


def test_changes_count_correct(tool: CodeRefactorTool) -> None:
    src = "x = 1\ny = x + x\nz = x * x * x\n"
    result = tool.rename_symbol(src, "x", "w")
    # 1 (assign target) + 2 (y=) + 3 (z=) = 6 occurrences of x
    assert result.changes == 6


def test_ast_unparse_available() -> None:
    # Required by our implementation; skip if (hypothetically) absent.
    if not hasattr(ast, "unparse"):
        pytest.skip("ast.unparse not available on this interpreter")
    assert callable(ast.unparse)


def test_large_file_handled(tool: CodeRefactorTool) -> None:
    # 1000-line synthetic file: alternating assignments and prints.
    lines = []
    for i in range(500):
        lines.append(f"v{i} = {i}")
        lines.append(f"print(v{i})")
    src = "\n".join(lines) + "\n"
    # Rename a single mid-file symbol; must succeed without error.
    result = tool.rename_symbol(src, "v250", "renamed")
    assert result.changes == 2
    assert "renamed = 250" in result.new_code


def test_add_type_hint_missing_param_warns(tool: CodeRefactorTool) -> None:
    result = tool.add_type_hint("def f(x):\n    return x\n", "f", "missing", "int")
    assert result.changes == 0
    assert result.warnings


def test_remove_unused_imports_from_import(tool: CodeRefactorTool) -> None:
    src = "from os import path, getcwd\nprint(path.join('a', 'b'))\n"
    result = tool.remove_unused_imports(src)
    assert "path" in result.new_code
    assert "getcwd" not in result.new_code
    assert result.changes == 1
