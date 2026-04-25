"""Tests for src.security.sandbox_executor."""

from __future__ import annotations

import pytest

from src.security.sandbox_executor import (
    SandboxConfig,
    SandboxExecutor,
    SandboxViolation,
)


def test_simple_arithmetic_executes():
    runner = SandboxExecutor()
    result = runner.execute("x = 2 + 3\nprint(x)")
    assert result.timed_out is False
    assert result.exception is None
    assert result.stdout.strip() == "5"


def test_timeout_returns_timed_out_true():
    runner = SandboxExecutor()
    cfg = SandboxConfig(timeout_seconds=0.25)
    code = "i = 0\nwhile True:\n    i = i + 1\n"
    result = runner.execute(code, cfg)
    assert result.timed_out is True
    assert result.exception is not None
    assert "Timeout" in result.exception


def test_oversized_code_raises_SandboxViolation():
    runner = SandboxExecutor()
    cfg = SandboxConfig(max_code_len=100)
    big = "x = 1\n" * 200
    with pytest.raises(SandboxViolation) as info:
        runner.execute(big, cfg)
    assert "max_code_len" in info.value.reason
    assert len(info.value.code_snippet) <= 200


def test_blocked_import_raises_or_errors():
    runner = SandboxExecutor()
    result = runner.execute("import os\nprint(os.getcwd())")
    assert result.exception is not None
    assert "getcwd" not in result.stdout


def test_blocked_dynamic_primitives_missing():
    runner = SandboxExecutor()
    # Each snippet references a name that must NOT be reachable from the
    # restricted builtins.
    snippets = [
        "handle = open('/etc/passwd')",
        "probe = __import__('os')",
        "fn = compile('1+1', 'x', 'eval')",
    ]
    for snippet in snippets:
        result = runner.execute(snippet)
        assert result.exception is not None, f"expected failure for {snippet!r}"


def test_stdout_captured():
    runner = SandboxExecutor()
    result = runner.execute("print('hello'); print('world')")
    assert result.exception is None
    assert "hello" in result.stdout
    assert "world" in result.stdout


def test_getattr_type_traversal_escape_blocked():
    runner = SandboxExecutor()
    traversal_snippets = [
        "getattr(object, '__class__')",
        "setattr(object, '__class__', int)",
        "type(object)",
        "hasattr(object, '__class__')",
        "[c for c in type.__subclasses__(type)]",
        "getattr(int, '__subclasses__')()",
    ]
    for snippet in traversal_snippets:
        result = runner.execute(snippet)
        assert result.exception is not None, (
            f"expected sandbox violation for {snippet!r} but got none"
        )


def test_default_allowed_builtins_excludes_getattr_type():
    from src.security.sandbox_executor import DEFAULT_ALLOWED_BUILTINS

    for name in ("getattr", "setattr", "type", "hasattr"):
        assert name not in DEFAULT_ALLOWED_BUILTINS, (
            f"{name} must not be in DEFAULT_ALLOWED_BUILTINS "
            f"(sandbox escape vector — AUR-SEC-2026-0027)"
        )
