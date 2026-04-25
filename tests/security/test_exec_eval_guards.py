"""Regression tests for AUR-SEC-2026-0021 (B102) and AUR-SEC-2026-0022 (B307).

Covers:
  * size-guard rejection of oversized code strings in src.eval.code_eval and
    src.inference.code_execution,
  * ast.literal parsing happy-path via src.serving.tool_executor.parse_literal,
  * rejection of non-literal code by parse_literal, and
  * AST-walker rejection of non-arithmetic expressions in _safe_eval.
"""

from __future__ import annotations

import pytest

from src.eval.code_eval import (
    _MAX_EXEC_LEN as EVAL_MAX,
    execute_code_safely,
)
from src.inference.code_execution import (
    _MAX_EXEC_LEN as EXEC_MAX,
    CodeInterpreterSession,
    ExecutionConfig,
    execute_python,
)
from src.serving.tool_executor import (
    _MAX_EXPR_LEN,
    _safe_eval,
    parse_literal,
)


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0021 -- size guard on sandboxed run sites
# ---------------------------------------------------------------------------


class TestCodeEvalSizeGuard:
    def test_oversized_solution_rejected(self) -> None:
        huge = "x = 1\n" * (EVAL_MAX // 5)
        assert len(huge) > EVAL_MAX
        passed, err = execute_code_safely(huge, ["assert True"], timeout=5.0)
        assert passed is False
        assert "too large" in err

    def test_empty_solution_rejected(self) -> None:
        passed, err = execute_code_safely("", ["assert True"], timeout=5.0)
        assert passed is False
        assert "empty" in err

    def test_normal_solution_still_works(self) -> None:
        code = "def f(x):\n    return x + 1\n"
        passed, err = execute_code_safely(code, ["assert f(1) == 2"], timeout=5.0)
        assert passed is True, err


class TestCodeExecutionSizeGuard:
    def test_execute_python_rejects_oversized(self) -> None:
        huge = "a = 1\n" * (EXEC_MAX // 5)
        assert len(huge) > EXEC_MAX
        result = execute_python(huge, ExecutionConfig())
        assert result.success is False
        assert result.error is not None
        assert "too large" in result.error

    def test_execute_python_rejects_empty(self) -> None:
        result = execute_python("", ExecutionConfig())
        assert result.success is False
        assert result.error is not None
        assert "empty" in result.error

    def test_session_rejects_oversized(self) -> None:
        session = CodeInterpreterSession(ExecutionConfig())
        huge = "a = 1\n" * (EXEC_MAX // 5)
        result = session.execute(huge)
        assert result.success is False
        assert result.error is not None
        assert "too large" in result.error

    def test_session_accepts_small_ok(self) -> None:
        session = CodeInterpreterSession(ExecutionConfig())
        result = session.execute("x = 1 + 1\nprint(x)\n")
        assert result.success is True, result.error


# ---------------------------------------------------------------------------
# AUR-SEC-2026-0022 -- literal parsing + arithmetic guard
# ---------------------------------------------------------------------------


class TestParseLiteral:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("42", 42),
            ("3.5", 3.5),
            ("'hello'", "hello"),
            ("[1, 2, 3]", [1, 2, 3]),
            ("{'a': 1}", {"a": 1}),
            ("(1, 2)", (1, 2)),
            ("True", True),
            ("None", None),
        ],
    )
    def test_valid_literals_parse(self, text: str, expected: object) -> None:
        assert parse_literal(text) == expected

    @pytest.mark.parametrize(
        "malicious",
        [
            "__import__('os').system('echo pwn')",
            "open('/etc/passwd').read()",
            "1 + 1",
            "lambda: 1",
        ],
    )
    def test_non_literal_rejected(self, malicious: str) -> None:
        with pytest.raises((ValueError, SyntaxError)):
            parse_literal(malicious)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_literal("")

    def test_oversized_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_literal("1" * (_MAX_EXPR_LEN + 1))

    def test_type_error_for_non_string(self) -> None:
        with pytest.raises(TypeError):
            parse_literal(123)  # type: ignore[arg-type]


class TestSafeEvalArithmetic:
    def test_basic_arithmetic(self) -> None:
        assert _safe_eval("2 + 3 * 4") == 14
        assert _safe_eval("-5 + 2") == -3
        assert _safe_eval("2 ** 10") == 1024

    def test_rejects_names(self) -> None:
        with pytest.raises(ValueError):
            _safe_eval("os")

    def test_rejects_calls(self) -> None:
        with pytest.raises(ValueError):
            _safe_eval("abs(-1)")

    def test_oversized_rejected(self) -> None:
        with pytest.raises(ValueError):
            _safe_eval("1+" * (_MAX_EXPR_LEN // 2 + 1) + "1")

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError):
            _safe_eval("")
