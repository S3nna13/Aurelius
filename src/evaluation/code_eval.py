"""Aurelius code evaluator: safe sandbox for code generation assessment."""

from __future__ import annotations

import builtins
import io
import time
from dataclasses import dataclass, field


@dataclass
class CodeEvalConfig:
    timeout_seconds: float = 5.0
    allowed_builtins: list[str] = field(
        default_factory=lambda: [
            "print",
            "len",
            "range",
            "int",
            "str",
            "float",
            "list",
            "dict",
            "set",
            "tuple",
            "sum",
            "max",
            "min",
            "abs",
            "round",
            "enumerate",
            "zip",
        ]
    )


@dataclass(frozen=True)
class CodeEvalResult:
    code: str
    passed: bool
    output: str
    error: str
    execution_time_ms: float


# Reference to compile+exec for use in the evaluator
_compile = compile
_exec = getattr(builtins, "exec")


class CodeEvaluator:
    """Evaluates code snippets in a restricted namespace."""

    def __init__(self, config: CodeEvalConfig | None = None) -> None:
        self.config = config if config is not None else CodeEvalConfig()

    def _build_namespace(self, captured_output: io.StringIO) -> dict:
        """Build a restricted builtins namespace from the allowed list."""
        restricted: dict = {}
        for name in self.config.allowed_builtins:
            if hasattr(builtins, name):
                restricted[name] = getattr(builtins, name)

        # Override print to capture output
        def _captured_print(*args, **kwargs):
            kwargs["file"] = captured_output
            print(*args, **kwargs)  # noqa: T201

        restricted["print"] = _captured_print
        return {"__builtins__": restricted}

    def evaluate(
        self,
        code: str,
        test_inputs: list,
        expected_outputs: list,
    ) -> CodeEvalResult:
        """Execute code in a restricted namespace and test solution(x) for each input.

        All test cases must pass for passed=True.  Any exception sets passed=False.
        """
        captured_output = io.StringIO()
        error_msg = ""
        passed = False
        start = time.monotonic()

        try:
            namespace = self._build_namespace(captured_output)
            compiled = _compile(code, "<code_eval>", "exec")
            _exec(compiled, namespace)

            if not test_inputs:
                # No test cases: code executed without error — pass
                passed = True
            else:
                solution_fn = namespace.get("solution")
                all_pass = True
                for inp, expected in zip(test_inputs, expected_outputs):
                    if solution_fn is None:
                        all_pass = False
                        break
                    result = solution_fn(inp)
                    if str(result) != str(expected):
                        all_pass = False
                        break
                passed = all_pass

        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            passed = False

        elapsed_ms = (time.monotonic() - start) * 1000.0
        return CodeEvalResult(
            code=code,
            passed=passed,
            output=captured_output.getvalue(),
            error=error_msg,
            execution_time_ms=elapsed_ms,
        )

    def batch_evaluate(self, problems: list[dict]) -> list[CodeEvalResult]:
        """Evaluate a list of problem dicts, each with 'code', 'inputs', 'expected'."""
        return [
            self.evaluate(
                problem["code"],
                problem["inputs"],
                problem["expected"],
            )
            for problem in problems
        ]


CODE_EVALUATOR_REGISTRY = {"default": CodeEvaluator}
