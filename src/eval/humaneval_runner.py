"""HumanEval Runner: execute and grade code-generation solutions.

Chen et al. (2021) "Evaluating Large Language Models Trained on Code"
(arXiv:2107.03374).  This module provides:

  * ``HumanEvalProblem``  — problem dataclass mirroring the HumanEval format
  * ``ExecutionResult``   — result of running a solution
  * ``HumanEvalRunner``   — load, execute, and batch-evaluate solutions
  * ``BENCHMARK_REGISTRY["humaneval"]`` — singleton runner instance

Execution is done in a sandboxed subprocess.  Only stdlib modules from a
curated ALLOW_LIST are accessible inside the executed code.

Dependencies: stdlib only (subprocess, textwrap, time, ast).
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HumanEvalProblem:
    """A single HumanEval problem.

    Attributes:
        task_id:            Unique identifier, e.g. ``"HumanEval/0"``.
        prompt:             The function signature and docstring shown to the model.
        canonical_solution: Reference implementation.
        test:               Test harness code (calls ``check(candidate)``).
        entry_point:        Name of the function to evaluate.
    """

    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str


@dataclass
class ExecutionResult:
    """Outcome of executing a solution against the test harness.

    Attributes:
        task_id:    Problem identifier.
        passed:     ``True`` iff the solution passed all tests.
        error:      Error message if execution failed, else ``None``.
        runtime_ms: Wall-clock execution time in milliseconds.
    """

    task_id: str
    passed: bool
    error: str | None
    runtime_ms: float


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# Modules that the sandboxed subprocess is permitted to import.
_ALLOW_LIST = frozenset(
    [
        "math",
        "cmath",
        "decimal",
        "fractions",
        "statistics",
        "itertools",
        "functools",
        "operator",
        "collections",
        "heapq",
        "bisect",
        "string",
        "re",
        "difflib",
        "textwrap",
        "typing",
        "types",
        "copy",
        "pprint",
        "enum",
        "dataclasses",
        "abc",
        "contextlib",
        "io",
        "json",
    ]
)

# Script template executed in the child process.
_SCRIPT_TEMPLATE = """\
import sys as _sys

# Block dangerous imports
_ALLOW_LIST = {allow_list!r}
_real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

def _safe_import(name, *args, **kwargs):
    top = name.split('.')[0]
    if top not in _ALLOW_LIST and top not in ('builtins', '__future__'):
        raise ImportError(f"Import '{{name}}' is not allowed in sandbox")
    return _real_import(name, *args, **kwargs)

import builtins as _builtins
_builtins.__import__ = _safe_import

# ---- solution ----
{solution}

# ---- tests ----
{test}

check({entry_point})
"""


class HumanEvalRunner:
    """Load HumanEval problems and evaluate model-generated solutions.

    Usage::

        runner = HumanEvalRunner()
        problems = runner.load_problems()          # 3 built-in stubs
        result = runner.execute_solution(problems[0], "def add(a, b): return a + b")
        metrics = runner.batch_evaluate(problems, solutions)
    """

    # ------------------------------------------------------------------
    # Problem loading
    # ------------------------------------------------------------------

    def load_problems(self, path: str | None = None) -> list[HumanEvalProblem]:
        """Load HumanEval problems from *path* (JSONL format), or return stubs.

        If *path* is ``None`` or the file cannot be read, returns 3 built-in
        stub problems suitable for unit-testing without downloading the
        full benchmark dataset.

        Args:
            path: Optional path to a HumanEval JSONL file.

        Returns:
            List of ``HumanEvalProblem`` instances.
        """
        if path is not None:
            try:
                import json  # stdlib

                problems = []
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        problems.append(
                            HumanEvalProblem(
                                task_id=obj["task_id"],
                                prompt=obj["prompt"],
                                canonical_solution=obj["canonical_solution"],
                                test=obj["test"],
                                entry_point=obj["entry_point"],
                            )
                        )
                if problems:
                    return problems
            except Exception:  # noqa: S110
                pass

        return self._stub_problems()

    @staticmethod
    def _stub_problems() -> list[HumanEvalProblem]:
        """Return 3 minimal HumanEval-style stub problems."""
        return [
            HumanEvalProblem(
                task_id="HumanEval/stub_0",
                prompt=('def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n'),
                canonical_solution="    return a + b\n",
                test=(
                    "def check(candidate):\n"
                    "    assert candidate(1, 2) == 3\n"
                    "    assert candidate(-1, 1) == 0\n"
                    "    assert candidate(0, 0) == 0\n"
                ),
                entry_point="add",
            ),
            HumanEvalProblem(
                task_id="HumanEval/stub_1",
                prompt=('def is_even(n: int) -> bool:\n    """Return True if n is even."""\n'),
                canonical_solution="    return n % 2 == 0\n",
                test=(
                    "def check(candidate):\n"
                    "    assert candidate(2) is True\n"
                    "    assert candidate(3) is False\n"
                    "    assert candidate(0) is True\n"
                ),
                entry_point="is_even",
            ),
            HumanEvalProblem(
                task_id="HumanEval/stub_2",
                prompt=(
                    "def maximum(lst: list) -> float:\n"
                    '    """Return the maximum element in lst."""\n'
                ),
                canonical_solution="    return max(lst)\n",
                test=(
                    "def check(candidate):\n"
                    "    assert candidate([1, 3, 2]) == 3\n"
                    "    assert candidate([-1, -2, -3]) == -1\n"
                    "    assert candidate([7]) == 7\n"
                ),
                entry_point="maximum",
            ),
        ]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_solution(
        self,
        problem: HumanEvalProblem,
        solution: str,
        timeout: float = 5.0,
    ) -> ExecutionResult:
        """Execute *solution* against *problem*'s test harness in a subprocess.

        The child process is given the concatenation of:
            1. The problem ``prompt`` (function signature + docstring)
            2. The *solution* body
            3. The problem ``test`` harness
            4. A call to ``check(<entry_point>)``

        Only modules in ``_ALLOW_LIST`` may be imported inside the sandbox.

        Args:
            problem:  The problem to evaluate against.
            solution: Model-generated code (should complete the function body).
            timeout:  Maximum wall-clock seconds before the process is killed.

        Returns:
            ``ExecutionResult`` with ``passed=True`` on success.
        """
        full_solution = problem.prompt + solution

        script = _SCRIPT_TEMPLATE.format(
            allow_list=_ALLOW_LIST,
            solution=full_solution,
            test=problem.test,
            entry_point=problem.entry_point,
        )

        t0 = time.monotonic()
        error_msg: str | None = None
        passed = False

        try:
            proc = subprocess.run(  # noqa: S603
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed_ms = (time.monotonic() - t0) * 1_000
            if proc.returncode == 0:
                passed = True
            else:
                stderr = proc.stderr.strip()
                stdout = proc.stdout.strip()
                error_msg = stderr or stdout or f"returncode={proc.returncode}"
        except subprocess.TimeoutExpired:
            elapsed_ms = timeout * 1_000
            error_msg = f"TimeoutExpired after {timeout}s"
        except Exception as exc:  # pragma: no cover
            elapsed_ms = (time.monotonic() - t0) * 1_000
            error_msg = str(exc)

        return ExecutionResult(
            task_id=problem.task_id,
            passed=passed,
            error=error_msg,
            runtime_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def batch_evaluate(
        self,
        problems: list[HumanEvalProblem],
        solutions: list[str],
        timeout: float = 5.0,
    ) -> dict:
        """Evaluate a list of solutions against the corresponding problems.

        Computes pass@1 (each problem gets exactly one solution attempt).

        Args:
            problems:  List of ``HumanEvalProblem`` instances.
            solutions: List of solution strings (same length as *problems*).
            timeout:   Per-problem timeout in seconds.

        Returns:
            Dictionary with keys:
            - ``pass_at_1``  (float): Fraction of problems solved.
            - ``n_passed``   (int):   Number of problems with passing solutions.
            - ``n_total``    (int):   Total number of problems.
            - ``results``    (list):  ``ExecutionResult`` for every problem.

        Raises:
            ValueError: If *problems* and *solutions* have different lengths.
        """
        if len(problems) != len(solutions):
            raise ValueError(
                f"problems and solutions must have the same length, "
                f"got {len(problems)} vs {len(solutions)}"
            )

        results: list[ExecutionResult] = []
        for problem, solution in zip(problems, solutions):
            result = self.execute_solution(problem, solution, timeout=timeout)
            results.append(result)

        n_total = len(results)
        n_passed = sum(1 for r in results if r.passed)
        pass_at_1 = n_passed / n_total if n_total > 0 else 0.0

        return {
            "pass_at_1": pass_at_1,
            "n_passed": n_passed,
            "n_total": n_total,
            "results": results,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

try:
    from src.eval import BENCHMARK_REGISTRY  # type: ignore[attr-defined]
except Exception:
    BENCHMARK_REGISTRY: dict = {}  # type: ignore[assignment]

BENCHMARK_REGISTRY["humaneval"] = HumanEvalRunner()
