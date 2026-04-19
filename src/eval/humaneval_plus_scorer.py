"""HumanEval+ scoring harness.

Implements the augmented functional-correctness evaluation of Liu et al. 2023
(EvalPlus, arXiv:2305.01210). HumanEval+ retains each original HumanEval
problem's canonical ``base_test`` but adds an "EvalPlus" set of extra test
cases (~81x more, generated via automatic mutation/fuzz testing) so that a
subtly-wrong completion that happens to satisfy the thin original test is
exposed by the richer plus suite.

Scoring mechanics are identical to plain HumanEval -- a subprocess runs the
prompt + completion + a test program, and a nonzero exit code means failure.
The public novelty here is:

1. Separate tracking of ``passed_base`` vs ``passed_plus``.
2. Aggregated ``base_pass@k``, ``plus_pass@k``, and ``robustness_gap`` =
   ``base_pass@1 - plus_pass@1``. The gap is a proxy for test-suite
   adequacy -- a large gap means the thin base suite was over-estimating
   correctness.

Design choice: base precedence
------------------------------
If ``base_test`` fails we do *not* execute ``plus_tests``. A completion that
fails the canonical HumanEval contract cannot meaningfully pass a strictly
larger suite, and running the plus tests would waste budget. The plus
counters reflect "did not run" by reporting ``passed_plus=False`` and
``plus_fail_count=len(plus_tests)``. Callers that want independent plus
timing should call the underlying helpers directly.

Pure stdlib. Reuses ``pass_at_k`` from ``src.eval.humaneval_scorer``.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

from .humaneval_scorer import pass_at_k


__all__ = [
    "HumanEvalPlusProblem",
    "HumanEvalPlusResult",
    "score_single",
    "score_problems",
    "pass_at_k",
]


@dataclass(frozen=True)
class HumanEvalPlusProblem:
    """A single HumanEval+ problem."""

    task_id: str
    prompt: str
    canonical_solution: str
    base_test: str
    plus_tests: list[str]
    entry_point: str


@dataclass
class HumanEvalPlusResult:
    """Outcome of running a completion against a HumanEval+ problem."""

    task_id: str
    passed_base: bool
    passed_plus: bool
    base_fail_count: int
    plus_fail_count: int
    duration_ms: float
    error: Optional[str]


def _build_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    return (
        prompt
        + completion
        + "\n"
        + test
        + "\ncheck("
        + entry_point
        + ")\n"
    )


def _make_preexec(max_memory_mb: int, timeout_seconds: float):
    if os.name != "posix":
        return None
    try:
        import resource  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return None

    mem_bytes = max(int(max_memory_mb), 1) * 1024 * 1024
    cpu_seconds = max(int(math.ceil(timeout_seconds)) + 1, 2)

    def _preexec() -> None:  # pragma: no cover
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, OSError):
            pass
        for limit_name in ("RLIMIT_AS", "RLIMIT_DATA"):
            limit = getattr(resource, limit_name, None)
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (mem_bytes, mem_bytes))
                break
            except (ValueError, OSError):
                continue
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass

    return _preexec


def _summarize_error(stderr: str, returncode: int) -> str:
    if not stderr.strip():
        return f"exit code {returncode}"
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if not line:
            continue
        if ":" in line and not line.startswith(("File ", "  ", "Traceback")):
            return line
        if line.endswith("Error") or line.endswith("Exception"):
            return line
    return f"exit code {returncode}: {stderr.strip().splitlines()[-1][:200]}"


def _run_one_test(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    timeout_seconds: float,
    max_memory_mb: int,
) -> tuple[bool, Optional[str]]:
    program = _build_program(prompt, completion, test, entry_point)
    preexec = _make_preexec(max_memory_mb, timeout_seconds)

    child_env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME"}
    }
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", program],
            capture_output=True,
            timeout=timeout_seconds,
            preexec_fn=preexec,
            env=child_env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"TimeoutExpired: exceeded {timeout_seconds}s"
    except OSError as os_exc:
        return False, f"OSError: {os_exc!r}"

    if completed.returncode == 0:
        return True, None
    stderr = completed.stderr.decode("utf-8", errors="replace")
    return False, _summarize_error(stderr, completed.returncode)


def score_single(
    problem: HumanEvalPlusProblem,
    completion: str,
    timeout_seconds: float = 10.0,
    max_memory_mb: int = 512,
) -> HumanEvalPlusResult:
    """Run the base test, then each plus test, in isolated subprocesses.

    If the base test fails, plus tests are skipped (see module docstring).
    """
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
    if max_memory_mb <= 0:
        raise ValueError(f"max_memory_mb must be positive, got {max_memory_mb}")

    start = time.perf_counter()

    base_passed, base_err = _run_one_test(
        problem.prompt,
        completion,
        problem.base_test,
        problem.entry_point,
        timeout_seconds,
        max_memory_mb,
    )
    base_fail_count = 0 if base_passed else 1

    if not base_passed:
        duration_ms = (time.perf_counter() - start) * 1000.0
        return HumanEvalPlusResult(
            task_id=problem.task_id,
            passed_base=False,
            passed_plus=False,
            base_fail_count=base_fail_count,
            plus_fail_count=len(problem.plus_tests),
            duration_ms=duration_ms,
            error=base_err,
        )

    plus_fail_count = 0
    first_plus_err: Optional[str] = None
    for ptest in problem.plus_tests:
        p_passed, p_err = _run_one_test(
            problem.prompt,
            completion,
            ptest,
            problem.entry_point,
            timeout_seconds,
            max_memory_mb,
        )
        if not p_passed:
            plus_fail_count += 1
            if first_plus_err is None:
                first_plus_err = p_err

    duration_ms = (time.perf_counter() - start) * 1000.0
    passed_plus = plus_fail_count == 0
    error = None if passed_plus else first_plus_err
    return HumanEvalPlusResult(
        task_id=problem.task_id,
        passed_base=True,
        passed_plus=passed_plus,
        base_fail_count=0,
        plus_fail_count=plus_fail_count,
        duration_ms=duration_ms,
        error=error,
    )


def score_problems(
    problems: list[HumanEvalPlusProblem],
    completions: list[list[str]],
    k_values: list[int] = [1, 10],
    timeout_seconds: float = 10.0,
    max_memory_mb: int = 512,
    max_workers: int = 4,
) -> dict:
    """Aggregate HumanEval+ scores across many problems and samples."""
    if len(problems) != len(completions):
        raise ValueError(
            f"problems and completions must align: "
            f"{len(problems)} vs {len(completions)}"
        )
    for k in k_values:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k_values must be positive ints, got {k}")
    if max_workers <= 0:
        raise ValueError(f"max_workers must be positive, got {max_workers}")

    out: dict = {
        "per_task": [],
        "n_problems": len(problems),
        "skipped_k": [],
        "base_pass@1": 0.0,
        "plus_pass@1": 0.0,
        "robustness_gap": 0.0,
    }
    if not problems:
        for k in k_values:
            out[f"base_pass@{k}"] = 0.0
            out[f"plus_pass@{k}"] = 0.0
        return out

    tasks = []
    for pi, (problem, samples) in enumerate(zip(problems, completions)):
        if not samples:
            raise ValueError(
                f"problem {problem.task_id} has zero completions; "
                f"pass@k is undefined"
            )
        for si, comp in enumerate(samples):
            tasks.append((pi, si, problem, comp))

    sample_results: list[list[Optional[HumanEvalPlusResult]]] = [
        [None] * len(samples) for samples in completions
    ]

    def _run(task):
        pi, si, problem, comp = task
        return (
            pi,
            si,
            score_single(
                problem,
                comp,
                timeout_seconds=timeout_seconds,
                max_memory_mb=max_memory_mb,
            ),
        )

    if max_workers == 1:
        for task in tasks:
            pi, si, res = _run(task)
            sample_results[pi][si] = res
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for pi, si, res in pool.map(_run, tasks):
                sample_results[pi][si] = res

    min_n = min(len(s) for s in completions)
    per_task = []
    for problem, samples, results in zip(problems, completions, sample_results):
        n = len(samples)
        c_base = sum(1 for r in results if r is not None and r.passed_base)
        c_plus = sum(1 for r in results if r is not None and r.passed_plus)
        entry = {
            "task_id": problem.task_id,
            "n_samples": n,
            "n_base_correct": c_base,
            "n_plus_correct": c_plus,
        }
        for k in k_values:
            if k <= n:
                entry[f"base_pass@{k}"] = pass_at_k(n, c_base, k)
                entry[f"plus_pass@{k}"] = pass_at_k(n, c_plus, k)
        per_task.append(entry)
    out["per_task"] = per_task

    effective_ks = list(k_values) if 1 in k_values else list(k_values) + [1]

    for k in effective_ks:
        if k > min_n:
            if k in k_values and k not in out["skipped_k"]:
                out["skipped_k"].append(k)
            continue
        base_vals = []
        plus_vals = []
        for samples, rs in zip(completions, sample_results):
            n = len(samples)
            if n < k:
                continue
            cb = sum(1 for r in rs if r.passed_base)
            cp = sum(1 for r in rs if r.passed_plus)
            base_vals.append(pass_at_k(n, cb, k))
            plus_vals.append(pass_at_k(n, cp, k))
        if base_vals:
            out[f"base_pass@{k}"] = float(sum(base_vals) / len(base_vals))
            out[f"plus_pass@{k}"] = float(sum(plus_vals) / len(plus_vals))
        else:
            out[f"base_pass@{k}"] = 0.0
            out[f"plus_pass@{k}"] = 0.0

    out["robustness_gap"] = out["base_pass@1"] - out["plus_pass@1"]
    return out
