"""MBPP (Mostly Basic Python Programming) scoring harness.

Implements the functional-correctness evaluation of Austin et al. 2021
(arXiv:2108.07732). Given an MBPP problem (task_id, text, code,
test_list, test_setup_code) and a model-generated completion, the scorer
concatenates completion + test_setup_code + test_list assertions, runs
the resulting program in a fresh isolated Python subprocess with a
wall-clock timeout and (on Unix) setrlimit memory/CPU caps, and
reports pass/fail along with which assertion (if any) failed.

Unlike HumanEval, MBPP supplies a list of bare assert statements
instead of a single check function. A problem passes iff every
assertion in test_list executes without raising.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

from src.eval.humaneval_scorer import pass_at_k

__all__ = [
    "MBPPProblem",
    "MBPPSampleResult",
    "score_single",
    "score_problems",
    "pass_at_k",
]


@dataclass(frozen=True)
class MBPPProblem:
    task_id: int
    text: str
    code: str
    test_list: list[str] = field(default_factory=list)
    test_setup_code: str = ""


@dataclass
class MBPPSampleResult:
    task_id: int
    passed: bool
    error: str | None
    duration_ms: float
    failed_test: str | None


_TEST_MARKER_PREFIX = "__MBPP_TEST_START__:"


def _build_program(problem: MBPPProblem, completion: str) -> str:
    parts: list[str] = [completion.rstrip() + "\n"]
    if problem.test_setup_code:
        parts.append(problem.test_setup_code.rstrip() + "\n")
    parts.append("import sys as _mbpp_sys\n")
    for idx, test in enumerate(problem.test_list):
        marker = f"{_TEST_MARKER_PREFIX}{idx}\n"
        parts.append(f"_mbpp_sys.stderr.write({marker!r}); _mbpp_sys.stderr.flush()\n")
        parts.append(test.rstrip() + "\n")
    return "".join(parts)


def _make_preexec(max_memory_mb: int, timeout_seconds: float):
    if os.name != "posix":
        return None
    try:
        import resource
    except ImportError:
        return None

    mem_bytes = max(int(max_memory_mb), 1) * 1024 * 1024
    cpu_seconds = max(int(math.ceil(timeout_seconds)) + 1, 2)

    def _preexec() -> None:
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
    lines = [ln for ln in stderr.splitlines() if not ln.startswith(_TEST_MARKER_PREFIX)]
    if not any(ln.strip() for ln in lines):
        return f"exit code {returncode}"
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if ":" in line and not line.startswith(("File ", "  ", "Traceback")):
            return line
        if line.endswith("Error") or line.endswith("Exception"):
            return line
    last = [ln for ln in lines if ln.strip()]
    return f"exit code {returncode}: {last[-1].strip()[:200]}"


def _extract_failed_test_index(stderr: str) -> int | None:
    matches = re.findall(re.escape(_TEST_MARKER_PREFIX) + r"(\d+)", stderr)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def score_single(
    problem: MBPPProblem,
    completion: str,
    timeout_seconds: float = 10.0,
    max_memory_mb: int = 512,
) -> MBPPSampleResult:
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
    if max_memory_mb <= 0:
        raise ValueError(f"max_memory_mb must be positive, got {max_memory_mb}")
    if not problem.test_list:
        raise ValueError(
            f"MBPP problem {problem.task_id} has empty test_list; "
            f"refusing to score (would inflate pass@k)."
        )

    program = _build_program(problem, completion)
    preexec = _make_preexec(max_memory_mb, timeout_seconds)

    child_env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME"}
    }
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"

    start = time.perf_counter()
    try:
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-I", "-c", program],
            capture_output=True,
            timeout=timeout_seconds,
            preexec_fn=preexec,
            env=child_env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = (time.perf_counter() - start) * 1000.0
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
        idx = _extract_failed_test_index(stderr)
        failed_test = (
            problem.test_list[idx]
            if idx is not None and 0 <= idx < len(problem.test_list)
            else None
        )
        return MBPPSampleResult(
            task_id=problem.task_id,
            passed=False,
            error=f"TimeoutExpired: exceeded {timeout_seconds}s",
            duration_ms=duration_ms,
            failed_test=failed_test,
        )
    except OSError as os_exc:
        duration_ms = (time.perf_counter() - start) * 1000.0
        return MBPPSampleResult(
            task_id=problem.task_id,
            passed=False,
            error=f"OSError: {os_exc!r}",
            duration_ms=duration_ms,
            failed_test=None,
        )

    duration_ms = (time.perf_counter() - start) * 1000.0
    stderr = completed.stderr.decode("utf-8", errors="replace")

    if completed.returncode == 0:
        return MBPPSampleResult(
            task_id=problem.task_id,
            passed=True,
            error=None,
            duration_ms=duration_ms,
            failed_test=None,
        )

    idx = _extract_failed_test_index(stderr)
    failed_test = (
        problem.test_list[idx] if idx is not None and 0 <= idx < len(problem.test_list) else None
    )
    error_tag = _summarize_error(stderr, completed.returncode)
    return MBPPSampleResult(
        task_id=problem.task_id,
        passed=False,
        error=error_tag,
        duration_ms=duration_ms,
        failed_test=failed_test,
    )


def score_problems(
    problems: list[MBPPProblem],
    completions: list[list[str]],
    k_values: list[int] = [1, 3],
    timeout_seconds: float = 10.0,
    max_memory_mb: int = 512,
    max_workers: int = 4,
) -> dict:
    if len(problems) != len(completions):
        raise ValueError(
            f"problems and completions must align: {len(problems)} vs {len(completions)}"
        )
    for k in k_values:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k_values must be positive ints, got {k}")
    if max_workers <= 0:
        raise ValueError(f"max_workers must be positive, got {max_workers}")

    result: dict = {
        "per_task": [],
        "n_problems": len(problems),
        "skipped_k": [],
    }
    if not problems:
        for k in k_values:
            result[f"pass@{k}"] = 0.0
        return result

    tasks: list[tuple[int, int, MBPPProblem, str]] = []
    for pi, (problem, samples) in enumerate(zip(problems, completions)):
        if not samples:
            raise ValueError(f"problem {problem.task_id} has zero completions; pass@k is undefined")
        for si, comp in enumerate(samples):
            tasks.append((pi, si, problem, comp))

    sample_results: list[list[MBPPSampleResult | None]] = [
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
        c = sum(1 for r in results if r is not None and r.passed)
        task_entry = {
            "task_id": problem.task_id,
            "n_samples": n,
            "n_correct": c,
            "samples": [asdict(r) for r in results if r is not None],
        }
        for k in k_values:
            if k <= n:
                task_entry[f"pass@{k}"] = pass_at_k(n, c, k)
        per_task.append(task_entry)
    result["per_task"] = per_task

    for k in k_values:
        if k > min_n:
            result["skipped_k"].append(k)
            continue
        vals = [
            pass_at_k(len(s), sum(1 for r in rs if r.passed), k)
            for s, rs in zip(completions, sample_results)
            if len(s) >= k
        ]
        result[f"pass@{k}"] = float(sum(vals) / len(vals)) if vals else 0.0

    return result
