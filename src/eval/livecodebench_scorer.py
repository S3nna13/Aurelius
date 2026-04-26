"""LiveCodeBench scoring harness.

Implements a pure-stdlib scorer for the LiveCodeBench benchmark
(Jain et al. 2024, arXiv:2403.07974). LiveCodeBench is a
contamination-free coding benchmark: problems are tagged with a
release date so evaluations can focus on problems released after the
model's training cutoff, and with a difficulty label
(easy/medium/hard) plus a contest source (leetcode, codeforces,
atcoder, ...).

Scoring follows the HumanEval/MBPP recipe: each problem ships a set
of (stdin, expected_stdout) test cases; we concatenate the problem's
starter code with the model completion, run the resulting script in
an isolated Python subprocess, feed it each stdin, and compare the
trimmed stdout to the expected output. A problem passes iff every
test case matches.

Parallel shape to ``humaneval_scorer`` and ``mbpp_scorer``:
``score_single`` returns a per-problem result; ``score_problems``
aggregates and additionally reports per-difficulty / per-source
accuracy and accepts a ``date_filter`` string of the form
``"after YYYY-MM-DD"`` or ``"before YYYY-MM-DD"``.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

from src.eval.humaneval_scorer import pass_at_k

__all__ = [
    "LiveCodeProblem",
    "LiveCodeResult",
    "score_single",
    "score_problems",
    "pass_at_k",
    "parse_date_filter",
]


@dataclass(frozen=True)
class LiveCodeProblem:
    """A single LiveCodeBench problem."""

    task_id: str
    prompt: str
    starter_code: str
    test_cases: list[tuple[str, str]] = field(default_factory=list)
    difficulty: str = "medium"
    release_date: str | None = None
    contest_source: str = "unknown"


@dataclass
class LiveCodeResult:
    """Outcome of running a completion against its test cases."""

    task_id: str
    passed: bool
    duration_ms: float
    error: str | None
    failed_case_idx: int | None


def _parse_iso_date(s: str) -> tuple[int, int, int]:
    parts = s.split("-")
    if len(parts) != 3:
        raise ValueError(f"expected YYYY-MM-DD, got {s!r}")
    y, m, d = parts
    if len(y) != 4 or len(m) != 2 or len(d) != 2:
        raise ValueError(f"expected YYYY-MM-DD, got {s!r}")
    return int(y), int(m), int(d)


def parse_date_filter(spec: str) -> tuple[str, tuple[int, int, int]]:
    """Parse ``"after YYYY-MM-DD"`` or ``"before YYYY-MM-DD"``."""
    if not isinstance(spec, str):
        raise ValueError(f"date_filter must be str, got {type(spec).__name__}")
    parts = spec.strip().split()
    if len(parts) != 2:
        raise ValueError(
            f"date_filter must be 'after YYYY-MM-DD' or 'before YYYY-MM-DD', got {spec!r}"
        )
    mode, date_str = parts[0].lower(), parts[1]
    if mode not in ("after", "before"):
        raise ValueError(f"date_filter mode must be 'after' or 'before', got {mode!r}")
    return mode, _parse_iso_date(date_str)


def _passes_date_filter(
    release_date: str | None,
    filt: tuple[str, tuple[int, int, int]] | None,
) -> bool:
    if filt is None:
        return True
    if release_date is None:
        return False
    try:
        rd = _parse_iso_date(release_date)
    except ValueError:
        return False
    mode, cutoff = filt
    if mode == "after":
        return rd > cutoff
    return rd < cutoff


def _make_preexec(max_memory_mb: int, timeout_seconds: float):
    if os.name != "posix":
        return None
    try:
        import resource  # noqa: PLC0415
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


def _build_program(problem: LiveCodeProblem, completion: str) -> str:
    parts: list[str] = []
    if problem.starter_code:
        parts.append(problem.starter_code.rstrip() + "\n")
    parts.append(completion)
    if not completion.endswith("\n"):
        parts.append("\n")
    return "".join(parts)


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


def _normalize(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def score_single(
    problem: LiveCodeProblem,
    completion: str,
    timeout_seconds: float = 10.0,
    max_memory_mb: int = 512,
) -> LiveCodeResult:
    """Run one completion against every test case in ``problem``."""
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
    if max_memory_mb <= 0:
        raise ValueError(f"max_memory_mb must be positive, got {max_memory_mb}")
    if not problem.test_cases:
        raise ValueError(
            f"LiveCodeBench problem {problem.task_id} has empty test_cases; "
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
    for idx, (stdin_str, expected) in enumerate(problem.test_cases):
        try:
            completed = subprocess.run(  # noqa: S603
                [sys.executable, "-I", "-c", program],
                input=stdin_str.encode("utf-8"),
                capture_output=True,
                timeout=timeout_seconds,
                preexec_fn=preexec,
                env=child_env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start) * 1000.0
            return LiveCodeResult(
                task_id=problem.task_id,
                passed=False,
                duration_ms=duration_ms,
                error=f"TimeoutExpired: exceeded {timeout_seconds}s on case {idx}",
                failed_case_idx=idx,
            )
        except OSError as os_exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            return LiveCodeResult(
                task_id=problem.task_id,
                passed=False,
                duration_ms=duration_ms,
                error=f"OSError: {os_exc!r}",
                failed_case_idx=idx,
            )

        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")

        if completed.returncode != 0:
            duration_ms = (time.perf_counter() - start) * 1000.0
            return LiveCodeResult(
                task_id=problem.task_id,
                passed=False,
                duration_ms=duration_ms,
                error=_summarize_error(stderr, completed.returncode),
                failed_case_idx=idx,
            )

        if _normalize(stdout) != _normalize(expected):
            duration_ms = (time.perf_counter() - start) * 1000.0
            return LiveCodeResult(
                task_id=problem.task_id,
                passed=False,
                duration_ms=duration_ms,
                error=(
                    f"output mismatch on case {idx}: "
                    f"expected {_normalize(expected)!r}, got {_normalize(stdout)!r}"
                ),
                failed_case_idx=idx,
            )

    duration_ms = (time.perf_counter() - start) * 1000.0
    return LiveCodeResult(
        task_id=problem.task_id,
        passed=True,
        duration_ms=duration_ms,
        error=None,
        failed_case_idx=None,
    )


def score_problems(
    problems: list[LiveCodeProblem],
    completions: list[list[str]],
    k_values: list[int] = [1],
    timeout_seconds: float = 10.0,
    max_memory_mb: int = 512,
    date_filter: str | None = None,
) -> dict:
    """Score many LiveCodeBench problems with per-category breakdown."""
    if len(problems) != len(completions):
        raise ValueError(
            f"problems and completions must align: {len(problems)} vs {len(completions)}"
        )
    for k in k_values:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k_values must be positive ints, got {k}")

    filt = parse_date_filter(date_filter) if date_filter is not None else None

    kept: list[tuple[LiveCodeProblem, list[str]]] = []
    filtered_out = 0
    for problem, samples in zip(problems, completions):
        if not _passes_date_filter(problem.release_date, filt):
            filtered_out += 1
            continue
        if not samples:
            raise ValueError(f"problem {problem.task_id} has zero completions; pass@k is undefined")
        kept.append((problem, samples))

    result: dict = {
        "per_task": [],
        "per_difficulty": {},
        "per_source": {},
        "n_problems": len(kept),
        "filtered_out": filtered_out,
    }
    if not kept:
        for k in k_values:
            result[f"pass@{k}"] = 0.0
        return result

    per_task: list[dict] = []
    for problem, samples in kept:
        sample_results: list[LiveCodeResult] = []
        for comp in samples:
            sample_results.append(
                score_single(
                    problem,
                    comp,
                    timeout_seconds=timeout_seconds,
                    max_memory_mb=max_memory_mb,
                )
            )
        n = len(sample_results)
        c = sum(1 for r in sample_results if r.passed)
        first = sample_results[0]
        per_task.append(
            {
                "task_id": problem.task_id,
                "difficulty": problem.difficulty,
                "contest_source": problem.contest_source,
                "release_date": problem.release_date,
                "n_samples": n,
                "n_correct": c,
                "passed": first.passed,
                "failed_case_idx": first.failed_case_idx,
                "error": first.error,
                "duration_ms": first.duration_ms,
            }
        )

    result["per_task"] = per_task

    for k in k_values:
        vals: list[float] = []
        for task in per_task:
            vals.append(pass_at_k(task["n_samples"], task["n_correct"], k))
        result[f"pass@{k}"] = float(sum(vals) / len(vals)) if vals else 0.0

    by_diff: dict[str, list[int]] = {}
    by_src: dict[str, list[int]] = {}
    for task in per_task:
        by_diff.setdefault(task["difficulty"], []).append(1 if task["passed"] else 0)
        by_src.setdefault(task["contest_source"], []).append(1 if task["passed"] else 0)
    result["per_difficulty"] = {d: (sum(v) / len(v)) if v else 0.0 for d, v in by_diff.items()}
    result["per_source"] = {s: (sum(v) / len(v)) if v else 0.0 for s, v in by_src.items()}

    return result
