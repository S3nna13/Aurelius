"""HumanEval scoring harness.

Implements the functional-correctness evaluation of Chen et al. 2021
(arXiv:2107.03374). Given a HumanEval-shaped problem (prompt, canonical
solution, test, entry_point) and a model-generated completion, the scorer
concatenates the prompt + completion + test + ``check(entry_point)`` call,
runs the resulting program in a fresh Python subprocess with a wall-clock
timeout and (on Unix) ``setrlimit`` memory / CPU caps, and reports pass/fail.

Aggregation across many problems uses the numerically stable pass@k estimator

    pass@k = 1 - prod_{i=0..k-1} (1 - c / (n - i))

where n is the number of samples per problem and c is the number of samples
that pass the unit tests.

Security notes
--------------
The subprocess runs with the *caller's* interpreter and filesystem; we do
not jail network access, chroot, or containerize. This is appropriate for
evaluating our own model's code in a trusted developer environment and for
unit testing. Production callers that score untrusted code SHOULD wrap
``score_single`` / ``score_problems`` in a further sandbox (docker,
firejail, gVisor, nsjail, etc.). We never evaluate the completion in the
scorer's own process, so an infinite loop, ``sys.exit``, or SIGSEGV in the
generated code cannot destabilize the scorer.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

__all__ = [
    "HumanEvalProblem",
    "SampleResult",
    "score_single",
    "score_problems",
    "pass_at_k",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HumanEvalProblem:
    """A single HumanEval problem.

    Attributes mirror the public ``openai_humaneval`` dataset fields.
    """

    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str


@dataclass
class SampleResult:
    """Outcome of running a single completion against its unit tests."""

    task_id: str
    passed: bool
    error: str | None
    duration_ms: float
    stdout: str
    stderr: str


# ---------------------------------------------------------------------------
# pass@k
# ---------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Numerically stable estimator of pass@k (Chen et al. 2021).

    Parameters
    ----------
    n: total samples drawn for the problem
    c: number of samples that passed
    k: k in pass@k

    Raises
    ------
    ValueError if arguments are out of domain.
    """

    if not isinstance(n, int) or not isinstance(c, int) or not isinstance(k, int):
        raise ValueError("n, c, k must be integers")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if k > n:
        raise ValueError(f"k ({k}) must be <= n ({n})")
    if c < 0 or c > n:
        raise ValueError(f"c must be in [0, n], got c={c}, n={n}")

    if n - c < k:
        # Enough correct samples that every k-subset must contain one.
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= 1.0 - c / (n - i)
    return 1.0 - prod


# ---------------------------------------------------------------------------
# Single-sample scoring
# ---------------------------------------------------------------------------


def _build_program(problem: HumanEvalProblem, completion: str) -> str:
    """Assemble the program run in the subprocess."""
    return (
        problem.prompt + completion + "\n" + problem.test + "\ncheck(" + problem.entry_point + ")\n"
    )


def _make_preexec(max_memory_mb: int, timeout_seconds: float):
    """Return a preexec_fn that applies resource limits on Unix.

    Returns None on platforms where ``resource`` is not usable (Windows).
    """
    if os.name != "posix":
        return None
    try:
        import resource  # noqa: PLC0415 - Unix-only
    except ImportError:  # pragma: no cover - truly exotic platform
        return None

    mem_bytes = max(int(max_memory_mb), 1) * 1024 * 1024
    # CPU cap: the wall-clock timeout already bounds the subprocess, but a
    # CPU rlimit kills a busy loop even if the parent timer races.
    cpu_seconds = max(int(math.ceil(timeout_seconds)) + 1, 2)

    def _preexec() -> None:  # pragma: no cover - runs in child
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


def score_single(
    problem: HumanEvalProblem,
    completion: str,
    timeout_seconds: float = 10.0,
    max_memory_mb: int = 512,
) -> SampleResult:
    """Run one completion against its unit tests in a subprocess.

    The subprocess is spawned with ``sys.executable -I -c <program>`` so it
    does not inherit any of the Aurelius package imports, current working
    directory state, or sys.path tweaks. ``-I`` isolates the child from
    ``PYTHON*`` env vars and user site-packages.
    """
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
    if max_memory_mb <= 0:
        raise ValueError(f"max_memory_mb must be positive, got {max_memory_mb}")

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
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
        return SampleResult(
            task_id=problem.task_id,
            passed=False,
            error=f"TimeoutExpired: exceeded {timeout_seconds}s",
            duration_ms=duration_ms,
            stdout=stdout,
            stderr=stderr,
        )
    except OSError as os_exc:
        duration_ms = (time.perf_counter() - start) * 1000.0
        return SampleResult(
            task_id=problem.task_id,
            passed=False,
            error=f"OSError: {os_exc!r}",
            duration_ms=duration_ms,
            stdout="",
            stderr="",
        )

    duration_ms = (time.perf_counter() - start) * 1000.0
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")

    if completed.returncode == 0:
        return SampleResult(
            task_id=problem.task_id,
            passed=True,
            error=None,
            duration_ms=duration_ms,
            stdout=stdout,
            stderr=stderr,
        )

    error_tag = _summarize_error(stderr, completed.returncode)
    return SampleResult(
        task_id=problem.task_id,
        passed=False,
        error=error_tag,
        duration_ms=duration_ms,
        stdout=stdout,
        stderr=stderr,
    )


def _summarize_error(stderr: str, returncode: int) -> str:
    """Pick a short error description from a subprocess traceback."""
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


# ---------------------------------------------------------------------------
# Aggregated scoring
# ---------------------------------------------------------------------------


def score_problems(
    problems: list[HumanEvalProblem],
    completions: list[list[str]],
    k_values: list[int] = [1, 10],
    timeout_seconds: float = 10.0,
    max_memory_mb: int = 512,
    max_workers: int = 4,
) -> dict:
    """Score many problems, each with multiple sampled completions.

    Parameters
    ----------
    problems: list of ``HumanEvalProblem``
    completions: parallel list; ``completions[i]`` is the list of sampled
        completions for ``problems[i]``.
    k_values: which ks to compute. Values greater than min(n_i) across
        problems are dropped from the aggregate report and recorded in
        ``skipped_k``.
    max_workers: thread-pool width. The subprocess IS the isolation
        boundary, so a ThreadPoolExecutor is appropriate here.

    Returns a dict::

        {
            "pass@1": float,
            "pass@10": float,
            "per_task": [ {...}, ...],
            "n_problems": int,
            "skipped_k": [int, ...],
        }
    """
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

    tasks: list[tuple[int, int, HumanEvalProblem, str]] = []
    for pi, (problem, samples) in enumerate(zip(problems, completions)):
        if not samples:
            raise ValueError(f"problem {problem.task_id} has zero completions; pass@k is undefined")
        for si, comp in enumerate(samples):
            tasks.append((pi, si, problem, comp))

    sample_results: list[list[SampleResult | None]] = [
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
