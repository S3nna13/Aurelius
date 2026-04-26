"""
src/cli/benchmark_runner.py

CLI-callable benchmark runner. Takes a benchmark name (resolved against
METRIC_REGISTRY / BENCHMARK_REGISTRY in src.eval) and a model-interface
callable (generate_fn), runs the benchmark, and prints a formatted report.

Invocation:
    python -m src.cli.benchmark_runner --benchmark ifeval \
        --problems-file problems.json [--output-file report.json] [--verbose]

Pure stdlib. No foreign imports.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRun:
    benchmark: str
    n_problems: int
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    per_problem: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------


def _load_registries():
    """Import METRIC_REGISTRY and BENCHMARK_REGISTRY from src.eval.

    Kept lazy so tests can stub the module via sys.modules if needed.
    """
    from src import eval as _eval  # noqa: WPS433  (internal, not foreign)

    metric = getattr(_eval, "METRIC_REGISTRY", {}) or {}
    bench = getattr(_eval, "BENCHMARK_REGISTRY", {}) or {}
    return metric, bench


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------


def _construct_problem(problem_cls: Any, raw: Any) -> Any:
    """Best-effort construction of a problem dataclass/type from a raw dict.

    Falls back to returning `raw` unchanged if construction fails.
    """
    if problem_cls is None:
        return raw
    if isinstance(raw, problem_cls):
        return raw
    if isinstance(raw, dict):
        try:
            return problem_cls(**raw)
        except TypeError:
            pass
    return raw


def _construct_problems(problem_cls: Any, raw_problems: Sequence[Any]) -> list[Any]:
    return [_construct_problem(problem_cls, p) for p in raw_problems]


# ---------------------------------------------------------------------------
# Scoring dispatch
# ---------------------------------------------------------------------------


def _is_class(obj: Any) -> bool:
    return inspect.isclass(obj)


def _call_scorer(
    scorer: Any, problems: list[Any], generate_fn: Callable[[Any], str]
) -> dict[str, Any]:
    """Run `scorer` over problems using `generate_fn`. Returns a metrics dict.

    Handles three shapes:
      1. scorer is a class with `.score(problems, responses)` method.
      2. scorer is a class with `.score(problems, generate_fn)` method.
      3. scorer is a plain callable `scorer(problems, generate_fn)`.
    """
    # Classes: try constructing with generate_fn first, else no-arg.
    if _is_class(scorer):
        instance = None
        try:
            instance = scorer(generate_fn=generate_fn)
        except TypeError:
            try:
                instance = scorer()
            except TypeError:
                instance = None
        if instance is None or not hasattr(instance, "score"):
            # Fall back to treating the class itself as a callable scorer.
            responses = [generate_fn(p) for p in problems]
            result = scorer(problems, responses)  # type: ignore[misc]
        else:
            score_fn = instance.score
            # Detect arg shape: (problems, responses) vs (problems, generate_fn).
            try:
                sig = inspect.signature(score_fn)
                params = list(sig.parameters.values())
                needs_responses = any(p.name in ("responses", "outputs", "answers") for p in params)
            except (TypeError, ValueError):
                needs_responses = True

            if needs_responses:
                responses = [generate_fn(p) for p in problems]
                result = score_fn(problems, responses)
            else:
                try:
                    result = score_fn(problems, generate_fn)
                except TypeError:
                    responses = [generate_fn(p) for p in problems]
                    result = score_fn(problems, responses)
        return _coerce_metrics(result, n=len(problems))

    # Plain callable metric function: commonly score_problems(problems, gen_fn)
    # or score_problems(problems, responses).
    try:
        sig = inspect.signature(scorer)
        params = [p.name for p in sig.parameters.values()]
    except (TypeError, ValueError):
        params = []

    if any(n in ("generate_fn", "generate", "gen_fn") for n in params):
        result = scorer(problems, generate_fn)
    else:
        responses = [generate_fn(p) for p in problems]
        try:
            result = scorer(problems, responses)
        except TypeError:
            result = scorer(problems, generate_fn)
    return _coerce_metrics(result, n=len(problems))


def _coerce_metrics(result: Any, n: int) -> dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    # Numeric scalar → wrap under "score".
    if isinstance(result, (int, float)):
        return {"score": float(result), "n_problems": n}
    # List of per-item results → aggregate pass-rate if boolean-ish.
    if isinstance(result, list):
        passed = 0
        for r in result:
            if r is True:
                passed += 1
            elif isinstance(r, dict) and r.get("passed"):
                passed += 1
            elif hasattr(r, "strict_pass") and getattr(r, "strict_pass"):
                passed += 1
        return {
            "accuracy": (passed / len(result)) if result else 0.0,
            "n_problems": n,
            "n_passed": passed,
        }
    # Fallback: stash repr under "raw".
    return {"raw": repr(result), "n_problems": n}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_benchmark(
    benchmark_name: str,
    problems: list[Any],
    generate_fn: Callable[[Any], str],
    **kwargs: Any,
) -> BenchmarkRun:
    """Run benchmark `benchmark_name` over `problems` using `generate_fn`.

    Returns a BenchmarkRun with aggregated metrics.
    """
    if generate_fn is None:
        raise ValueError("generate_fn is required and must be callable")
    if not callable(generate_fn):
        raise ValueError("generate_fn must be callable")

    metric_reg, bench_reg = _load_registries()
    if benchmark_name not in metric_reg and benchmark_name not in bench_reg:
        known = sorted(set(list(metric_reg.keys()) + list(bench_reg.keys())))
        raise KeyError(f"Unknown benchmark {benchmark_name!r}. Known: {known}")

    problem_cls = bench_reg.get(benchmark_name)
    scorer = metric_reg.get(benchmark_name)

    constructed = _construct_problems(problem_cls, problems)

    start = time.perf_counter()
    if not constructed:
        metrics: dict[str, Any] = {"n_problems": 0}
        per_problem: list[dict[str, Any]] = []
    else:
        if scorer is None:
            # No scorer registered; just run generate_fn and report count.
            responses = [generate_fn(p) for p in constructed]
            metrics = {"n_problems": len(constructed), "n_responses": len(responses)}
            per_problem = [{"response": r} for r in responses]
        else:
            # Track per-problem outputs when verbose is requested.
            collected: list[dict[str, Any]] = []

            def _tracking_gen(p: Any) -> str:
                out = generate_fn(p)
                collected.append({"response": out})
                return out

            metrics = _call_scorer(scorer, constructed, _tracking_gen)
            per_problem = collected
    # Always ensure a tiny positive duration so downstream consumers can trust
    # duration_s > 0 even on ultra-fast paths.
    duration = max(time.perf_counter() - start, 1e-9)

    return BenchmarkRun(
        benchmark=benchmark_name,
        n_problems=len(constructed),
        metrics=metrics,
        duration_s=duration,
        per_problem=per_problem,
    )


def format_report(run: BenchmarkRun, verbose: bool = False) -> str:
    """Render a human-readable report for a BenchmarkRun."""
    lines: list[str] = []
    lines.append(f"Benchmark: {run.benchmark}")
    lines.append(f"Problems:  {run.n_problems}")
    lines.append(f"Duration:  {run.duration_s:.4f}s")
    lines.append("Metrics:")
    if not run.metrics:
        lines.append("  (no metrics)")
    else:
        for k, v in run.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            elif isinstance(v, dict):
                lines.append(f"  {k}:")
                for kk, vv in v.items():
                    if isinstance(vv, float):
                        lines.append(f"    {kk}: {vv:.4f}")
                    else:
                        lines.append(f"    {kk}: {vv}")
            else:
                lines.append(f"  {k}: {v}")
    if verbose and run.per_problem:
        lines.append("Per-problem:")
        for i, item in enumerate(run.per_problem):
            preview = item.get("response", "")
            if isinstance(preview, str) and len(preview) > 120:
                preview = preview[:117] + "..."
            lines.append(f"  [{i}] {preview!r}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default generate_fn for CLI (oracle echo)
# ---------------------------------------------------------------------------


def _default_generate_fn(problem: Any) -> str:
    """Extremely conservative default: return problem prompt if available.

    Used only when the CLI is invoked without a programmatic generate_fn.
    Intended for smoke-testing the runner itself.
    """
    if hasattr(problem, "prompt"):
        return str(getattr(problem, "prompt"))
    if isinstance(problem, dict) and "prompt" in problem:
        return str(problem["prompt"])
    return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.cli.benchmark_runner",
        description="Run an Aurelius evaluation benchmark and print a report.",
    )
    p.add_argument(
        "--benchmark",
        "-b",
        required=True,
        help="Benchmark name registered in src.eval.BENCHMARK_REGISTRY.",
    )
    p.add_argument(
        "--problems-file",
        "-p",
        default=None,
        help="Path to a JSON file containing a list of problem dicts.",
    )
    p.add_argument(
        "--output-file",
        "-o",
        default=None,
        help="If set, write the BenchmarkRun (JSON) to this path.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Include per-problem results in the report.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Load problems.
    if args.problems_file:
        try:
            with open(args.problems_file, encoding="utf-8") as fh:
                problems = json.load(fh)
        except FileNotFoundError:
            print(f"error: problems file not found: {args.problems_file}", file=sys.stderr)
            return 2
        except json.JSONDecodeError as e:
            print(f"error: invalid JSON in {args.problems_file}: {e}", file=sys.stderr)
            return 2
        if not isinstance(problems, list):
            print("error: problems file must contain a JSON list", file=sys.stderr)
            return 2
    else:
        problems = []

    try:
        run = run_benchmark(
            benchmark_name=args.benchmark,
            problems=problems,
            generate_fn=_default_generate_fn,
        )
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 4
    except Exception as e:  # noqa: BLE001
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    report = format_report(run, verbose=args.verbose)
    print(report)

    if args.output_file:
        payload = asdict(run)
        try:
            with open(args.output_file, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
        except OSError as e:
            print(f"error: could not write output: {e}", file=sys.stderr)
            return 5

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BenchmarkRun",
    "run_benchmark",
    "format_report",
    "main",
]
