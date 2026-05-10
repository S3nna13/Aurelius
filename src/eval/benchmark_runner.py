"""Benchmark runner: execute multiple benchmarks, collect results, report."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    benchmark_names: list
    max_samples: int = 100
    timeout_seconds: float = 300.0
    seed: int = 42
    use_ilr: bool = False
    ilr_trials: int = 3
    ilr_seed: int | None = None
    score_key: str = "accuracy"


@dataclass
class BenchmarkResult:
    benchmark_name: str
    score: float
    n_samples: int
    elapsed_seconds: float
    metadata: dict = field(default_factory=dict)


class BenchmarkRunner:
    def __init__(self, benchmark_registry=None):
        if benchmark_registry is None:
            from src.eval import BENCHMARK_REGISTRY

            benchmark_registry = BENCHMARK_REGISTRY
        self._registry = benchmark_registry

    def run_benchmark(self, name: str, predictions: dict, config=None) -> BenchmarkResult:
        if config is None:
            config = RunConfig(benchmark_names=[name])
        start = time.monotonic()
        benchmark = self._registry.get(name)
        if benchmark is None:
            raise ValueError(f"Unknown benchmark: {name!r}")
        result = {}
        if benchmark is not None:
            if config.use_ilr:
                from src.eval.ilr_harness import ILRConfig, ILRHarness

                ilr_seed = config.ilr_seed if config.ilr_seed is not None else config.seed
                benchmark = ILRHarness(
                    benchmark,
                    ILRConfig(
                        n_trials=config.ilr_trials,
                        seed=ilr_seed,
                    ),
                )
            try:
                raw = benchmark.evaluate(predictions)
                if isinstance(raw, dict):
                    result = raw
            except Exception as exc:
                import traceback
                logger.error("Benchmark %s failed: %s", name, traceback.format_exc())
                result = {"_error": f"{type(exc).__name__}: {exc}", "_traceback": traceback.format_exc()}
        elapsed = time.monotonic() - start
        score = float(result.get(config.score_key, 0.0))
        n_samples = int(result.get("total", len(predictions)))
        return BenchmarkResult(
            benchmark_name=name,
            score=score,
            n_samples=n_samples,
            elapsed_seconds=elapsed,
            metadata=result,
        )

    def run_all(self, predictions: dict, config=None) -> list:
        if config is None:
            config = RunConfig(benchmark_names=list(self._registry.keys()))
        results = []
        for name in config.benchmark_names:
            if name not in self._registry:
                raise ValueError(f"Unknown benchmark: {name!r}")
            results.append(self.run_benchmark(name, predictions, config))
        return results

    def report(self, results: list) -> str:
        lines = []
        for r in results:
            lines.append(
                f"  {r.benchmark_name}: {r.score:.3f} ({r.n_samples} samples, {r.elapsed_seconds:.2f}s)"  # noqa: E501
            )
        return "\n".join(lines)

    def best(self, results: list):
        if not results:
            return None
        return max(results, key=lambda r: r.score)


def main() -> None:
    """CLI entry-point for the benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Aurelius benchmarks with optional ILR",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmark names to run (default: all in registry)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to JSON file containing predictions dict",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples per benchmark",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--ilr",
        action="store_true",
        help="Enable Instance-Level Randomization (ILR)",
    )
    parser.add_argument(
        "--ilr-trials",
        type=int,
        default=3,
        help="Number of ILR trials per instance",
    )
    parser.add_argument(
        "--ilr-seed",
        type=int,
        default=None,
        help="Seed for ILR randomization (defaults to --seed)",
    )
    args = parser.parse_args()

    predictions: dict[str, Any] = {}
    if args.predictions:
        path = Path(args.predictions)
        if path.exists():
            try:
                predictions = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Error: malformed JSON in predictions file {path}: {exc}")

    runner = BenchmarkRunner()
    config = RunConfig(
        benchmark_names=args.benchmarks or list(runner._registry.keys()),
        max_samples=args.max_samples,
        seed=args.seed,
        use_ilr=args.ilr,
        ilr_trials=args.ilr_trials,
        ilr_seed=args.ilr_seed,
    )

    results = []
    for name in config.benchmark_names:
        results.append(runner.run_benchmark(name, predictions, config))

    print(runner.report(results))


if __name__ == "__main__":
    main()
