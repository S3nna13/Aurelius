"""Benchmark runner: execute multiple benchmarks, collect results, report."""

import time
from dataclasses import dataclass, field


@dataclass
class RunConfig:
    benchmark_names: list
    max_samples: int = 100
    timeout_seconds: float = 300.0
    seed: int = 42


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
        result = {}
        if benchmark is not None:
            try:
                raw = benchmark.evaluate(predictions)
                if isinstance(raw, dict):
                    result = raw
            except Exception:
                result = {}
        elapsed = time.monotonic() - start
        score = float(result.get("accuracy", 0.0))
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
        for name in self._registry:
            results.append(self.run_benchmark(name, predictions, config))
        return results

    def report(self, results: list) -> str:
        lines = []
        for r in results:
            lines.append(
                f"  {r.benchmark_name}: {r.score:.3f} ({r.n_samples} samples, {r.elapsed_seconds:.2f}s)"
            )
        return "\n".join(lines)

    def best(self, results: list):
        if not results:
            return None
        return max(results, key=lambda r: r.score)
