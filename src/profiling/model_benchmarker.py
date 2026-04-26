from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    warmup_runs: int = 3
    bench_runs: int = 10
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8])


@dataclass(frozen=True)
class BenchmarkStats:
    batch_size: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_samples_per_s: float


class ModelBenchmarker:
    def __init__(self, config: BenchmarkConfig | None = None):
        self.config = config if config is not None else BenchmarkConfig()

    def run(
        self,
        fn: Callable,
        batch_size: int,
        input_factory: Callable,
    ) -> BenchmarkStats:
        inp = input_factory(batch_size)
        for _ in range(self.config.warmup_runs):
            fn(inp)

        samples_ms: list[float] = []
        for _ in range(self.config.bench_runs):
            start = time.perf_counter()
            fn(inp)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            samples_ms.append(elapsed_ms)

        samples_sorted = sorted(samples_ms)
        n = len(samples_sorted)
        mean_ms = sum(samples_sorted) / n
        p50_idx = max(math.ceil(n * 0.50) - 1, 0)
        p95_idx = max(math.ceil(n * 0.95) - 1, 0)
        p99_idx = max(math.ceil(n * 0.99) - 1, 0)
        p50 = samples_sorted[p50_idx]
        p95 = samples_sorted[p95_idx]
        p99 = samples_sorted[p99_idx]
        mean_s = mean_ms / 1000.0
        throughput = batch_size / mean_s if mean_s > 0 else 0.0

        return BenchmarkStats(
            batch_size=batch_size,
            latency_mean_ms=mean_ms,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            throughput_samples_per_s=throughput,
        )

    def run_sweep(
        self,
        fn: Callable,
        input_factory: Callable,
    ) -> list[BenchmarkStats]:
        return [self.run(fn, bs, input_factory) for bs in self.config.batch_sizes]

    def report(self, stats_list: list[BenchmarkStats]) -> str:
        if not stats_list:
            return "No benchmark stats."
        header = f"{'batch_size':>10} | {'p50_ms':>10} | {'p99_ms':>10} | {'throughput':>12}"
        sep = "-" * len(header)
        lines = [header, sep]
        for s in stats_list:
            lines.append(
                f"{s.batch_size:>10} | {s.latency_p50_ms:>10.3f} | "
                f"{s.latency_p99_ms:>10.3f} | {s.throughput_samples_per_s:>12.2f}"
            )
        return "\n".join(lines)

    def best_throughput(self, stats_list: list[BenchmarkStats]) -> BenchmarkStats:
        if not stats_list:
            raise ValueError("stats_list is empty")
        return max(stats_list, key=lambda s: s.throughput_samples_per_s)


MODEL_BENCHMARKER_REGISTRY: dict[str, type[ModelBenchmarker]] = {
    "default": ModelBenchmarker,
}
