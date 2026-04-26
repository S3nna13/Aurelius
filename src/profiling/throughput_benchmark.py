from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class ThroughputBenchmark:
    def measure(self, fn: Callable, iterations: int = 100, args: tuple[Any, ...] = ()) -> float:
        start = time.perf_counter()
        for _ in range(iterations):
            fn(*args)
        elapsed = time.perf_counter() - start
        return iterations / elapsed if elapsed > 0 else float("inf")

    def measure_with_warmup(
        self, fn: Callable, iterations: int = 100, warmup: int = 10, args: tuple[Any, ...] = ()
    ) -> float:
        for _ in range(warmup):
            fn(*args)
        return self.measure(fn, iterations, args)


THROUGHPUT_BENCHMARK = ThroughputBenchmark()
