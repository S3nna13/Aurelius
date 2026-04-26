"""Tests for throughput_benchmark — throughput measurement."""
from __future__ import annotations

import time

from src.profiling.throughput_benchmark import ThroughputBenchmark


class TestThroughputBenchmark:
    def test_measure_returns_positive_throughput(self):
        def dummy():
            for _ in range(1000):
                pass
        t = ThroughputBenchmark().measure(dummy, iterations=1000)
        assert t > 0

    def test_faster_function_higher_throughput(self):
        bm = ThroughputBenchmark()

        def slow():
            time.sleep(0.001)

        def fast():
            pass

        t_slow = bm.measure(slow, iterations=5)
        t_fast = bm.measure(fast, iterations=100)
        assert t_fast > t_slow

    def test_measure_with_args(self):
        def adder(x, y):
            return x + y
        t = ThroughputBenchmark().measure(adder, iterations=100, args=(1, 2))
        assert t > 0

    def test_batch_size_scaling(self):
        bm = ThroughputBenchmark()
        r1 = bm.measure(lambda: None, iterations=1)
        r10 = bm.measure(lambda: None, iterations=100)
        assert r1 > 0 and r10 > 0
