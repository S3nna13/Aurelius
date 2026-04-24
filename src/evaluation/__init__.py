"""Aurelius evaluation surface."""
from .eval_harness import EvalHarness, EvalTask, EvalResult, EVAL_HARNESS_REGISTRY
from .benchmark_runner import BenchmarkRunner, BenchmarkSuite, BenchmarkResult, BENCHMARK_RUNNER_REGISTRY
from .metric_aggregator import MetricAggregator, MetricWeight, MetricScore, METRIC_AGGREGATOR_REGISTRY

EVALUATION_REGISTRY = {
    "eval_harness": EVAL_HARNESS_REGISTRY,
    "benchmark_runner": BENCHMARK_RUNNER_REGISTRY,
    "metric_aggregator": METRIC_AGGREGATOR_REGISTRY,
}
