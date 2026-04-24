"""Aurelius evaluation surface."""
from .eval_harness import EvalHarness, EvalTask, EvalResult, EVAL_HARNESS_REGISTRY
from .benchmark_runner import BenchmarkRunner, BenchmarkSuite, BenchmarkResult, BENCHMARK_RUNNER_REGISTRY
from .metric_aggregator import MetricAggregator, MetricWeight, MetricScore, METRIC_AGGREGATOR_REGISTRY
from .perplexity_scorer import PerplexityScorer, PerplexityResult, PERPLEXITY_SCORER_REGISTRY
from .rouge_scorer import RougeScorer, RougeScores, ROUGE_SCORER_REGISTRY
from .code_eval import CodeEvaluator, CodeEvalConfig, CodeEvalResult, CODE_EVALUATOR_REGISTRY

EVALUATION_REGISTRY = {
    "eval_harness": EVAL_HARNESS_REGISTRY,
    "benchmark_runner": BENCHMARK_RUNNER_REGISTRY,
    "metric_aggregator": METRIC_AGGREGATOR_REGISTRY,
    "perplexity_scorer": PERPLEXITY_SCORER_REGISTRY,
    "rouge_scorer": ROUGE_SCORER_REGISTRY,
    "code_eval": CODE_EVALUATOR_REGISTRY,
}
