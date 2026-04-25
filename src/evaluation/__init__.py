"""Aurelius evaluation surface."""
from .eval_harness import EvalHarness, EvalTask, EvalResult, EVAL_HARNESS_REGISTRY
from .benchmark_runner import BenchmarkRunner, BenchmarkSuite, BenchmarkResult, BENCHMARK_RUNNER_REGISTRY
from .metric_aggregator import MetricAggregator, MetricWeight, MetricScore, METRIC_AGGREGATOR_REGISTRY
from .perplexity_scorer import PerplexityScorer, PerplexityResult, PERPLEXITY_SCORER_REGISTRY
from .rouge_scorer import RougeScorer, RougeScores, ROUGE_SCORER_REGISTRY
from .code_eval import CodeEvaluator, CodeEvalConfig, CodeEvalResult, CODE_EVALUATOR_REGISTRY
from .bleu_scorer import BLEUScorer, BLEUResult, BLEU_SCORER_REGISTRY
from .faithfulness_scorer import FaithfulnessScorer, FaithfulnessResult, FAITHFULNESS_SCORER_REGISTRY
from .f1_scorer import F1Scorer, F1Result, F1_SCORER_REGISTRY
from .leaderboard_tracker import Leaderboard, LeaderboardEntry, LEADERBOARD_TRACKER_REGISTRY
from .cross_validation import CrossValidator, CVResult, FoldResult, CROSS_VALIDATOR_REGISTRY
from .calibration_scorer import CalibrationScorer, CalibrationResult, CalibrationBin, CALIBRATION_SCORER_REGISTRY

EVALUATION_REGISTRY = {
    "eval_harness": EVAL_HARNESS_REGISTRY,
    "benchmark_runner": BENCHMARK_RUNNER_REGISTRY,
    "metric_aggregator": METRIC_AGGREGATOR_REGISTRY,
    "perplexity_scorer": PERPLEXITY_SCORER_REGISTRY,
    "rouge_scorer": ROUGE_SCORER_REGISTRY,
    "code_eval": CODE_EVALUATOR_REGISTRY,
    "bleu_scorer": BLEU_SCORER_REGISTRY,
    "faithfulness_scorer": FAITHFULNESS_SCORER_REGISTRY,
    "f1_scorer": F1_SCORER_REGISTRY,
    "leaderboard_tracker": LEADERBOARD_TRACKER_REGISTRY,
    "cross_validation": CROSS_VALIDATOR_REGISTRY,
    "calibration_scorer": CALIBRATION_SCORER_REGISTRY,
}
