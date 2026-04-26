"""Aurelius evaluation surface."""

from .benchmark_runner import (
    BENCHMARK_RUNNER_REGISTRY,
)
from .benchmark_runner import (
    BenchmarkResult as BenchmarkResult,
)
from .benchmark_runner import (
    BenchmarkRunner as BenchmarkRunner,
)
from .benchmark_runner import (
    BenchmarkSuite as BenchmarkSuite,
)
from .bleu_scorer import BLEU_SCORER_REGISTRY as BLEU_SCORER_REGISTRY
from .bleu_scorer import BLEUResult as BLEUResult
from .bleu_scorer import BLEUScorer as BLEUScorer
from .calibration_scorer import (
    CALIBRATION_SCORER_REGISTRY,
)
from .calibration_scorer import (
    CalibrationBin as CalibrationBin,
)
from .calibration_scorer import (
    CalibrationResult as CalibrationResult,
)
from .calibration_scorer import (
    CalibrationScorer as CalibrationScorer,
)
from .code_eval import CODE_EVALUATOR_REGISTRY as CODE_EVALUATOR_REGISTRY
from .code_eval import CodeEvalConfig as CodeEvalConfig
from .code_eval import CodeEvalResult as CodeEvalResult
from .code_eval import CodeEvaluator as CodeEvaluator
from .cross_validation import CROSS_VALIDATOR_REGISTRY as CROSS_VALIDATOR_REGISTRY
from .cross_validation import CrossValidator as CrossValidator
from .cross_validation import CVResult as CVResult
from .cross_validation import FoldResult as FoldResult
from .eval_harness import EVAL_HARNESS_REGISTRY as EVAL_HARNESS_REGISTRY
from .eval_harness import EvalHarness as EvalHarness
from .eval_harness import EvalResult as EvalResult
from .eval_harness import EvalTask as EvalTask
from .f1_scorer import F1_SCORER_REGISTRY as F1_SCORER_REGISTRY
from .f1_scorer import F1Result as F1Result
from .f1_scorer import F1Scorer as F1Scorer
from .faithfulness_scorer import (
    FAITHFULNESS_SCORER_REGISTRY,
)
from .faithfulness_scorer import (
    FaithfulnessResult as FaithfulnessResult,
)
from .faithfulness_scorer import (
    FaithfulnessScorer as FaithfulnessScorer,
)
from .leaderboard_tracker import LEADERBOARD_TRACKER_REGISTRY as LEADERBOARD_TRACKER_REGISTRY
from .leaderboard_tracker import Leaderboard as Leaderboard
from .leaderboard_tracker import LeaderboardEntry as LeaderboardEntry
from .metric_aggregator import (
    METRIC_AGGREGATOR_REGISTRY,
)
from .metric_aggregator import (
    MetricAggregator as MetricAggregator,
)
from .metric_aggregator import (
    MetricScore as MetricScore,
)
from .metric_aggregator import (
    MetricWeight as MetricWeight,
)
from .perplexity_scorer import PERPLEXITY_SCORER_REGISTRY as PERPLEXITY_SCORER_REGISTRY
from .perplexity_scorer import PerplexityResult as PerplexityResult
from .perplexity_scorer import PerplexityScorer as PerplexityScorer
from .rouge_scorer import ROUGE_SCORER_REGISTRY as ROUGE_SCORER_REGISTRY
from .rouge_scorer import RougeScorer as RougeScorer
from .rouge_scorer import RougeScores as RougeScores

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
