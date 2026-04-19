from .benchmark_config import (
    BenchmarkSpec,
    ALL_BENCHMARKS,
    BENCHMARK_BY_NAME,
    MMLU,
    HELLASWAG,
    ARC_CHALLENGE,
    TRUTHFULQA,
    GSM8K,
    HUMANEVAL,
    MATH500,
    GPQA_DIAMOND,
    LIVECODEBENCH,
)

__all__ = [
    "BenchmarkSpec",
    "ALL_BENCHMARKS",
    "BENCHMARK_BY_NAME",
    "MMLU",
    "HELLASWAG",
    "ARC_CHALLENGE",
    "TRUTHFULQA",
    "GSM8K",
    "HUMANEVAL",
    "MATH500",
    "GPQA_DIAMOND",
    "LIVECODEBENCH",
]

# --- Additive registration for Needle-in-a-Haystack (NIAH) -------------------
# Safe to import lazily; does not remove or override any existing symbols.
from .needle_in_haystack import (
    NeedleInHaystackBenchmark as _NeedleInHaystackBenchmark,
    default_haystack_filler as _default_haystack_filler,
)

NeedleInHaystackBenchmark = _NeedleInHaystackBenchmark
default_haystack_filler = _default_haystack_filler

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("niah", NeedleInHaystackBenchmark)
BENCHMARK_REGISTRY.setdefault("niah", NeedleInHaystackBenchmark)

__all__ = list(__all__) + [
    "NeedleInHaystackBenchmark",
    "default_haystack_filler",
    "METRIC_REGISTRY",
    "BENCHMARK_REGISTRY",
]

# --- Additive registration for RULER long-context benchmark ------------------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above
# and leaves the "niah" entries untouched.
from .ruler_benchmark import RULERBenchmark as _RULERBenchmark

RULERBenchmark = _RULERBenchmark

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("ruler", RULERBenchmark)
BENCHMARK_REGISTRY.setdefault("ruler", RULERBenchmark)

__all__ = list(__all__) + ["RULERBenchmark"]

# --- Additive registration for HumanEval functional-correctness benchmark ----
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above
# and leaves the "niah" and "ruler" entries untouched.
from .humaneval_scorer import (
    HumanEvalProblem as _HumanEvalProblem,
    SampleResult as _HumanEvalSampleResult,
    score_single as _humaneval_score_single,
    score_problems as _humaneval_score_problems,
    pass_at_k as _humaneval_pass_at_k,
)

HumanEvalProblem = _HumanEvalProblem
HumanEvalSampleResult = _HumanEvalSampleResult
humaneval_score_single = _humaneval_score_single
humaneval_score_problems = _humaneval_score_problems
humaneval_pass_at_k = _humaneval_pass_at_k

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("humaneval", _humaneval_score_problems)
BENCHMARK_REGISTRY.setdefault("humaneval", _HumanEvalProblem)

__all__ = list(__all__) + [
    "HumanEvalProblem",
    "HumanEvalSampleResult",
    "humaneval_score_single",
    "humaneval_score_problems",
    "humaneval_pass_at_k",
]
