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
