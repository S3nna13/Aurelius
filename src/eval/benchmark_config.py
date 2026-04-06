"""Benchmark configuration and expected performance ranges for Aurelius 1.3B.

Each benchmark defines:
- ``task``: lm-evaluation-harness task name(s)
- ``metric``: primary metric key reported by the harness
- ``expected_low`` / ``expected_high``: plausible accuracy range for a
  well-trained 1.3B-parameter model (used for regression alerting, not gating).
- ``num_fewshot``: default few-shot count for the benchmark
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Specification for a single evaluation benchmark."""

    name: str
    task: str
    metric: str
    num_fewshot: int
    expected_low: float
    expected_high: float
    description: str = ""

    def in_expected_range(self, score: float) -> bool:
        """Return ``True`` if *score* falls within the expected range."""
        return self.expected_low <= score <= self.expected_high

    def status_label(self, score: float) -> str:
        """Human-readable status string for a given score."""
        if score < self.expected_low:
            return "BELOW_EXPECTED"
        if score > self.expected_high:
            return "ABOVE_EXPECTED"
        return "IN_RANGE"


# ---------------------------------------------------------------------------
# Benchmark definitions — expected ranges for a 1.3B decoder-only model
# ---------------------------------------------------------------------------

MMLU = BenchmarkSpec(
    name="MMLU",
    task="mmlu",
    metric="acc",
    num_fewshot=5,
    expected_low=0.42,
    expected_high=0.48,
    description="Massive Multitask Language Understanding (57 subjects)",
)

HELLASWAG = BenchmarkSpec(
    name="HellaSwag",
    task="hellaswag",
    metric="acc_norm",
    num_fewshot=10,
    expected_low=0.65,
    expected_high=0.72,
    description="Commonsense NLI — sentence completion",
)

ARC_CHALLENGE = BenchmarkSpec(
    name="ARC-Challenge",
    task="arc_challenge",
    metric="acc_norm",
    num_fewshot=25,
    expected_low=0.45,
    expected_high=0.55,
    description="AI2 Reasoning Challenge — grade-school science (hard split)",
)

TRUTHFULQA = BenchmarkSpec(
    name="TruthfulQA",
    task="truthfulqa_mc2",
    metric="acc",
    num_fewshot=0,
    expected_low=0.35,
    expected_high=0.45,
    description="TruthfulQA multiple-choice (MC2 scoring)",
)

GSM8K = BenchmarkSpec(
    name="GSM8K",
    task="gsm8k",
    metric="exact_match",
    num_fewshot=5,
    expected_low=0.20,
    expected_high=0.30,
    description="Grade-school math word problems",
)

HUMANEVAL = BenchmarkSpec(
    name="HumanEval",
    task="humaneval",
    metric="pass@1",
    num_fewshot=0,
    expected_low=0.25,
    expected_high=0.35,
    description="OpenAI HumanEval — Python code generation (pass@1)",
)


# Ordered list used by the evaluation harness.
ALL_BENCHMARKS: list[BenchmarkSpec] = [
    MMLU,
    HELLASWAG,
    ARC_CHALLENGE,
    TRUTHFULQA,
    GSM8K,
    HUMANEVAL,
]

BENCHMARK_BY_NAME: dict[str, BenchmarkSpec] = {b.name: b for b in ALL_BENCHMARKS}


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result of running a single benchmark against a checkpoint."""

    benchmark: BenchmarkSpec
    score: float
    checkpoint_path: str
    checkpoint_step: int | None = None
    raw_results: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.benchmark.status_label(self.score)

    @property
    def in_range(self) -> bool:
        return self.benchmark.in_expected_range(self.score)

    def summary_line(self) -> str:
        pct = self.score * 100
        low_pct = self.benchmark.expected_low * 100
        high_pct = self.benchmark.expected_high * 100
        return (
            f"{self.benchmark.name:20s}  {pct:5.1f}%  "
            f"(expected {low_pct:.0f}-{high_pct:.0f}%)  [{self.status}]"
        )
