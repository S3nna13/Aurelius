"""Benchmark configuration and expected performance ranges for Aurelius 1.3B.

Each benchmark defines:
- ``task``: lm-evaluation-harness task name(s)
- ``metric``: primary metric key reported by the harness
- ``expected_low`` / ``expected_high``: plausible accuracy range for a
  well-trained 1.3B-parameter model (used for regression alerting, not gating).
- ``num_fewshot``: default few-shot count for the benchmark
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Specification for a single evaluation benchmark."""

    name: str
    task: str  # lm-evaluation-harness task name
    metric: str
    num_fewshot: int
    expected_low: float
    expected_high: float
    description: str = ""
    subset: str = ""  # dataset subset/config (empty = default)


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

MATH500 = BenchmarkSpec(
    name="MATH-500",
    task="math",
    metric="exact_match",
    num_fewshot=4,
    expected_low=0.05,
    expected_high=0.20,
    description="500-problem subset of MATH; tests competition-level math reasoning",
)

GPQA_DIAMOND = BenchmarkSpec(
    name="GPQA-Diamond",
    task="gpqa_diamond",
    metric="exact_match",
    num_fewshot=0,
    expected_low=0.25,
    expected_high=0.38,
    description="198 expert-level science questions (PhD-level); tests deep domain reasoning",
    subset="gpqa_diamond",
)

LIVECODEBENCH = BenchmarkSpec(
    name="LiveCodeBench",
    task="livecodebench",
    metric="pass@1",
    num_fewshot=0,
    expected_low=0.05,
    expected_high=0.18,
    description="Contamination-free coding problems from competitive programming sites",
)

AMC_MEMORY = BenchmarkSpec(
    name="AMC-Memory",
    task="amc_memory",
    metric="exact_match",
    num_fewshot=0,
    expected_low=0.70,
    expected_high=1.00,
    description=(
        "Aurelius-specific memory benchmark: cross-session recall, surprise "
        "gate selectivity, consolidation preference, and contradiction quarantine"
    ),
)


# Ordered list used by the lm-evaluation-harness checkpoint runner.
LM_EVAL_BENCHMARKS: list[BenchmarkSpec] = [
    MMLU,
    HELLASWAG,
    ARC_CHALLENGE,
    TRUTHFULQA,
    GSM8K,
    HUMANEVAL,
    MATH500,
    GPQA_DIAMOND,
    LIVECODEBENCH,
]

# Internal Aurelius benchmarks require project-native runners rather than
# lm-evaluation-harness task names.
INTERNAL_BENCHMARKS: list[BenchmarkSpec] = [AMC_MEMORY]

# Full registry list; do not use this as the default lm-eval task list.
ALL_BENCHMARKS: list[BenchmarkSpec] = [
    *LM_EVAL_BENCHMARKS,
    *INTERNAL_BENCHMARKS,
]

BENCHMARK_BY_NAME: dict[str, BenchmarkSpec] = {b.name: b for b in ALL_BENCHMARKS}
