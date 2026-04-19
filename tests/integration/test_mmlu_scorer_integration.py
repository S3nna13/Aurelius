"""Integration test: MMLU scorer is registered and runs end-to-end."""

from __future__ import annotations

from src import eval as eval_pkg
from src.eval.mmlu_scorer import MMLUProblem, MMLUScorer


def test_prior_registry_entries_intact():
    reg = eval_pkg.METRIC_REGISTRY
    for key in (
        "niah",
        "ruler",
        "humaneval",
        "mbpp",
        "swebench_lite",
        "ifeval",
        "mtbench",
        "alpacaeval",
        "arena_hard",
        "gpqa",
        "livecodebench",
    ):
        assert key in reg, f"prior registry entry '{key}' missing"


def test_mmlu_registered_in_metric_registry():
    assert "mmlu" in eval_pkg.METRIC_REGISTRY
    assert "mmlu" in eval_pkg.BENCHMARK_REGISTRY
    assert eval_pkg.BENCHMARK_REGISTRY["mmlu"] is MMLUProblem
    assert eval_pkg.METRIC_REGISTRY["mmlu"] is MMLUScorer


def test_end_to_end_three_problem_scoring():
    probs = [
        MMLUProblem(
            question_id="astro-1",
            subject="astronomy",
            question="Which planet is closest to the Sun?",
            choices=["Venus", "Earth", "Mercury", "Mars"],
            correct_index=2,
        ),
        MMLUProblem(
            question_id="bio-1",
            subject="biology",
            question="The powerhouse of the cell is the?",
            choices=["Nucleus", "Mitochondrion", "Ribosome", "Golgi"],
            correct_index=1,
        ),
        MMLUProblem(
            question_id="hist-1",
            subject="history",
            question="In what year did WWII end?",
            choices=["1943", "1944", "1945", "1946"],
            correct_index=2,
        ),
    ]

    responses = [
        "Mercury is closest. Answer: C",
        "[[B]]",
        "Final answer: (C)",
    ]

    scorer = MMLUScorer(n_shots=0)
    agg = scorer.score(probs, responses)
    assert agg["overall_accuracy"] == 1.0
    assert agg["n_valid"] == 3
    assert agg["n_total"] == 3
    assert agg["per_subject"]["astronomy"]["accuracy"] == 1.0
    assert agg["per_subject"]["biology"]["accuracy"] == 1.0
    assert agg["per_subject"]["history"]["accuracy"] == 1.0

    # Exercise run() with a deterministic stub generate_fn that keys off
    # question content.
    def gen(prompt: str) -> str:
        if "closest to the Sun" in prompt:
            return "Answer: C"
        if "powerhouse of the cell" in prompt:
            return "[[B]]"
        if "WWII end" in prompt:
            return "Answer: A"  # wrong
        return "Answer: A"

    s2 = MMLUScorer(generate_fn=gen, n_shots=2)
    results = s2.run(probs)
    assert len(results) == 3
    assert results[0].correct is True
    assert results[1].correct is True
    assert results[2].correct is False
    assert results[2].predicted_letter == "A"
    assert results[2].subject == "history"
