"""Integration test: GPQA scorer is registered and runs end-to-end."""

from __future__ import annotations

from src import eval as eval_pkg
from src.eval.gpqa_scorer import GPQAProblem, GPQAScorer


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
    ):
        assert key in reg, f"prior registry entry '{key}' missing"


def test_gpqa_registered_in_metric_registry():
    assert "gpqa" in eval_pkg.METRIC_REGISTRY
    assert "gpqa" in eval_pkg.BENCHMARK_REGISTRY
    # Sanity: the benchmark registry entry is the problem dataclass
    assert eval_pkg.BENCHMARK_REGISTRY["gpqa"] is GPQAProblem


def test_end_to_end_two_problem_scoring():
    probs = [
        GPQAProblem(
            question_id="bio-1",
            question="Which organelle is the powerhouse of the cell?",
            choices=["Nucleus", "Mitochondrion", "Ribosome", "Golgi apparatus"],
            correct_index=1,
            domain="biology",
            difficulty="hard",
        ),
        GPQAProblem(
            question_id="chem-1",
            question="What is the chemical symbol for gold?",
            choices=["Ag", "Gd", "Au", "Go"],
            correct_index=2,
            domain="chemistry",
            difficulty="hard",
        ),
    ]

    # First response correct via [[B]], second correct via "Answer: C"
    responses = [
        "Reasoning ... final answer is B.\n[[B]]",
        "The symbol is Au. Answer: C",
    ]

    scorer = GPQAScorer()
    agg = scorer.score(probs, responses)
    assert agg["overall_accuracy"] == 1.0
    assert agg["n_valid"] == 2
    assert agg["per_domain"]["biology"]["accuracy"] == 1.0
    assert agg["per_domain"]["chemistry"]["accuracy"] == 1.0

    # Also exercise run() via a deterministic stub generate_fn
    canned = {
        "bio-1": "[[B]]",
        "chem-1": "Answer: A",  # wrong
    }

    def gen(prompt: str) -> str:
        for qid, resp in canned.items():
            if qid in prompt or canned_match(prompt, qid):
                return resp
        # Fall back by first-choice heuristic for the stub
        return "Answer: A"

    def canned_match(prompt: str, qid: str) -> bool:
        if qid == "bio-1":
            return "powerhouse" in prompt
        if qid == "chem-1":
            return "gold" in prompt
        return False

    s2 = GPQAScorer(generate_fn=gen)
    results = s2.run(probs)
    assert len(results) == 2
    assert results[0].correct is True
    assert results[1].correct is False
    assert results[1].predicted_letter == "A"
