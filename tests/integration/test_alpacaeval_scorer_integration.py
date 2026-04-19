"""Integration: AlpacaEval scorer registered in METRIC_REGISTRY; end-to-end."""

from __future__ import annotations

import src.eval as eval_pkg
from src.eval.alpacaeval_scorer import (
    AlpacaComparison,
    AlpacaEvalScorer,
    AlpacaProblem,
)


def test_metric_registry_contains_alpacaeval():
    assert "alpacaeval" in eval_pkg.METRIC_REGISTRY
    assert eval_pkg.METRIC_REGISTRY["alpacaeval"] is AlpacaEvalScorer


def test_prior_registry_entries_intact():
    for name in (
        "niah",
        "ruler",
        "humaneval",
        "mbpp",
        "swebench_lite",
        "ifeval",
        "mtbench",
    ):
        assert name in eval_pkg.METRIC_REGISTRY, name


def test_module_level_exports():
    assert hasattr(eval_pkg, "AlpacaEvalScorer")
    assert hasattr(eval_pkg, "AlpacaProblem")
    assert hasattr(eval_pkg, "AlpacaComparison")


def test_end_to_end_three_problem_scoring():
    # Judge prefers whichever side contains the GOOD marker.
    def judge_fn(prompt: str) -> str:
        a_idx = prompt.find("The Start of Assistant A")
        b_idx = prompt.find("The Start of Assistant B")
        a_block = prompt[a_idx:b_idx]
        b_block = prompt[b_idx:]
        a = "GOOD" in a_block
        b = "GOOD" in b_block
        if a and not b:
            return "A wins. [[A]]"
        if b and not a:
            return "B wins. [[B]]"
        return "equal. [[C]]"

    scorer = AlpacaEvalScorer(judge_fn, swap_order=True)
    problems = [
        AlpacaProblem("Write a poem.", "a baseline poem"),
        AlpacaProblem("Explain FFT.", "a baseline explanation"),
        AlpacaProblem("Translate hi.", "a baseline translation"),
    ]
    candidates = [
        "GOOD candidate poem that is clearly better",
        "mediocre candidate explanation",  # tie (neither has GOOD)
        "GOOD candidate translation also better",
    ]
    result = scorer.score(problems, candidates)
    assert result["n_total"] == 3
    assert result["n_valid"] == 3
    # Two candidate wins, one tie.
    assert abs(result["win_rate"] - 2 / 3) < 1e-9
    assert abs(result["tie_rate"] - 1 / 3) < 1e-9
    assert result["reference_rate"] == 0.0
    # LC in [0, 1].
    assert 0.0 <= result["length_controlled_winrate"] <= 1.0

    # Sanity: each comparison round-trips the dataclass type.
    comps = [scorer.compare(p, c) for p, c in zip(problems, candidates)]
    assert all(isinstance(c, AlpacaComparison) for c in comps)
