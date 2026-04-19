"""Integration: MT-Bench judge registered in METRIC_REGISTRY; end-to-end flow."""

from __future__ import annotations

import src.eval as eval_pkg
from src.eval.mtbench_judge import (
    MTBenchJudge,
    MTBenchQuestion,
    PairwiseResult,
    SingleAnswerScore,
)


def test_metric_registry_contains_mtbench():
    assert "mtbench" in eval_pkg.METRIC_REGISTRY
    assert eval_pkg.METRIC_REGISTRY["mtbench"] is MTBenchJudge


def test_prior_registry_entries_intact():
    # Prior entries from other additive registrations must remain present.
    for name in ("niah", "ruler", "humaneval", "mbpp", "swebench_lite", "ifeval"):
        assert name in eval_pkg.METRIC_REGISTRY, name


def test_module_level_exports():
    assert hasattr(eval_pkg, "MTBenchJudge")
    assert hasattr(eval_pkg, "MTBenchQuestion")
    assert hasattr(eval_pkg, "SingleAnswerScore")
    assert hasattr(eval_pkg, "PairwiseResult")


def test_end_to_end_single_scoring():
    # Deterministic fake judge that scores based on answer length.
    def judge_fn(prompt: str) -> str:
        # prompt content: look for a token embedded in the answer block.
        if "excellent" in prompt:
            return "Great answer. Rating: [[9]]"
        if "mediocre" in prompt:
            return "Decent. Rating: [[5]]"
        return "Poor. Rating: [[2]]"

    judge = MTBenchJudge(judge_fn)
    qs = [
        MTBenchQuestion("q1", "writing", ["Write a story."]),
        MTBenchQuestion("q2", "reasoning", ["Explain Bayes."]),
        MTBenchQuestion("q3", "math", ["Solve x+1=2."], reference="x=1"),
    ]
    answers = ["this is excellent", "this is mediocre", "bad"]
    results = [judge.score_single(q, a) for q, a in zip(qs, answers)]
    assert [r.score for r in results] == [9.0, 5.0, 2.0]
    assert all(isinstance(r, SingleAnswerScore) for r in results)
    agg = MTBenchJudge.aggregate_single(results)
    assert agg["n_valid"] == 3
    assert abs(agg["mean"] - (9 + 5 + 2) / 3) < 1e-9


def test_end_to_end_pairwise():
    def judge_fn(prompt: str) -> str:
        # Prefer whichever side contains the literal "[WIN]" marker.
        a_idx = prompt.find("The Start of Assistant A")
        b_idx = prompt.find("The Start of Assistant B")
        a_block = prompt[a_idx:b_idx]
        b_block = prompt[b_idx:]
        a_win = "[WIN]" in a_block
        b_win = "[WIN]" in b_block
        if a_win and not b_win:
            return "A is better. [[A]]"
        if b_win and not a_win:
            return "B is better. [[B]]"
        return "They are equivalent. [[C]]"

    judge = MTBenchJudge(judge_fn)
    q = MTBenchQuestion("p1", "writing", ["Compare."])
    results = [
        judge.score_pairwise(q, "[WIN] good", "bad"),
        judge.score_pairwise(q, "bad", "[WIN] good"),
        judge.score_pairwise(q, "same", "same"),
        judge.score_pairwise(q, "[WIN] x", "[WIN] y"),  # tie via both
    ]
    winners = [r.winner for r in results]
    assert winners == ["A", "B", "tie", "tie"]
    assert all(isinstance(r, PairwiseResult) for r in results)
    agg = MTBenchJudge.aggregate_pairwise(results)
    assert agg["n_valid"] == 4
    assert abs(agg["win_rate_a"] + agg["win_rate_b"] + agg["tie_rate"] - 1.0) < 1e-9
