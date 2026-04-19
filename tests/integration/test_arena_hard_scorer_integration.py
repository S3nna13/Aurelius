"""Integration tests for Arena-Hard: registry entry + end-to-end 3-model round-robin."""

from __future__ import annotations

from src import eval as eval_pkg
from src.eval.arena_hard_scorer import (
    ArenaComparison,
    ArenaHardScorer,
    ArenaProblem,
)


def test_metric_registry_has_arena_hard():
    assert "arena_hard" in eval_pkg.METRIC_REGISTRY
    assert eval_pkg.METRIC_REGISTRY["arena_hard"] is ArenaHardScorer
    assert "arena_hard" in eval_pkg.BENCHMARK_REGISTRY
    assert eval_pkg.BENCHMARK_REGISTRY["arena_hard"] is ArenaProblem


def test_prior_registry_entries_intact():
    expected = {
        "niah",
        "ruler",
        "humaneval",
        "mbpp",
        "swebench_lite",
        "ifeval",
        "mtbench",
        "alpacaeval",
    }
    for key in expected:
        assert key in eval_pkg.METRIC_REGISTRY, f"missing METRIC_REGISTRY[{key!r}]"
        assert key in eval_pkg.BENCHMARK_REGISTRY, f"missing BENCHMARK_REGISTRY[{key!r}]"


def test_end_to_end_three_model_round_robin():
    """3 models x 2 problems; judge prefers answers containing 'STRONG'."""

    def judge(prompt: str) -> str:
        a_start = prompt.index("[The Start of Assistant A's Answer]")
        a_end = prompt.index("[The End of Assistant A's Answer]")
        b_start = prompt.index("[The Start of Assistant B's Answer]")
        b_end = prompt.index("[The End of Assistant B's Answer]")
        a_text = prompt[a_start:a_end]
        b_text = prompt[b_start:b_end]
        a_strong = "STRONG" in a_text
        b_strong = "STRONG" in b_text
        a_mid = "MID" in a_text
        b_mid = "MID" in b_text
        # Strong beats all. Mid beats weak. Tie otherwise.
        if a_strong and not b_strong:
            return "[[A]]"
        if b_strong and not a_strong:
            return "[[B]]"
        if a_mid and not b_mid:
            return "[[A]]"
        if b_mid and not a_mid:
            return "[[B]]"
        return "[[C]]"

    scorer = ArenaHardScorer(judge_fn=judge, swap_order=True)
    problems = [
        ArenaProblem(prompt_id="p1", prompt="Hard prompt 1", category="reasoning"),
        ArenaProblem(prompt_id="p2", prompt="Hard prompt 2", category="coding"),
    ]
    responses = {
        "strong_model": ["STRONG answer to p1", "STRONG answer to p2"],
        "mid_model": ["MID answer to p1", "MID answer to p2"],
        "weak_model": ["weak answer p1", "weak answer p2"],
    }

    comps = scorer.run_round_robin(problems, responses)
    # 3 pairs x 2 problems = 6 comparisons.
    assert len(comps) == 6
    assert all(isinstance(c, ArenaComparison) for c in comps)

    names = list(responses.keys())
    ratings = ArenaHardScorer.fit_bradley_terry(comps, names, n_iters=200)
    assert ratings["strong_model"] > ratings["mid_model"] > ratings["weak_model"]

    ci = ArenaHardScorer.bootstrap_confidence_intervals(
        comps, names, n_bootstrap=30, ci=0.9, seed=7
    )
    for m in names:
        mean, lo, hi = ci[m]
        assert lo <= mean <= hi
