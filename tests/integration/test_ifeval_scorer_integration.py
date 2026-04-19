"""Integration tests: IFEval registry + end-to-end scoring."""

from __future__ import annotations

from src import eval as ev


def test_metric_registry_has_ifeval() -> None:
    assert "ifeval" in ev.METRIC_REGISTRY


def test_benchmark_registry_has_ifeval() -> None:
    assert "ifeval" in ev.BENCHMARK_REGISTRY


def test_prior_registry_entries_intact() -> None:
    for key in ("niah", "ruler", "humaneval", "mbpp", "swebench_lite"):
        assert key in ev.METRIC_REGISTRY, f"METRIC_REGISTRY missing {key}"
        assert key in ev.BENCHMARK_REGISTRY, f"BENCHMARK_REGISTRY missing {key}"


def test_end_to_end_single_problem_two_constraints() -> None:
    problem = ev.IFEvalProblem(
        prompt="Greet the user with a short hello containing the word 'hello'.",
        constraints=[
            ev.IFEvalConstraint("contains_keyword", {"keyword": "hello"}),
            ev.IFEvalConstraint("length_words", {"min": 2, "max": 8}),
        ],
    )
    response = "hello there friend"

    scorer = ev.IFEvalScorer()
    single = scorer.score_one(problem, response)
    assert single.strict_pass is True
    assert single.passed == [True, True]

    agg = scorer.score([problem], [response])
    assert agg["n_problems"] == 1
    assert agg["strict_accuracy"] == 1.0
    assert agg["loose_accuracy"] == 1.0
    assert set(agg["per_type_accuracy"]) == {"contains_keyword", "length_words"}
