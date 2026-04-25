"""Tests for src/evaluation/benchmark_runner.py — ≥28 test cases."""

import pytest

from src.evaluation.benchmark_runner import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    BENCHMARK_RUNNER_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _always_correct_predict(prompt: str) -> str:
    """Returns prompt unchanged — correct when reference == prompt."""
    return prompt


def _always_wrong_predict(prompt: str) -> str:
    return "__never_matches__"


def _make_suite(name="suite1", prompts_refs=None, metric="accuracy"):
    if prompts_refs is None:
        prompts_refs = [("What is 2+2?", "4"), ("Capital of France?", "Paris")]
    tasks = [{"prompt": p, "reference": r} for p, r in prompts_refs]
    return BenchmarkSuite(name=name, tasks=tasks, metric=metric)


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------

class TestBenchmarkSuite:
    def test_suite_creation(self):
        suite = BenchmarkSuite(name="s1", tasks=[])
        assert suite.name == "s1"
        assert suite.tasks == []

    def test_suite_default_metric(self):
        suite = BenchmarkSuite(name="s1", tasks=[])
        assert suite.metric == "accuracy"

    def test_suite_custom_metric(self):
        suite = BenchmarkSuite(name="s1", tasks=[], metric="f1")
        assert suite.metric == "f1"


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def test_result_frozen(self):
        import dataclasses
        result = BenchmarkResult(
            suite_name="s1", metric="accuracy", score=1.0,
            num_tasks=1, details=[]
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.score = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BenchmarkRunner.run_suite
# ---------------------------------------------------------------------------

class TestRunSuite:
    def test_run_suite_empty(self):
        suite = BenchmarkSuite(name="empty", tasks=[])
        runner = BenchmarkRunner([suite])
        result = runner.run_suite(suite, _always_correct_predict)
        assert result.num_tasks == 0
        assert result.score == 0.0
        assert result.details == []

    def test_run_suite_all_correct(self):
        # prompt == reference so identity predict is always correct
        suite = _make_suite(prompts_refs=[("ans", "ans"), ("foo", "foo")])
        runner = BenchmarkRunner([suite])
        result = runner.run_suite(suite, _always_correct_predict)
        assert result.score == pytest.approx(1.0)
        assert result.num_tasks == 2

    def test_run_suite_all_wrong(self):
        suite = _make_suite(prompts_refs=[("q1", "a1"), ("q2", "a2")])
        runner = BenchmarkRunner([suite])
        result = runner.run_suite(suite, _always_wrong_predict)
        assert result.score == pytest.approx(0.0)

    def test_run_suite_partial_correct(self):
        # Only the first task has prompt == reference
        suite = _make_suite(prompts_refs=[("a", "a"), ("b", "x")])
        runner = BenchmarkRunner([suite])
        result = runner.run_suite(suite, _always_correct_predict)
        assert result.score == pytest.approx(0.5)

    def test_run_suite_suite_name_in_result(self):
        suite = _make_suite(name="MySuite")
        runner = BenchmarkRunner([suite])
        result = runner.run_suite(suite, _always_wrong_predict)
        assert result.suite_name == "MySuite"

    def test_run_suite_metric_in_result(self):
        suite = _make_suite(metric="bleu")
        runner = BenchmarkRunner([suite])
        result = runner.run_suite(suite, _always_wrong_predict)
        assert result.metric == "bleu"

    def test_run_suite_num_tasks(self):
        tasks = [{"prompt": f"q{i}", "reference": f"a{i}"} for i in range(7)]
        suite = BenchmarkSuite(name="s", tasks=tasks)
        runner = BenchmarkRunner([suite])
        result = runner.run_suite(suite, _always_wrong_predict)
        assert result.num_tasks == 7

    def test_run_suite_details_length(self):
        suite = _make_suite(prompts_refs=[("q1", "a1"), ("q2", "a2"), ("q3", "a3")])
        runner = BenchmarkRunner([suite])
        result = runner.run_suite(suite, _always_wrong_predict)
        assert len(result.details) == 3

    def test_run_suite_details_contain_prompt(self):
        suite = _make_suite(prompts_refs=[("my_prompt", "ref")])
        runner = BenchmarkRunner([suite])
        result = runner.run_suite(suite, _always_wrong_predict)
        assert result.details[0]["prompt"] == "my_prompt"

    def test_run_suite_details_contain_predicted(self):
        suite = _make_suite(prompts_refs=[("q", "ref")])
        runner = BenchmarkRunner([suite])
        result = runner.run_suite(suite, _always_wrong_predict)
        assert "predicted" in result.details[0]
        assert result.details[0]["predicted"] == "__never_matches__"

    def test_run_suite_details_contain_correct(self):
        suite = _make_suite(prompts_refs=[("q", "q")])
        runner = BenchmarkRunner([suite])
        result = runner.run_suite(suite, _always_correct_predict)
        assert "correct" in result.details[0]
        assert result.details[0]["correct"] is True

    def test_run_suite_details_correct_false_when_wrong(self):
        suite = _make_suite(prompts_refs=[("q", "different")])
        runner = BenchmarkRunner([suite])
        result = runner.run_suite(suite, _always_correct_predict)
        assert result.details[0]["correct"] is False

    def test_run_suite_strips_whitespace_for_exact_match(self):
        suite = BenchmarkSuite(name="s", tasks=[{"prompt": " ans ", "reference": "ans"}])
        runner = BenchmarkRunner([suite])
        result = runner.run_suite(suite, _always_correct_predict)
        assert result.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BenchmarkRunner.run_all
# ---------------------------------------------------------------------------

class TestRunAll:
    def test_run_all_returns_all_results(self):
        suites = [_make_suite(f"suite{i}") for i in range(3)]
        runner = BenchmarkRunner(suites)
        results = runner.run_all(_always_wrong_predict)
        assert len(results) == 3

    def test_run_all_empty_suites(self):
        runner = BenchmarkRunner([])
        results = runner.run_all(_always_wrong_predict)
        assert results == []

    def test_run_all_result_names(self):
        names = ["alpha", "beta", "gamma"]
        suites = [_make_suite(name=n) for n in names]
        runner = BenchmarkRunner(suites)
        results = runner.run_all(_always_wrong_predict)
        result_names = [r.suite_name for r in results]
        assert set(result_names) == set(names)


# ---------------------------------------------------------------------------
# BenchmarkRunner.leaderboard
# ---------------------------------------------------------------------------

class TestLeaderboard:
    def test_leaderboard_sorted_descending(self):
        results = [
            BenchmarkResult(suite_name="A", metric="accuracy", score=0.4, num_tasks=5, details=[]),
            BenchmarkResult(suite_name="B", metric="accuracy", score=0.9, num_tasks=5, details=[]),
            BenchmarkResult(suite_name="C", metric="accuracy", score=0.6, num_tasks=5, details=[]),
        ]
        runner = BenchmarkRunner([])
        board = runner.leaderboard(results)
        scores = [e["score"] for e in board]
        assert scores == sorted(scores, reverse=True)

    def test_leaderboard_ranks_start_at_1(self):
        results = [
            BenchmarkResult(suite_name="X", metric="accuracy", score=0.5, num_tasks=2, details=[]),
        ]
        runner = BenchmarkRunner([])
        board = runner.leaderboard(results)
        assert board[0]["rank"] == 1

    def test_leaderboard_ranks_sequential(self):
        results = [
            BenchmarkResult(suite_name=f"S{i}", metric="accuracy", score=float(i) / 10, num_tasks=1, details=[])
            for i in range(5)
        ]
        runner = BenchmarkRunner([])
        board = runner.leaderboard(results)
        assert [e["rank"] for e in board] == [1, 2, 3, 4, 5]

    def test_leaderboard_entry_has_suite(self):
        results = [
            BenchmarkResult(suite_name="MySuite", metric="accuracy", score=0.8, num_tasks=2, details=[]),
        ]
        runner = BenchmarkRunner([])
        board = runner.leaderboard(results)
        assert board[0]["suite"] == "MySuite"

    def test_leaderboard_entry_has_score(self):
        results = [
            BenchmarkResult(suite_name="S", metric="accuracy", score=0.75, num_tasks=2, details=[]),
        ]
        runner = BenchmarkRunner([])
        board = runner.leaderboard(results)
        assert board[0]["score"] == pytest.approx(0.75)

    def test_leaderboard_empty(self):
        runner = BenchmarkRunner([])
        board = runner.leaderboard([])
        assert board == []

    def test_leaderboard_first_is_best(self):
        results = [
            BenchmarkResult(suite_name="Low", metric="accuracy", score=0.1, num_tasks=1, details=[]),
            BenchmarkResult(suite_name="High", metric="accuracy", score=0.99, num_tasks=1, details=[]),
        ]
        runner = BenchmarkRunner([])
        board = runner.leaderboard(results)
        assert board[0]["suite"] == "High"


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in BENCHMARK_RUNNER_REGISTRY

    def test_registry_default_is_benchmark_runner(self):
        assert BENCHMARK_RUNNER_REGISTRY["default"] is BenchmarkRunner
