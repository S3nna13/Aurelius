"""Tests for src/eval/benchmark_runner.py"""

from src.eval.benchmark_runner import BenchmarkResult, BenchmarkRunner, RunConfig

# ---------------------------------------------------------------------------
# RunConfig defaults
# ---------------------------------------------------------------------------


def test_runconfig_default_max_samples():
    cfg = RunConfig(benchmark_names=["foo"])
    assert cfg.max_samples == 100


def test_runconfig_default_timeout():
    cfg = RunConfig(benchmark_names=["foo"])
    assert cfg.timeout_seconds == 300.0


def test_runconfig_default_seed():
    cfg = RunConfig(benchmark_names=["foo"])
    assert cfg.seed == 42


def test_runconfig_benchmark_names():
    cfg = RunConfig(benchmark_names=["a", "b"])
    assert cfg.benchmark_names == ["a", "b"]


def test_runconfig_custom_values():
    cfg = RunConfig(benchmark_names=["x"], max_samples=50, timeout_seconds=60.0, seed=7)
    assert cfg.max_samples == 50
    assert cfg.timeout_seconds == 60.0
    assert cfg.seed == 7


# ---------------------------------------------------------------------------
# BenchmarkResult fields
# ---------------------------------------------------------------------------


def test_benchmarkresult_fields():
    r = BenchmarkResult("foo", 0.9, 10, 1.23)
    assert r.benchmark_name == "foo"
    assert r.score == 0.9
    assert r.n_samples == 10
    assert r.elapsed_seconds == 1.23


def test_benchmarkresult_default_metadata():
    r = BenchmarkResult("bar", 0.5, 5, 0.1)
    assert isinstance(r.metadata, dict)
    assert r.metadata == {}


def test_benchmarkresult_custom_metadata():
    r = BenchmarkResult("baz", 0.7, 20, 2.0, metadata={"note": "ok"})
    assert r.metadata["note"] == "ok"


# ---------------------------------------------------------------------------
# BenchmarkRunner construction
# ---------------------------------------------------------------------------


def test_runner_accepts_empty_registry():
    runner = BenchmarkRunner(benchmark_registry={})
    assert runner is not None


def test_runner_accepts_custom_registry():
    registry = {"test_bench": object()}
    runner = BenchmarkRunner(benchmark_registry=registry)
    assert runner is not None


def test_runner_default_registry_not_none():
    runner = BenchmarkRunner()
    assert runner._registry is not None


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


def _make_runner_with_stub(score_val=0.75, total_val=10):
    class StubBenchmark:
        def evaluate(self, predictions):
            return {"accuracy": score_val, "total": total_val}

    registry = {"stub": StubBenchmark()}
    return BenchmarkRunner(benchmark_registry=registry)


def test_run_benchmark_returns_benchmark_result():
    runner = _make_runner_with_stub()
    result = runner.run_benchmark("stub", {"q1": "a1"})
    assert isinstance(result, BenchmarkResult)


def test_run_benchmark_name():
    runner = _make_runner_with_stub()
    result = runner.run_benchmark("stub", {"q1": "a1"})
    assert result.benchmark_name == "stub"


def test_run_benchmark_score_from_evaluate():
    runner = _make_runner_with_stub(score_val=0.75)
    result = runner.run_benchmark("stub", {"q1": "a1"})
    assert result.score == 0.75


def test_run_benchmark_score_in_range():
    runner = _make_runner_with_stub(score_val=0.5)
    result = runner.run_benchmark("stub", {"q1": "a1"})
    assert 0.0 <= result.score <= 1.0


def test_run_benchmark_n_samples_from_evaluate():
    runner = _make_runner_with_stub(total_val=10)
    result = runner.run_benchmark("stub", {"q1": "a1"})
    assert result.n_samples == 10


def test_run_benchmark_elapsed_nonnegative():
    runner = _make_runner_with_stub()
    result = runner.run_benchmark("stub", {"q1": "a1"})
    assert result.elapsed_seconds >= 0.0


def test_run_benchmark_unknown_name_returns_result():
    runner = BenchmarkRunner(benchmark_registry={})
    result = runner.run_benchmark("nonexistent", {"q": "a"})
    assert isinstance(result, BenchmarkResult)


def test_run_benchmark_unknown_name_score_zero():
    runner = BenchmarkRunner(benchmark_registry={})
    result = runner.run_benchmark("nonexistent", {"q": "a"})
    assert result.score == 0.0


def test_run_benchmark_with_config():
    runner = _make_runner_with_stub()
    cfg = RunConfig(benchmark_names=["stub"], max_samples=50)
    result = runner.run_benchmark("stub", {"q1": "a1"}, config=cfg)
    assert isinstance(result, BenchmarkResult)


def test_run_benchmark_evaluate_exception_returns_zero_score():
    class FailBenchmark:
        def evaluate(self, predictions):
            raise RuntimeError("intentional")

    registry = {"fail": FailBenchmark()}
    runner = BenchmarkRunner(benchmark_registry=registry)
    result = runner.run_benchmark("fail", {"q": "a"})
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


def _make_multi_runner():
    class Bench:
        def __init__(self, acc):
            self._acc = acc

        def evaluate(self, predictions):
            return {"accuracy": self._acc, "total": len(predictions)}

    registry = {"bench_a": Bench(0.8), "bench_b": Bench(0.6), "bench_c": Bench(0.9)}
    return BenchmarkRunner(benchmark_registry=registry)


def test_run_all_returns_list():
    runner = _make_multi_runner()
    results = runner.run_all({"q1": "a1"})
    assert isinstance(results, list)


def test_run_all_length_matches_registry():
    runner = _make_multi_runner()
    results = runner.run_all({"q1": "a1"})
    assert len(results) == 3


def test_run_all_all_benchmark_results():
    runner = _make_multi_runner()
    results = runner.run_all({"q1": "a1"})
    assert all(isinstance(r, BenchmarkResult) for r in results)


def test_run_all_empty_registry():
    runner = BenchmarkRunner(benchmark_registry={})
    results = runner.run_all({})
    assert results == []


def test_run_all_contains_all_names():
    runner = _make_multi_runner()
    results = runner.run_all({"q1": "a1"})
    names = {r.benchmark_name for r in results}
    assert "bench_a" in names and "bench_b" in names and "bench_c" in names


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_report_returns_string():
    runner = _make_multi_runner()
    results = runner.run_all({"q1": "a1"})
    report = runner.report(results)
    assert isinstance(report, str)


def test_report_non_empty():
    runner = _make_multi_runner()
    results = runner.run_all({"q1": "a1"})
    report = runner.report(results)
    assert len(report) > 0


def test_report_contains_benchmark_names():
    runner = _make_multi_runner()
    results = runner.run_all({"q1": "a1"})
    report = runner.report(results)
    assert "bench_a" in report
    assert "bench_b" in report
    assert "bench_c" in report


def test_report_contains_score_format():
    runner = _make_runner_with_stub(0.75, 10)
    results = [runner.run_benchmark("stub", {"q": "a"})]
    report = runner.report(results)
    assert "0.750" in report


def test_report_empty_list():
    runner = BenchmarkRunner(benchmark_registry={})
    report = runner.report([])
    assert report == ""


# ---------------------------------------------------------------------------
# best
# ---------------------------------------------------------------------------


def test_best_returns_none_for_empty():
    runner = BenchmarkRunner(benchmark_registry={})
    assert runner.best([]) is None


def test_best_returns_benchmark_result():
    runner = _make_multi_runner()
    results = runner.run_all({"q1": "a1"})
    b = runner.best(results)
    assert isinstance(b, BenchmarkResult)


def test_best_highest_score():
    runner = _make_multi_runner()
    results = runner.run_all({"q1": "a1"})
    b = runner.best(results)
    assert b.benchmark_name == "bench_c"
    assert b.score == 0.9


def test_best_single_element():
    r = BenchmarkResult("only", 0.5, 5, 0.1)
    runner = BenchmarkRunner(benchmark_registry={})
    assert runner.best([r]) is r


def test_best_tie_returns_one():
    r1 = BenchmarkResult("a", 0.8, 5, 0.1)
    r2 = BenchmarkResult("b", 0.8, 5, 0.1)
    runner = BenchmarkRunner(benchmark_registry={})
    b = runner.best([r1, r2])
    assert b is not None
    assert b.score == 0.8
