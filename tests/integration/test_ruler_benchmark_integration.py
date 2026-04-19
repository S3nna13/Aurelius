"""Integration tests for RULER benchmark registration and end-to-end run."""

from __future__ import annotations

import src.eval as eval_pkg
from src.eval.ruler_benchmark import RULERBenchmark


def test_metric_registry_has_both_niah_and_ruler():
    assert "niah" in eval_pkg.METRIC_REGISTRY
    assert "ruler" in eval_pkg.METRIC_REGISTRY
    assert eval_pkg.METRIC_REGISTRY["ruler"] is RULERBenchmark


def test_benchmark_registry_has_both_niah_and_ruler():
    assert "niah" in eval_pkg.BENCHMARK_REGISTRY
    assert "ruler" in eval_pkg.BENCHMARK_REGISTRY
    assert eval_pkg.BENCHMARK_REGISTRY["ruler"] is RULERBenchmark


def test_existing_niah_registration_untouched():
    # The NIAH registrations should still resolve to the NIAH class, not RULER.
    niah_cls = eval_pkg.METRIC_REGISTRY["niah"]
    assert niah_cls is not RULERBenchmark
    assert niah_cls.__name__ == "NeedleInHaystackBenchmark"


def test_ruler_exposed_on_package():
    assert hasattr(eval_pkg, "RULERBenchmark")
    assert eval_pkg.RULERBenchmark is RULERBenchmark


def test_end_to_end_tiny_run_via_registry():
    cls = eval_pkg.BENCHMARK_REGISTRY["ruler"]
    bench = cls()
    # Oracle: replay each builder with matching seeds to synthesize answers.
    lookup = {}
    tasks = ["multi_key_niah", "aggregation_sum"]
    context_lengths = [128]
    samples_per = 2
    for t in tasks:
        for L in context_lengths:
            for s in range(samples_per):
                prompt, expected = bench._build_task(t, L, seed=s)
                lookup[prompt] = (
                    " ".join(str(v) for v in expected)
                    if isinstance(expected, list)
                    else str(expected)
                )

    def oracle(prompt: str) -> str:
        return lookup[prompt]

    results = bench.evaluate(
        oracle, tasks=tasks, context_lengths=context_lengths, samples_per=samples_per
    )
    assert set(results.keys()) == set(tasks)
    for task, info in results.items():
        assert info["pass_rate"] == 1.0
        assert info["n"] == len(context_lengths) * samples_per

    per = bench.score_per_task(results)
    assert all(v == 1.0 for v in per.values())
    assert bench.overall_score(results) == 1.0
