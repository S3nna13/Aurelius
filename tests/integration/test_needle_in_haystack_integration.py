"""Integration tests for NIAH through the src.eval package surface."""

from __future__ import annotations

import src.eval as eval_pkg
from src.eval import NeedleInHaystackBenchmark


def _oracle(prompt: str) -> str:
    return "42" if "The magic number is 42." in prompt else "unknown"


def test_niah_via_package_import_runs_and_scores_perfect():
    bench = NeedleInHaystackBenchmark()
    results = bench.evaluate(_oracle, context_lengths=[64, 128], depths=[0.0, 0.5])
    assert len(results) == 4
    assert bench.score(results) == 1.0


def test_registry_entries_present_and_preexisting_benchmarks_untouched():
    # NIAH registered.
    assert "niah" in eval_pkg.METRIC_REGISTRY
    assert "niah" in eval_pkg.BENCHMARK_REGISTRY
    assert eval_pkg.METRIC_REGISTRY["niah"] is NeedleInHaystackBenchmark

    # Preexisting package-level symbols from benchmark_config remain intact.
    assert hasattr(eval_pkg, "BENCHMARK_BY_NAME")
    assert hasattr(eval_pkg, "ALL_BENCHMARKS")
    assert "mmlu" in eval_pkg.BENCHMARK_BY_NAME or "MMLU" in eval_pkg.BENCHMARK_BY_NAME \
        or len(eval_pkg.BENCHMARK_BY_NAME) > 0
    # ALL_BENCHMARKS should still be non-empty.
    assert len(eval_pkg.ALL_BENCHMARKS) > 0


def test_niah_idempotent_registration():
    # Re-importing should not mutate or duplicate the registry entry.
    before = eval_pkg.BENCHMARK_REGISTRY["niah"]
    import importlib
    importlib.reload(eval_pkg)
    after = eval_pkg.BENCHMARK_REGISTRY["niah"]
    assert before is after or before.__name__ == after.__name__
