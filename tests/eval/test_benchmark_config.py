"""Tests for extended eval benchmark configuration."""

from src.eval.benchmark_config import (
    ALL_BENCHMARKS,
    BENCHMARK_BY_NAME,
    GPQA_DIAMOND,
    LIVECODEBENCH,
    MATH500,
)


def test_math500_benchmark_registered():
    """MATH-500 is in the registry with correct fields."""
    assert "MATH-500" in BENCHMARK_BY_NAME
    spec = BENCHMARK_BY_NAME["MATH-500"]
    assert spec.name == "MATH-500"
    assert spec.task == "math"
    assert spec.metric == "exact_match"
    assert spec.num_fewshot == 4
    assert "MATH" in spec.description or "math" in spec.description.lower()
    assert spec is MATH500


def test_gpqa_diamond_registered():
    """GPQA-Diamond is in the registry with num_fewshot=0 and diamond subset."""
    assert "GPQA-Diamond" in BENCHMARK_BY_NAME
    spec = BENCHMARK_BY_NAME["GPQA-Diamond"]
    assert spec.name == "GPQA-Diamond"
    assert spec.task == "gpqa_diamond"
    assert spec.num_fewshot == 0
    assert spec.metric == "exact_match"
    assert spec.subset == "gpqa_diamond"
    assert spec is GPQA_DIAMOND


def test_livecodebench_registered():
    """LiveCodeBench is in the registry with metric='pass@1'."""
    assert "LiveCodeBench" in BENCHMARK_BY_NAME
    spec = BENCHMARK_BY_NAME["LiveCodeBench"]
    assert spec.name == "LiveCodeBench"
    assert spec.task == "livecodebench"
    assert spec.metric == "pass@1"
    assert spec.num_fewshot == 0
    assert spec is LIVECODEBENCH


def test_no_duplicate_names():
    """All benchmark names in ALL_BENCHMARKS are unique."""
    names = [b.name for b in ALL_BENCHMARKS]
    assert len(names) == len(set(names)), f"Duplicate benchmark names found: {names}"
