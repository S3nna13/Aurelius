"""Tests for src/quantization/quant_benchmark.py."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.quantization.awq_quantizer import AWQConfig
from src.quantization.gptq_quantizer import GPTQConfig
from src.quantization.quant_benchmark import (
    BenchmarkResult,
    QUANT_BENCHMARK_REGISTRY,
    QuantBenchmark,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def weight() -> "torch.Tensor":
    torch.manual_seed(7)
    return torch.randn(8, 16)


@pytest.fixture()
def activations() -> "torch.Tensor":
    torch.manual_seed(8)
    return torch.randn(32, 16)


@pytest.fixture()
def bench() -> QuantBenchmark:
    return QuantBenchmark()


# ---------------------------------------------------------------------------
# BenchmarkResult container
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    def test_frozen(self):
        r = BenchmarkResult("gptq", 4, 128, 0.01, 8.0, 1.0)
        with pytest.raises(Exception):
            r.mse = 0.0  # type: ignore[misc]

    def test_fields(self):
        r = BenchmarkResult("gptq", 4, 128, 0.01, 8.0, 1.0)
        assert r.method == "gptq"
        assert r.bits == 4
        assert r.group_size == 128
        assert r.mse == pytest.approx(0.01)
        assert r.compression_ratio == pytest.approx(8.0)
        assert r.quantize_time_ms == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# run_gptq
# ---------------------------------------------------------------------------

class TestRunGPTQ:
    def test_returns_benchmark_result(self, bench, weight):
        r = bench.run_gptq(weight)
        assert isinstance(r, BenchmarkResult)

    def test_method_name(self, bench, weight):
        assert bench.run_gptq(weight).method == "gptq"

    def test_default_bits(self, bench, weight):
        assert bench.run_gptq(weight).bits == 4

    def test_custom_config(self, bench, weight):
        r = bench.run_gptq(weight, GPTQConfig(bits=8))
        assert r.bits == 8

    def test_mse_finite_nonnegative(self, bench, weight):
        r = bench.run_gptq(weight)
        assert r.mse >= 0.0
        assert r.mse == r.mse  # not NaN

    def test_time_positive(self, bench, weight):
        assert bench.run_gptq(weight).quantize_time_ms >= 0.0

    def test_compression_ratio_4bit(self, bench, weight):
        assert bench.run_gptq(weight).compression_ratio == pytest.approx(8.0)

    def test_compression_ratio_8bit(self, bench, weight):
        r = bench.run_gptq(weight, GPTQConfig(bits=8))
        assert r.compression_ratio == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# run_awq
# ---------------------------------------------------------------------------

class TestRunAWQ:
    def test_returns_benchmark_result(self, bench, weight, activations):
        r = bench.run_awq(weight, activations)
        assert isinstance(r, BenchmarkResult)

    def test_method_name(self, bench, weight, activations):
        assert bench.run_awq(weight, activations).method == "awq"

    def test_default_bits(self, bench, weight, activations):
        assert bench.run_awq(weight, activations).bits == 4

    def test_custom_config(self, bench, weight, activations):
        r = bench.run_awq(weight, activations, AWQConfig(bits=8))
        assert r.bits == 8

    def test_mse_finite_nonnegative(self, bench, weight, activations):
        r = bench.run_awq(weight, activations)
        assert r.mse >= 0.0

    def test_time_nonnegative(self, bench, weight, activations):
        assert bench.run_awq(weight, activations).quantize_time_ms >= 0.0

    def test_compression_ratio(self, bench, weight, activations):
        r = bench.run_awq(weight, activations)
        assert r.compression_ratio == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

class TestCompare:
    def test_keys(self, bench, weight, activations):
        results = [bench.run_gptq(weight), bench.run_awq(weight, activations)]
        cmp = bench.compare(results)
        assert set(cmp.keys()) == {"best_mse", "best_compression", "summary"}

    def test_summary_sorted_by_mse(self, bench):
        rs = [
            BenchmarkResult("a", 4, 128, 0.5, 8.0, 1.0),
            BenchmarkResult("b", 4, 128, 0.1, 8.0, 1.0),
            BenchmarkResult("c", 4, 128, 0.3, 8.0, 1.0),
        ]
        cmp = bench.compare(rs)
        mses = [r.mse for r in cmp["summary"]]
        assert mses == sorted(mses)

    def test_best_mse(self, bench):
        rs = [
            BenchmarkResult("a", 4, 128, 0.5, 8.0, 1.0),
            BenchmarkResult("b", 4, 128, 0.1, 8.0, 1.0),
        ]
        assert bench.compare(rs)["best_mse"].method == "b"

    def test_best_compression(self, bench):
        rs = [
            BenchmarkResult("a", 8, 128, 0.1, 4.0, 1.0),
            BenchmarkResult("b", 4, 128, 0.2, 8.0, 1.0),
        ]
        assert bench.compare(rs)["best_compression"].method == "b"

    def test_empty(self, bench):
        cmp = bench.compare([])
        assert cmp["best_mse"] is None
        assert cmp["best_compression"] is None
        assert cmp["summary"] == []


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

class TestReport:
    def test_returns_string(self, bench):
        s = bench.report([BenchmarkResult("gptq", 4, 128, 0.01, 8.0, 1.0)])
        assert isinstance(s, str)

    def test_contains_method(self, bench):
        s = bench.report([BenchmarkResult("gptq", 4, 128, 0.01, 8.0, 1.0)])
        assert "gptq" in s

    def test_contains_header_columns(self, bench):
        s = bench.report([])
        for col in ("method", "bits", "mse", "compression"):
            assert col in s

    def test_empty_message(self, bench):
        s = bench.report([])
        assert "no results" in s

    def test_multiple_rows(self, bench):
        rs = [
            BenchmarkResult("gptq", 4, 128, 0.01, 8.0, 1.0),
            BenchmarkResult("awq", 4, 128, 0.02, 8.0, 1.5),
        ]
        s = bench.report(rs)
        assert "gptq" in s and "awq" in s
        assert s.count("\n") >= 3  # header, separator, 2 rows

    def test_formatted_numbers(self, bench):
        s = bench.report([BenchmarkResult("gptq", 4, 128, 0.012345, 8.0, 1.234)])
        # MSE formatted to 6 decimal places somewhere
        assert "0.012345" in s


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_default_key(self):
        assert "default" in QUANT_BENCHMARK_REGISTRY

    def test_default_cls(self):
        assert QUANT_BENCHMARK_REGISTRY["default"] is QuantBenchmark


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline(self, bench, weight, activations):
        results = [
            bench.run_gptq(weight, GPTQConfig(bits=4)),
            bench.run_gptq(weight, GPTQConfig(bits=8)),
            bench.run_awq(weight, activations, AWQConfig(bits=4)),
        ]
        cmp = bench.compare(results)
        assert cmp["best_mse"] is not None
        report = bench.report(results)
        assert isinstance(report, str)
        assert len(report) > 0
