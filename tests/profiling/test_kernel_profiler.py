import pytest

from src.profiling.kernel_profiler import (
    KERNEL_PROFILER_REGISTRY,
    KernelProfiler,
    KernelRecord,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_key():
    assert "default" in KERNEL_PROFILER_REGISTRY
    assert KERNEL_PROFILER_REGISTRY["default"] is KernelProfiler


# ---------------------------------------------------------------------------
# KernelRecord – construction and frozen
# ---------------------------------------------------------------------------


def test_kernel_record_is_frozen():
    r = KernelRecord(
        kernel_name="matmul",
        grid_size=(16, 16),
        block_size=(32, 32),
        duration_us=100.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        r.duration_us = 0.0  # type: ignore


def test_kernel_record_default_memory_bytes():
    r = KernelRecord("k", (1,), (1,), 1.0)
    assert r.memory_bytes == 0


def test_kernel_record_stores_fields():
    r = KernelRecord("conv2d", (8, 8, 1), (16, 16, 1), 250.0, memory_bytes=4096)
    assert r.kernel_name == "conv2d"
    assert r.grid_size == (8, 8, 1)
    assert r.block_size == (16, 16, 1)
    assert r.duration_us == pytest.approx(250.0)
    assert r.memory_bytes == 4096


# ---------------------------------------------------------------------------
# KernelRecord – throughput_gflops
# ---------------------------------------------------------------------------


def test_throughput_gflops_formula():
    # 1e9 flops in 1 us → 1e9 / (1e-6 s) / 1e9 = 1_000_000 Gflops (= 1 Pflops)
    r = KernelRecord("k", (1,), (1,), duration_us=1.0)
    assert r.throughput_gflops(1_000_000_000) == pytest.approx(1_000_000.0, rel=1e-6)


def test_throughput_gflops_zero_duration():
    r = KernelRecord("k", (1,), (1,), duration_us=0.0)
    assert r.throughput_gflops(1_000_000) == 0.0


def test_throughput_gflops_scales_with_flops():
    r = KernelRecord("k", (1,), (1,), duration_us=1000.0)
    result_double = r.throughput_gflops(2_000_000_000)
    result_single = r.throughput_gflops(1_000_000_000)
    assert result_double == pytest.approx(result_single * 2, rel=1e-6)


# ---------------------------------------------------------------------------
# KernelProfiler – record_kernel
# ---------------------------------------------------------------------------


def test_record_kernel_returns_kernel_record():
    p = KernelProfiler()
    r = p.record_kernel("gemm", (32,), (256,), 500.0)
    assert isinstance(r, KernelRecord)


def test_record_kernel_stores_name():
    p = KernelProfiler()
    r = p.record_kernel("softmax", (1,), (1,), 10.0)
    assert r.kernel_name == "softmax"


def test_record_kernel_stores_duration():
    p = KernelProfiler()
    r = p.record_kernel("relu", (4,), (128,), 77.5)
    assert r.duration_us == pytest.approx(77.5)


def test_record_kernel_stores_memory_bytes():
    p = KernelProfiler()
    r = p.record_kernel("memcpy", (1,), (1,), 50.0, memory_bytes=8192)
    assert r.memory_bytes == 8192


def test_record_kernel_grid_and_block_tuples():
    p = KernelProfiler()
    r = p.record_kernel("k", (2, 3), (4, 5), 1.0)
    assert r.grid_size == (2, 3)
    assert r.block_size == (4, 5)


# ---------------------------------------------------------------------------
# KernelProfiler – top_kernels
# ---------------------------------------------------------------------------


def test_top_kernels_sorted_descending():
    p = KernelProfiler()
    p.record_kernel("fast", (1,), (1,), 10.0)
    p.record_kernel("slow", (1,), (1,), 500.0)
    p.record_kernel("medium", (1,), (1,), 100.0)
    top = p.top_kernels(3)
    durations = [r.duration_us for r in top]
    assert durations == sorted(durations, reverse=True)


def test_top_kernels_default_n_ten():
    p = KernelProfiler()
    for i in range(15):
        p.record_kernel(f"k{i}", (1,), (1,), float(i))
    assert len(p.top_kernels()) == 10


def test_top_kernels_n_greater_than_available():
    p = KernelProfiler()
    p.record_kernel("a", (1,), (1,), 1.0)
    p.record_kernel("b", (1,), (1,), 2.0)
    top = p.top_kernels(n=50)
    assert len(top) == 2


def test_top_kernels_n_one():
    p = KernelProfiler()
    p.record_kernel("slow", (1,), (1,), 999.0)
    p.record_kernel("fast", (1,), (1,), 1.0)
    top = p.top_kernels(n=1)
    assert len(top) == 1
    assert top[0].duration_us == pytest.approx(999.0)


def test_top_kernels_empty():
    p = KernelProfiler()
    assert p.top_kernels() == []


# ---------------------------------------------------------------------------
# KernelProfiler – summary_by_name
# ---------------------------------------------------------------------------


def test_summary_by_name_single_kernel():
    p = KernelProfiler()
    p.record_kernel("gemm", (1,), (1,), 100.0)
    s = p.summary_by_name()
    assert "gemm" in s
    assert s["gemm"]["count"] == 1
    assert s["gemm"]["total_us"] == pytest.approx(100.0)
    assert s["gemm"]["mean_us"] == pytest.approx(100.0)


def test_summary_by_name_groups_same_name():
    p = KernelProfiler()
    p.record_kernel("layernorm", (1,), (1,), 50.0)
    p.record_kernel("layernorm", (1,), (1,), 150.0)
    s = p.summary_by_name()
    assert s["layernorm"]["count"] == 2
    assert s["layernorm"]["total_us"] == pytest.approx(200.0)
    assert s["layernorm"]["mean_us"] == pytest.approx(100.0)


def test_summary_by_name_multiple_kernels():
    p = KernelProfiler()
    p.record_kernel("relu", (1,), (1,), 10.0)
    p.record_kernel("softmax", (1,), (1,), 20.0)
    s = p.summary_by_name()
    assert "relu" in s
    assert "softmax" in s


def test_summary_by_name_count_accuracy():
    p = KernelProfiler()
    for _ in range(7):
        p.record_kernel("attn", (1,), (1,), 30.0)
    assert p.summary_by_name()["attn"]["count"] == 7


def test_summary_by_name_empty():
    p = KernelProfiler()
    assert p.summary_by_name() == {}


# ---------------------------------------------------------------------------
# KernelProfiler – total_time_us
# ---------------------------------------------------------------------------


def test_total_time_us_sum():
    p = KernelProfiler()
    p.record_kernel("a", (1,), (1,), 100.0)
    p.record_kernel("b", (1,), (1,), 200.0)
    p.record_kernel("c", (1,), (1,), 300.0)
    assert p.total_time_us() == pytest.approx(600.0)


def test_total_time_us_empty():
    p = KernelProfiler()
    assert p.total_time_us() == 0.0


def test_total_time_us_single():
    p = KernelProfiler()
    p.record_kernel("only", (1,), (1,), 42.5)
    assert p.total_time_us() == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# KernelProfiler – reset
# ---------------------------------------------------------------------------


def test_reset_clears_records():
    p = KernelProfiler()
    p.record_kernel("k", (1,), (1,), 100.0)
    p.reset()
    assert p.total_time_us() == 0.0


def test_reset_top_kernels_empty():
    p = KernelProfiler()
    p.record_kernel("k", (1,), (1,), 100.0)
    p.reset()
    assert p.top_kernels() == []


def test_reset_summary_empty():
    p = KernelProfiler()
    p.record_kernel("k", (1,), (1,), 100.0)
    p.reset()
    assert p.summary_by_name() == {}


def test_reset_allows_reuse():
    p = KernelProfiler()
    p.record_kernel("k", (1,), (1,), 100.0)
    p.reset()
    p.record_kernel("k2", (1,), (1,), 50.0)
    assert p.total_time_us() == pytest.approx(50.0)
