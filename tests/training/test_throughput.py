import time

import pytest

from src.training.throughput import ProfileResult, ThroughputProfiler


def test_profiler_start_stop():
    profiler = ThroughputProfiler()
    profiler.start(n_tokens=1024)
    time.sleep(0.01)
    result = profiler.stop()
    assert isinstance(result, ProfileResult)
    assert result.step_time_ms > 0
    assert result.tokens_per_sec > 0


def test_profiler_tokens_per_sec():
    profiler = ThroughputProfiler()
    profiler.start(n_tokens=1000)
    time.sleep(0.1)  # ~10k tokens/sec
    result = profiler.stop()
    assert 5000 < result.tokens_per_sec < 50000  # reasonable range


def test_profiler_context_manager():
    profiler = ThroughputProfiler()
    with profiler.profile(n_tokens=512) as ctx:
        time.sleep(0.01)
    assert ctx.result is not None
    assert ctx.result.n_tokens == 512


def test_profiler_mfu_computed():
    profiler = ThroughputProfiler(
        model_params=1_000_000,
        hardware_flops_per_sec=1e12,
    )
    profiler.start(n_tokens=100)
    result = profiler.stop()
    assert result.mfu > 0


def test_profiler_mfu_zero_without_hardware_spec():
    profiler = ThroughputProfiler(model_params=1_000_000)
    profiler.start(n_tokens=100)
    result = profiler.stop()
    assert result.mfu == 0.0


def test_profiler_flops_estimate():
    profiler = ThroughputProfiler(model_params=1_000_000)
    profiler.start(n_tokens=100)
    result = profiler.stop()
    # 6 * 1M * 100 = 600M
    assert result.flops_estimate == pytest.approx(6e8)


def test_profiler_results_accumulate():
    profiler = ThroughputProfiler()
    for _ in range(3):
        with profiler.profile(n_tokens=100):
            pass
    assert len(profiler.results) == 3


def test_profiler_average_result():
    profiler = ThroughputProfiler()
    for _ in range(4):
        with profiler.profile(n_tokens=100):
            time.sleep(0.01)
    avg = profiler.average_result()
    assert avg is not None
    assert avg.step_time_ms > 0


def test_profiler_summary_string():
    profiler = ThroughputProfiler()
    profiler.start(n_tokens=1024)
    result = profiler.stop()
    s = result.summary()
    assert "tokens/sec" in s
    assert "Step time" in s
