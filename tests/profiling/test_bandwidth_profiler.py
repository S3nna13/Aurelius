import pytest

from src.profiling.bandwidth_profiler import (
    BANDWIDTH_PROFILER_REGISTRY,
    BandwidthProfiler,
    BandwidthSample,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_key():
    assert "default" in BANDWIDTH_PROFILER_REGISTRY
    assert BANDWIDTH_PROFILER_REGISTRY["default"] is BandwidthProfiler


# ---------------------------------------------------------------------------
# BandwidthSample – construction
# ---------------------------------------------------------------------------


def test_record_returns_bandwidth_sample():
    p = BandwidthProfiler()
    s = p.record(bytes_read=1024, bytes_written=512, duration_ms=1.0)
    assert isinstance(s, BandwidthSample)


def test_sample_stores_bytes_read():
    p = BandwidthProfiler()
    s = p.record(bytes_read=2048, bytes_written=0, duration_ms=1.0)
    assert s.bytes_read == 2048


def test_sample_stores_bytes_written():
    p = BandwidthProfiler()
    s = p.record(bytes_read=0, bytes_written=4096, duration_ms=1.0)
    assert s.bytes_written == 4096


def test_sample_stores_duration_ms():
    p = BandwidthProfiler()
    s = p.record(bytes_read=0, bytes_written=0, duration_ms=5.0)
    assert s.duration_ms == pytest.approx(5.0)


def test_sample_timestamp_is_positive():
    p = BandwidthProfiler()
    s = p.record(bytes_read=100, bytes_written=100, duration_ms=1.0)
    assert s.timestamp > 0


def test_sample_is_frozen():
    p = BandwidthProfiler()
    s = p.record(bytes_read=100, bytes_written=100, duration_ms=1.0)
    with pytest.raises((AttributeError, TypeError)):
        s.bytes_read = 0  # type: ignore


# ---------------------------------------------------------------------------
# BandwidthSample – bandwidth properties
# ---------------------------------------------------------------------------


def test_read_bandwidth_gbps_formula():
    # 1 GB read in 1000 ms → 1 GB/s = 1 Gbps
    s = BandwidthSample(
        timestamp=0.0, bytes_read=1_000_000_000, bytes_written=0, duration_ms=1000.0
    )
    assert s.read_bandwidth_gbps == pytest.approx(1.0, rel=1e-6)


def test_write_bandwidth_gbps_formula():
    # 2 GB written in 1000 ms → 2 Gbps
    s = BandwidthSample(
        timestamp=0.0, bytes_read=0, bytes_written=2_000_000_000, duration_ms=1000.0
    )
    assert s.write_bandwidth_gbps == pytest.approx(2.0, rel=1e-6)


def test_total_bandwidth_gbps_is_sum():
    s = BandwidthSample(
        timestamp=0.0, bytes_read=1_000_000_000, bytes_written=1_000_000_000, duration_ms=1000.0
    )
    assert s.total_bandwidth_gbps == pytest.approx(s.read_bandwidth_gbps + s.write_bandwidth_gbps)


def test_read_bandwidth_zero_duration_returns_zero():
    s = BandwidthSample(timestamp=0.0, bytes_read=1000, bytes_written=0, duration_ms=0.0)
    assert s.read_bandwidth_gbps == 0.0


def test_write_bandwidth_zero_duration_returns_zero():
    s = BandwidthSample(timestamp=0.0, bytes_read=0, bytes_written=1000, duration_ms=0.0)
    assert s.write_bandwidth_gbps == 0.0


def test_total_bandwidth_zero_duration_returns_zero():
    s = BandwidthSample(timestamp=0.0, bytes_read=1000, bytes_written=1000, duration_ms=0.0)
    assert s.total_bandwidth_gbps == 0.0


# ---------------------------------------------------------------------------
# BandwidthProfiler – utilization
# ---------------------------------------------------------------------------


def test_utilization_within_range():
    p = BandwidthProfiler(peak_bandwidth_gbps=900.0)
    s = p.record(bytes_read=1_000_000_000, bytes_written=0, duration_ms=1000.0)
    u = p.utilization(s)
    assert 0.0 <= u <= 100.0


def test_utilization_clamped_max():
    # Saturate bandwidth far above peak
    p = BandwidthProfiler(peak_bandwidth_gbps=1.0)
    s = BandwidthSample(
        timestamp=0.0, bytes_read=999_000_000_000, bytes_written=999_000_000_000, duration_ms=1.0
    )
    assert p.utilization(s) == pytest.approx(100.0)


def test_utilization_clamped_min():
    p = BandwidthProfiler(peak_bandwidth_gbps=900.0)
    s = BandwidthSample(timestamp=0.0, bytes_read=0, bytes_written=0, duration_ms=1.0)
    assert p.utilization(s) == pytest.approx(0.0)


def test_utilization_proportional():
    # 450 Gbps on a 900 Gbps peak → 50%
    p = BandwidthProfiler(peak_bandwidth_gbps=900.0)
    s = BandwidthSample(
        timestamp=0.0, bytes_read=450_000_000_000, bytes_written=0, duration_ms=1000.0
    )
    assert p.utilization(s) == pytest.approx(50.0, rel=1e-4)


# ---------------------------------------------------------------------------
# BandwidthProfiler – summary
# ---------------------------------------------------------------------------


def test_summary_keys_present():
    p = BandwidthProfiler()
    s = p.summary()
    for key in ("samples", "mean_read_gbps", "mean_write_gbps", "peak_utilization_pct"):
        assert key in s


def test_summary_empty_profiler():
    p = BandwidthProfiler()
    s = p.summary()
    assert s["samples"] == 0
    assert s["mean_read_gbps"] == 0.0
    assert s["mean_write_gbps"] == 0.0
    assert s["peak_utilization_pct"] == 0.0


def test_summary_sample_count():
    p = BandwidthProfiler()
    p.record(1_000_000, 1_000_000, 1.0)
    p.record(2_000_000, 2_000_000, 1.0)
    assert p.summary()["samples"] == 2


def test_summary_mean_read_gbps():
    p = BandwidthProfiler(peak_bandwidth_gbps=900.0)
    # Two samples each with 1 GB read in 1000 ms → mean 1 Gbps
    p.record(1_000_000_000, 0, 1000.0)
    p.record(1_000_000_000, 0, 1000.0)
    assert p.summary()["mean_read_gbps"] == pytest.approx(1.0, rel=1e-4)


def test_summary_peak_utilization_pct_is_max():
    p = BandwidthProfiler(peak_bandwidth_gbps=900.0)
    p.record(1_000_000_000, 0, 1000.0)  # ~0.11%
    p.record(90_000_000_000, 0, 1000.0)  # ~10%
    util = p.summary()["peak_utilization_pct"]
    assert util >= p.utilization(BandwidthSample(0.0, 1_000_000_000, 0, 1000.0))


# ---------------------------------------------------------------------------
# BandwidthProfiler – reset
# ---------------------------------------------------------------------------


def test_reset_clears_samples():
    p = BandwidthProfiler()
    p.record(1000, 1000, 1.0)
    p.reset()
    assert p.summary()["samples"] == 0


def test_reset_summary_mean_zero():
    p = BandwidthProfiler()
    p.record(1_000_000_000, 0, 1000.0)
    p.reset()
    s = p.summary()
    assert s["mean_read_gbps"] == 0.0
    assert s["mean_write_gbps"] == 0.0
