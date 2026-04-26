import math

import pytest

from src.profiling.latency_histogram import (
    LATENCY_HISTOGRAM_REGISTRY,
    HistogramBucket,
    LatencyHistogram,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_key():
    assert "default" in LATENCY_HISTOGRAM_REGISTRY
    assert LATENCY_HISTOGRAM_REGISTRY["default"] is LatencyHistogram


# ---------------------------------------------------------------------------
# Empty histogram
# ---------------------------------------------------------------------------


def test_empty_total_count():
    h = LatencyHistogram()
    assert h.total_count() == 0


def test_empty_mean_returns_zero():
    h = LatencyHistogram()
    assert h.mean() == 0.0


def test_empty_percentile_returns_zero():
    h = LatencyHistogram()
    assert h.percentile(50) == 0.0


def test_empty_percentile_p99_returns_zero():
    h = LatencyHistogram()
    assert h.percentile(99) == 0.0


def test_empty_buckets_counts_all_zero():
    h = LatencyHistogram()
    assert all(b.count == 0 for b in h.buckets())


def test_empty_to_dict_total_zero():
    h = LatencyHistogram()
    d = h.to_dict()
    assert d["total"] == 0


# ---------------------------------------------------------------------------
# Record / count
# ---------------------------------------------------------------------------


def test_record_increments_total_count():
    h = LatencyHistogram()
    h.record(10.0)
    assert h.total_count() == 1


def test_record_multiple_increments_total_count():
    h = LatencyHistogram()
    for _ in range(5):
        h.record(20.0)
    assert h.total_count() == 5


def test_record_goes_into_correct_bucket():
    h = LatencyHistogram()
    h.record(7.0)  # should land in the ≤10 bucket (index after 5)
    hit = [b for b in h.buckets() if b.count > 0]
    assert len(hit) == 1
    assert hit[0].upper == 10.0


def test_record_inf_bucket():
    h = LatencyHistogram()
    h.record(9999.0)  # exceeds 1000, lands in +inf bucket
    hit = [b for b in h.buckets() if b.count > 0]
    assert len(hit) == 1
    assert math.isinf(hit[0].upper)


# ---------------------------------------------------------------------------
# Mean
# ---------------------------------------------------------------------------


def test_mean_single_sample():
    h = LatencyHistogram()
    h.record(42.0)
    assert h.mean() == pytest.approx(42.0)


def test_mean_multiple_samples():
    h = LatencyHistogram()
    for v in [10.0, 20.0, 30.0]:
        h.record(v)
    assert h.mean() == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Percentile
# ---------------------------------------------------------------------------


def test_percentile_p50_single_sample():
    h = LatencyHistogram()
    h.record(50.0)
    p50 = h.percentile(50)
    assert p50 >= 0.0


def test_percentile_p50_returns_nonnegative():
    h = LatencyHistogram()
    for v in [10, 20, 30, 40, 50]:
        h.record(float(v))
    assert h.percentile(50) >= 0.0


def test_percentile_p95_gte_p50():
    h = LatencyHistogram()
    for v in range(1, 101):
        h.record(float(v))
    assert h.percentile(95) >= h.percentile(50)


def test_percentile_p99_gte_p95():
    h = LatencyHistogram()
    for v in range(1, 101):
        h.record(float(v))
    assert h.percentile(99) >= h.percentile(95)


def test_percentile_p0_nonnegative():
    h = LatencyHistogram()
    h.record(5.0)
    assert h.percentile(0) >= 0.0


# ---------------------------------------------------------------------------
# Bucket list format
# ---------------------------------------------------------------------------


def test_buckets_returns_histogram_bucket_instances():
    h = LatencyHistogram()
    for b in h.buckets():
        assert isinstance(b, HistogramBucket)


def test_buckets_are_sorted_by_upper():
    h = LatencyHistogram()
    bkts = h.buckets()
    uppers = [b.upper for b in bkts if not math.isinf(b.upper)]
    assert uppers == sorted(uppers)


def test_buckets_lower_matches_previous_upper():
    h = LatencyHistogram()
    bkts = h.buckets()
    for i in range(1, len(bkts)):
        assert bkts[i].lower == bkts[i - 1].upper


def test_buckets_frozen():
    h = LatencyHistogram()
    b = h.buckets()[0]
    with pytest.raises((AttributeError, TypeError)):
        b.count = 99  # type: ignore


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_total_count():
    h = LatencyHistogram()
    h.record(10.0)
    h.reset()
    assert h.total_count() == 0


def test_reset_clears_mean():
    h = LatencyHistogram()
    h.record(50.0)
    h.reset()
    assert h.mean() == 0.0


def test_reset_clears_bucket_counts():
    h = LatencyHistogram()
    h.record(10.0)
    h.reset()
    assert all(b.count == 0 for b in h.buckets())


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_has_required_keys():
    h = LatencyHistogram()
    d = h.to_dict()
    for key in ("total", "mean_ms", "p50", "p95", "p99", "buckets"):
        assert key in d


def test_to_dict_buckets_is_list():
    h = LatencyHistogram()
    d = h.to_dict()
    assert isinstance(d["buckets"], list)


def test_to_dict_bucket_entry_has_keys():
    h = LatencyHistogram()
    d = h.to_dict()
    for entry in d["buckets"]:
        assert "lower" in entry
        assert "upper" in entry
        assert "count" in entry


def test_to_dict_reflects_recorded_total():
    h = LatencyHistogram()
    h.record(5.0)
    h.record(15.0)
    assert h.to_dict()["total"] == 2


def test_to_dict_mean_ms_correct():
    h = LatencyHistogram()
    h.record(10.0)
    h.record(30.0)
    assert h.to_dict()["mean_ms"] == pytest.approx(20.0)
