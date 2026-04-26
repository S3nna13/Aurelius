"""Tests for src/serving/load_shedder.py"""

from src.serving.load_shedder import (
    SERVING_REGISTRY,
    LoadShedder,
    ShedPolicy,
)

# ---------------------------------------------------------------------------
# DROP_TAIL
# ---------------------------------------------------------------------------


def test_drop_tail_below_max_no_shed():
    ls = LoadShedder(max_depth=100)
    assert not ls.should_shed(50, 0.0, ShedPolicy.DROP_TAIL)


def test_drop_tail_at_max_no_shed():
    ls = LoadShedder(max_depth=100)
    assert not ls.should_shed(100, 0.0, ShedPolicy.DROP_TAIL)


def test_drop_tail_above_max_sheds():
    ls = LoadShedder(max_depth=100)
    assert ls.should_shed(101, 0.0, ShedPolicy.DROP_TAIL)


# ---------------------------------------------------------------------------
# PRIORITY_QUEUE (placeholder — never shed)
# ---------------------------------------------------------------------------


def test_priority_queue_never_sheds():
    ls = LoadShedder()
    for depth in [0, 1000, 9999]:
        assert not ls.should_shed(depth, 9999.0, ShedPolicy.PRIORITY_QUEUE)


# ---------------------------------------------------------------------------
# TOKEN_BUCKET
# ---------------------------------------------------------------------------


def test_token_bucket_allows_when_tokens_available():
    ls = LoadShedder(token_rate=1000.0, bucket_capacity=10.0)
    # Should have 10 tokens initially — first 10 calls should pass
    results = [ls.should_shed(0, 0.0, ShedPolicy.TOKEN_BUCKET) for _ in range(10)]
    assert all(not r for r in results)


def test_token_bucket_sheds_when_exhausted():
    ls = LoadShedder(token_rate=0.0, bucket_capacity=2.0)
    ls.should_shed(0, 0.0, ShedPolicy.TOKEN_BUCKET)
    ls.should_shed(0, 0.0, ShedPolicy.TOKEN_BUCKET)
    # Tokens exhausted, rate=0 so no refill
    assert ls.should_shed(0, 0.0, ShedPolicy.TOKEN_BUCKET)


# ---------------------------------------------------------------------------
# ADAPTIVE
# ---------------------------------------------------------------------------


def test_adaptive_no_shed_with_no_samples():
    ls = LoadShedder(p99_threshold_ms=2000.0)
    # p99=0 when no samples — should not shed
    assert not ls.should_shed(0, 0.0, ShedPolicy.ADAPTIVE)


def test_adaptive_sheds_when_p99_exceeds_threshold():
    ls = LoadShedder(p99_threshold_ms=100.0)
    for _ in range(200):
        ls.record_latency(500.0)  # all samples are 500ms
    assert ls.should_shed(0, 500.0, ShedPolicy.ADAPTIVE)


def test_adaptive_no_shed_when_p99_below_threshold():
    ls = LoadShedder(p99_threshold_ms=2000.0)
    for _ in range(200):
        ls.record_latency(50.0)
    assert not ls.should_shed(0, 50.0, ShedPolicy.ADAPTIVE)


# ---------------------------------------------------------------------------
# record_latency + p99
# ---------------------------------------------------------------------------


def test_record_latency_affects_stats():
    ls = LoadShedder()
    for ms in [10.0, 20.0, 30.0]:
        ls.record_latency(ms)
    stats = ls.get_stats()
    assert stats["p99_latency"] > 0.0


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


def test_get_stats_keys_present():
    ls = LoadShedder()
    stats = ls.get_stats()
    assert "queue_depth" in stats
    assert "p99_latency" in stats
    assert "shed_count" in stats


def test_shed_count_increments():
    ls = LoadShedder(max_depth=5)
    ls.should_shed(10, 0.0, ShedPolicy.DROP_TAIL)
    ls.should_shed(10, 0.0, ShedPolicy.DROP_TAIL)
    assert ls.get_stats()["shed_count"] == 2


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_serving_registry_contains_load_shedder():
    assert "load_shedder" in SERVING_REGISTRY
    assert isinstance(SERVING_REGISTRY["load_shedder"], LoadShedder)
