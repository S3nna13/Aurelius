"""Tests for src/serving/rate_limiter_v2.py — advanced token-bucket rate limiter."""

import threading
import time

from src.serving.rate_limiter_v2 import (
    RATE_LIMITER_V2,
    RateLimitConfig,
    RateLimiterV2,
    TokenBucketV2,
)

# ---------------------------------------------------------------------------
# TokenBucketV2
# ---------------------------------------------------------------------------


def test_token_bucket_allows_within_burst():
    bucket = TokenBucketV2(rate=10.0, burst=5)
    # Burst is 5 — first 5 consumes should succeed.
    results = [bucket.consume() for _ in range(5)]
    assert all(results)


def test_token_bucket_denies_when_exhausted():
    bucket = TokenBucketV2(rate=1.0, burst=2)
    bucket.consume()
    bucket.consume()
    assert bucket.consume() is False


def test_token_bucket_refills_over_time():
    bucket = TokenBucketV2(rate=100.0, burst=1)
    bucket.consume()  # drain
    assert bucket.consume() is False
    time.sleep(0.02)  # wait ~2 tokens worth at 100 rps
    assert bucket.consume() is True


def test_token_bucket_tokens_available():
    bucket = TokenBucketV2(rate=10.0, burst=10)
    available = bucket.tokens_available()
    assert 0.0 <= available <= 10.0


def test_token_bucket_burst_cap():
    bucket = TokenBucketV2(rate=1000.0, burst=5)
    time.sleep(0.01)  # would generate 10 tokens at 1000 rps, but cap is 5
    assert bucket.tokens_available() <= 5.0


# ---------------------------------------------------------------------------
# RateLimiterV2 — global bucket
# ---------------------------------------------------------------------------


def test_global_allow():
    config = RateLimitConfig(requests_per_second=100.0, burst=50)
    limiter = RateLimiterV2(config)
    result = limiter.check()
    assert result.allowed is True
    assert result.reason == ""


def test_global_deny_when_burst_exhausted():
    config = RateLimitConfig(requests_per_second=1.0, burst=1)
    limiter = RateLimiterV2(config)
    # First request consumes the single burst token.
    first = limiter.check()
    assert first.allowed is True
    # Second request should be denied.
    second = limiter.check()
    assert second.allowed is False
    assert second.reason == "global_limit"
    assert second.retry_after_seconds > 0.0


# ---------------------------------------------------------------------------
# RateLimiterV2 — per-user bucket
# ---------------------------------------------------------------------------


def test_per_user_allow():
    config = RateLimitConfig(
        requests_per_second=100.0, burst=50, per_user_rps=10.0, per_user_burst=5
    )
    limiter = RateLimiterV2(config)
    result = limiter.check(user_id="alice")
    assert result.allowed is True


def test_per_user_deny_when_exhausted():
    config = RateLimitConfig(
        requests_per_second=1000.0, burst=1000, per_user_rps=1.0, per_user_burst=2
    )
    limiter = RateLimiterV2(config)
    limiter.check(user_id="bob")
    limiter.check(user_id="bob")
    result = limiter.check(user_id="bob")
    assert result.allowed is False
    assert result.reason == "user_limit"
    assert result.retry_after_seconds > 0.0


def test_per_user_isolated_buckets():
    config = RateLimitConfig(
        requests_per_second=1000.0, burst=1000, per_user_rps=1.0, per_user_burst=1
    )
    limiter = RateLimiterV2(config)
    # Exhaust alice.
    limiter.check(user_id="alice")
    assert limiter.check(user_id="alice").allowed is False
    # Bob should still be allowed.
    assert limiter.check(user_id="bob").allowed is True


# ---------------------------------------------------------------------------
# Exponential jitter backoff
# ---------------------------------------------------------------------------


def test_backoff_attempt_0_near_base():
    result = RateLimiterV2.exponential_jitter_backoff(attempt=0, base=1.0, cap=60.0, jitter=0.0)
    assert abs(result - 1.0) < 1e-9


def test_backoff_grows_with_attempt():
    def no_jitter(a):
        return RateLimiterV2.exponential_jitter_backoff(a, base=1.0, cap=1000.0, jitter=0.0)

    assert no_jitter(1) > no_jitter(0)
    assert no_jitter(2) > no_jitter(1)
    assert no_jitter(3) > no_jitter(2)


def test_backoff_capped():
    result = RateLimiterV2.exponential_jitter_backoff(attempt=100, base=1.0, cap=60.0, jitter=0.0)
    assert result <= 60.0


def test_backoff_jitter_range():
    # With jitter=0.3, result should be within [0.7*base, 1.3*base] for attempt=0.
    samples = [
        RateLimiterV2.exponential_jitter_backoff(0, base=10.0, cap=60.0, jitter=0.3)
        for _ in range(50)
    ]
    assert all(7.0 <= s <= 13.0 for s in samples)


def test_backoff_non_negative():
    result = RateLimiterV2.exponential_jitter_backoff(0, base=0.001, cap=60.0, jitter=0.9)
    assert result >= 0.0


# ---------------------------------------------------------------------------
# Thread safety (light-touch)
# ---------------------------------------------------------------------------


def test_thread_safety_global_bucket():
    config = RateLimitConfig(requests_per_second=1000.0, burst=100)
    limiter = RateLimiterV2(config)
    results = []
    lock = threading.Lock()

    def worker():
        r = limiter.check()
        with lock:
            results.append(r.allowed)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 20
    assert sum(results) <= 100  # at most burst allowed


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


def test_rate_limiter_v2_singleton_exists():
    assert RATE_LIMITER_V2 is not None
    assert isinstance(RATE_LIMITER_V2, RateLimiterV2)
