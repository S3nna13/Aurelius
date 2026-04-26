"""Tests for the token-bucket rate limiter (STRIDE-DoS defense)."""

from __future__ import annotations

import threading
import time

from src.serving.rate_limiter import (
    DEFAULT_RATE_LIMITER,
    RATE_LIMIT_REGISTRY,
    RateLimitConfig,
    RateLimiterChain,
    RateLimitResult,
    TokenBucketLimiter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_limiter(rps: float = 10.0, burst: int = 5, per: str = "key") -> TokenBucketLimiter:
    return TokenBucketLimiter(RateLimitConfig(requests_per_second=rps, burst_size=burst, per=per))


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestSingleRequest:
    def test_fresh_bucket_allowed(self):
        limiter = make_limiter(rps=10.0, burst=5)
        result = limiter.check("user-1")
        assert result.allowed is True
        assert result.remaining == 4  # burst_size - 1
        assert result.retry_after_s == 0.0
        assert result.limit == 10.0

    def test_result_is_rate_limit_result(self):
        limiter = make_limiter()
        result = limiter.check("any")
        assert isinstance(result, RateLimitResult)


class TestBurstExhaustion:
    def test_exhaust_burst_last_denied(self):
        burst = 5
        limiter = make_limiter(rps=10.0, burst=burst)
        results = [limiter.check("key") for _ in range(burst + 1)]
        # First `burst` requests should be allowed
        for r in results[:burst]:
            assert r.allowed is True
        # The (burst+1)-th request should be denied
        assert results[burst].allowed is False
        assert results[burst].remaining == 0
        assert results[burst].retry_after_s > 0.0

    def test_denied_has_positive_retry_after(self):
        limiter = make_limiter(rps=5.0, burst=2)
        limiter.check("k")
        limiter.check("k")
        result = limiter.check("k")
        assert result.allowed is False
        assert result.retry_after_s > 0.0


class TestRefill:
    def test_refill_after_sleep(self):
        rps = 10.0
        limiter = make_limiter(rps=rps, burst=1)
        # Exhaust the single token
        r1 = limiter.check("refill-key")
        assert r1.allowed is True
        r2 = limiter.check("refill-key")
        assert r2.allowed is False
        # Wait long enough for at least one token to refill (1/rps seconds)
        time.sleep(1.0 / rps + 0.05)
        r3 = limiter.check("refill-key")
        assert r3.allowed is True


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_bucket(self):
        burst = 3
        limiter = make_limiter(rps=10.0, burst=burst)
        for _ in range(burst):
            limiter.check("reset-key")
        exhausted = limiter.check("reset-key")
        assert exhausted.allowed is False

        limiter.reset("reset-key")
        after_reset = limiter.check("reset-key")
        assert after_reset.allowed is True

    def test_reset_all_clears_buckets(self):
        limiter = make_limiter(rps=10.0, burst=2)
        for _ in range(3):
            limiter.check("k1")
        for _ in range(3):
            limiter.check("k2")
        limiter.reset_all()
        assert limiter.check("k1").allowed is True
        assert limiter.check("k2").allowed is True

    def test_reset_unknown_key_no_error(self):
        limiter = make_limiter()
        # Should not raise even if bucket doesn't exist yet
        limiter.reset("nonexistent-key")


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_requests_respect_burst(self):
        burst = 10
        limiter = make_limiter(rps=100.0, burst=burst)
        identifier = "shared-key"
        allowed_count = 0
        lock = threading.Lock()

        def worker():
            nonlocal allowed_count
            for _ in range(10):
                result = limiter.check(identifier)
                if result.allowed:
                    with lock:
                        allowed_count += 1

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total allowed must not exceed the burst size
        assert allowed_count <= burst


# ---------------------------------------------------------------------------
# RateLimiterChain
# ---------------------------------------------------------------------------


class TestRateLimiterChain:
    def test_chain_all_allow(self):
        key_limiter = make_limiter(rps=100.0, burst=50, per="key")
        ip_limiter = make_limiter(rps=100.0, burst=50, per="ip")
        chain = RateLimiterChain([("key", key_limiter), ("ip", ip_limiter)])
        result = chain.check_all(key="user-1", ip="1.2.3.4", route="/chat")
        assert result.allowed is True

    def test_chain_key_allows_ip_denies(self):
        # key limiter has plenty of capacity
        key_limiter = make_limiter(rps=100.0, burst=100, per="key")
        # ip limiter is exhausted
        ip_limiter = make_limiter(rps=10.0, burst=2, per="ip")
        # Exhaust the IP bucket
        ip_limiter.check("1.2.3.4")
        ip_limiter.check("1.2.3.4")

        chain = RateLimiterChain([("key", key_limiter), ("ip", ip_limiter)])
        result = chain.check_all(key="user-1", ip="1.2.3.4", route="/chat")
        assert result.allowed is False

    def test_chain_ip_allows_key_denies(self):
        key_limiter = make_limiter(rps=10.0, burst=2, per="key")
        # Exhaust key bucket
        key_limiter.check("user-bad")
        key_limiter.check("user-bad")

        ip_limiter = make_limiter(rps=100.0, burst=100, per="ip")
        chain = RateLimiterChain([("key", key_limiter), ("ip", ip_limiter)])
        result = chain.check_all(key="user-bad", ip="9.9.9.9", route="/chat")
        assert result.allowed is False

    def test_chain_route_identifier_routing(self):
        route_limiter = make_limiter(rps=10.0, burst=2, per="route")
        route_limiter.check("/admin")
        route_limiter.check("/admin")

        chain = RateLimiterChain([("route", route_limiter)])
        result = chain.check_all(key="any", ip="1.1.1.1", route="/admin")
        assert result.allowed is False

        # A different route should be unaffected
        result2 = chain.check_all(key="any", ip="1.1.1.1", route="/health")
        assert result2.allowed is True


# ---------------------------------------------------------------------------
# Adversarial inputs
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_empty_string_identifier(self):
        limiter = make_limiter()
        result = limiter.check("")
        assert isinstance(result, RateLimitResult)
        assert result.allowed is True

    def test_very_long_identifier(self):
        limiter = make_limiter()
        long_id = "x" * 10_000
        result = limiter.check(long_id)
        assert isinstance(result, RateLimitResult)
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Global per-mode
# ---------------------------------------------------------------------------


class TestGlobalPer:
    def test_global_same_bucket_regardless_of_identifier(self):
        burst = 3
        limiter = make_limiter(rps=10.0, burst=burst, per="global")
        # Different identifiers share one bucket
        limiter.check("alice")
        limiter.check("bob")
        limiter.check("charlie")
        # The bucket should now be exhausted regardless of caller
        result = limiter.check("dave")
        assert result.allowed is False

    def test_global_reset_via_any_identifier(self):
        burst = 2
        limiter = make_limiter(rps=10.0, burst=burst, per="global")
        limiter.check("a")
        limiter.check("b")
        assert limiter.check("c").allowed is False
        # Reset using any identifier — internally resolves to __global__
        limiter.reset("whatever")
        assert limiter.check("z").allowed is True


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_default_limiter_in_registry(self):
        assert "default" in RATE_LIMIT_REGISTRY
        assert RATE_LIMIT_REGISTRY["default"] is DEFAULT_RATE_LIMITER

    def test_default_limiter_config(self):
        assert DEFAULT_RATE_LIMITER.config.requests_per_second == 100.0
        assert DEFAULT_RATE_LIMITER.config.burst_size == 200
