"""Tests for src/protocol/rate_limiting_middleware.py"""

import pytest

from src.protocol.rate_limiting_middleware import (
    RATE_LIMIT_MIDDLEWARE_REGISTRY,
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitingMiddleware,
    RateLimitResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tb_mw(**kw):
    """Return a TOKEN_BUCKET middleware with convenient overrides."""
    cfg = RateLimitConfig(
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        rate=kw.get("rate", 100.0),
        burst=kw.get("burst", 5),
        window_s=kw.get("window_s", 1.0),
    )
    return RateLimitingMiddleware(cfg)


def _fw_mw(**kw):
    cfg = RateLimitConfig(
        algorithm=RateLimitAlgorithm.FIXED_WINDOW,
        rate=kw.get("rate", 100.0),
        burst=kw.get("burst", 3),
        window_s=kw.get("window_s", 1.0),
    )
    return RateLimitingMiddleware(cfg)


def _sl_mw(**kw):
    cfg = RateLimitConfig(
        algorithm=RateLimitAlgorithm.SLIDING_LOG,
        rate=kw.get("rate", 100.0),
        burst=kw.get("burst", 3),
        window_s=kw.get("window_s", 1.0),
    )
    return RateLimitingMiddleware(cfg)


# ---------------------------------------------------------------------------
# TOKEN_BUCKET – basic allow
# ---------------------------------------------------------------------------


def test_token_bucket_first_call_allowed():
    mw = _tb_mw(burst=5)
    result = mw.check("c1", now=0.0)
    assert result.allowed is True


def test_token_bucket_first_call_remaining():
    mw = _tb_mw(burst=5)
    result = mw.check("c1", now=0.0)
    # After 1 token consumed, 4 remain
    assert result.remaining == 4


def test_token_bucket_client_id_in_result():
    mw = _tb_mw()
    result = mw.check("alice", now=0.0)
    assert result.client_id == "alice"


# ---------------------------------------------------------------------------
# TOKEN_BUCKET – burst exhaustion
# ---------------------------------------------------------------------------


def test_token_bucket_burst_exhausted_denied():
    mw = _tb_mw(burst=3, rate=0.0)
    mw.check("c", now=0.0)
    mw.check("c", now=0.0)
    mw.check("c", now=0.0)
    result = mw.check("c", now=0.0)  # 4th call – denied
    assert result.allowed is False


def test_token_bucket_burst_exhausted_remaining_zero():
    mw = _tb_mw(burst=2, rate=0.0)
    mw.check("c", now=0.0)
    mw.check("c", now=0.0)
    result = mw.check("c", now=0.0)
    assert result.remaining == 0


def test_token_bucket_exactly_burst_calls_allowed():
    mw = _tb_mw(burst=4, rate=0.0)
    results = [mw.check("c", now=0.0) for _ in range(4)]
    assert all(r.allowed for r in results)


# ---------------------------------------------------------------------------
# TOKEN_BUCKET – refill over time
# ---------------------------------------------------------------------------


def test_token_bucket_refill_allows_after_wait():
    mw = _tb_mw(burst=1, rate=2.0)  # 2 tokens/sec
    mw.check("c", now=0.0)  # consume last token
    result = mw.check("c", now=1.0)  # 2 tokens refilled → allowed
    assert result.allowed is True


def test_token_bucket_partial_refill():
    mw = _tb_mw(burst=10, rate=1.0)
    for _ in range(10):
        mw.check("c", now=0.0)  # exhaust
    result = mw.check("c", now=5.0)  # 5 tokens refilled
    assert result.allowed is True
    assert result.remaining == 4


def test_token_bucket_refill_capped_at_burst():
    mw = _tb_mw(burst=3, rate=100.0)
    # Wait a very long time → should not exceed burst
    result = mw.check("c", now=1000.0)
    assert result.remaining <= 2  # one consumed, cap at burst


# ---------------------------------------------------------------------------
# FIXED_WINDOW – within window
# ---------------------------------------------------------------------------


def test_fixed_window_allows_within_burst():
    mw = _fw_mw(burst=3, window_s=1.0)
    for _ in range(3):
        r = mw.check("c", now=0.0)
        assert r.allowed is True


def test_fixed_window_denies_over_burst():
    mw = _fw_mw(burst=2, window_s=1.0)
    mw.check("c", now=0.0)
    mw.check("c", now=0.0)
    result = mw.check("c", now=0.0)
    assert result.allowed is False


def test_fixed_window_reset_at_new_window():
    mw = _fw_mw(burst=2, window_s=1.0)
    mw.check("c", now=0.0)
    mw.check("c", now=0.0)
    # New window starts at t=1.0
    result = mw.check("c", now=1.0)
    assert result.allowed is True


def test_fixed_window_reset_after_s_positive():
    mw = _fw_mw(burst=5, window_s=2.0)
    result = mw.check("c", now=0.5)
    assert result.reset_after_s > 0.0


def test_fixed_window_client_id_propagated():
    mw = _fw_mw()
    r = mw.check("bob", now=0.0)
    assert r.client_id == "bob"


# ---------------------------------------------------------------------------
# SLIDING_LOG – removes old entries
# ---------------------------------------------------------------------------


def test_sliding_log_allows_within_burst():
    mw = _sl_mw(burst=3, window_s=1.0)
    for _ in range(3):
        r = mw.check("c", now=0.0)
        assert r.allowed is True


def test_sliding_log_denies_over_burst():
    mw = _sl_mw(burst=2, window_s=1.0)
    mw.check("c", now=0.0)
    mw.check("c", now=0.0)
    result = mw.check("c", now=0.5)
    assert result.allowed is False


def test_sliding_log_removes_old_entries():
    mw = _sl_mw(burst=2, window_s=1.0)
    mw.check("c", now=0.0)
    mw.check("c", now=0.0)
    # Both old entries are outside the window now
    result = mw.check("c", now=1.5)
    assert result.allowed is True


def test_sliding_log_partial_eviction():
    mw = _sl_mw(burst=3, window_s=2.0)
    mw.check("c", now=0.0)  # will be evicted at t=2.5
    mw.check("c", now=1.0)  # in window at t=2.5
    mw.check("c", now=1.5)  # in window at t=2.5
    # At t=2.5 the first entry (0.0) is evicted → 2 in log → allow
    result = mw.check("c", now=2.5)
    assert result.allowed is True


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_clears_token_bucket_state():
    mw = _tb_mw(burst=1, rate=0.0)
    mw.check("c", now=0.0)  # exhaust
    mw.reset("c")
    result = mw.check("c", now=0.0)
    assert result.allowed is True


def test_reset_clears_fixed_window_state():
    mw = _fw_mw(burst=1, window_s=10.0)
    mw.check("c", now=0.0)  # exhaust
    mw.reset("c")
    result = mw.check("c", now=0.0)
    assert result.allowed is True


def test_reset_clears_sliding_log_state():
    mw = _sl_mw(burst=1, window_s=10.0)
    mw.check("c", now=0.0)  # exhaust
    mw.reset("c")
    result = mw.check("c", now=0.0)
    assert result.allowed is True


def test_reset_unknown_client_is_noop():
    mw = _tb_mw()
    mw.reset("nonexistent")  # must not raise


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------


def test_stats_total_clients_zero():
    mw = _tb_mw()
    assert mw.stats()["total_clients"] == 0


def test_stats_total_clients_increments():
    mw = _tb_mw()
    mw.check("a", now=0.0)
    mw.check("b", now=0.0)
    assert mw.stats()["total_clients"] == 2


def test_stats_algorithm_reported():
    mw = _fw_mw()
    assert mw.stats()["algorithm"] == RateLimitAlgorithm.FIXED_WINDOW.value


def test_stats_algorithm_sliding_log():
    mw = _sl_mw()
    assert mw.stats()["algorithm"] == RateLimitAlgorithm.SLIDING_LOG.value


# ---------------------------------------------------------------------------
# Frozen dataclass immutability
# ---------------------------------------------------------------------------


def test_rate_limit_result_is_frozen():
    r = RateLimitResult(allowed=True, remaining=5, reset_after_s=0.0, client_id="x")
    with pytest.raises((AttributeError, TypeError)):
        r.allowed = False  # type: ignore[misc]


def test_rate_limit_config_is_frozen():
    cfg = RateLimitConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.burst = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in RATE_LIMIT_MIDDLEWARE_REGISTRY


def test_registry_default_is_class():
    assert RATE_LIMIT_MIDDLEWARE_REGISTRY["default"] is RateLimitingMiddleware


def test_registry_default_is_instantiable():
    cls = RATE_LIMIT_MIDDLEWARE_REGISTRY["default"]
    mw = cls()
    assert isinstance(mw, RateLimitingMiddleware)
