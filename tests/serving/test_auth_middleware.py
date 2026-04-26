"""Tests for src/serving/auth_middleware.py

Finding AUR-SEC-2026-0012 — authentication middleware for the serving surface.

Run with:
    .venv/bin/python3.14 -m pytest tests/serving/test_auth_middleware.py -v --tb=short
"""

from __future__ import annotations

import hashlib

import pytest

from src.serving.auth_middleware import (
    AUTH_MIDDLEWARE_REGISTRY,
    DEFAULT_AUTH_MIDDLEWARE,
    APIKey,
    AuthConfig,
    AuthMiddleware,
    AuthResult,
    _sha256_hex,
    _timing_safe_equal,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RAW_KEY = "test-secret-key-abc123"
RAW_KEY_2 = "another-secret-xyz"


@pytest.fixture()
def middleware() -> AuthMiddleware:
    """Fresh middleware with require_auth=True and one pre-registered key."""
    config = AuthConfig(keys={}, require_auth=True)
    mw = AuthMiddleware(config)
    mw.add_key("key-1", RAW_KEY, frozenset({"read", "write"}), rate_limit_rps=50.0)
    return mw


@pytest.fixture()
def open_middleware() -> AuthMiddleware:
    """Middleware with require_auth=False (dev mode)."""
    config = AuthConfig(keys={}, require_auth=False)
    return AuthMiddleware(config)


# ---------------------------------------------------------------------------
# 1. Valid Bearer token → authenticated=True
# ---------------------------------------------------------------------------


def test_valid_bearer_token(middleware: AuthMiddleware) -> None:
    result = middleware.authenticate({"Authorization": f"Bearer {RAW_KEY}"})
    assert result.authenticated is True
    assert result.key_id == "key-1"
    assert result.error is None
    assert "read" in result.scopes
    assert "write" in result.scopes


def test_bearer_token_case_insensitive_prefix(middleware: AuthMiddleware) -> None:
    """'bearer' prefix must be matched case-insensitively."""
    result = middleware.authenticate({"Authorization": f"BEARER {RAW_KEY}"})
    assert result.authenticated is True


# ---------------------------------------------------------------------------
# 2. Valid X-API-Key header → authenticated=True
# ---------------------------------------------------------------------------


def test_valid_x_api_key(middleware: AuthMiddleware) -> None:
    result = middleware.authenticate({"X-API-Key": RAW_KEY})
    assert result.authenticated is True
    assert result.key_id == "key-1"
    assert result.error is None


def test_x_api_key_header_case_insensitive(middleware: AuthMiddleware) -> None:
    """Header name lookup must be case-insensitive."""
    result = middleware.authenticate({"x-api-key": RAW_KEY})
    assert result.authenticated is True


# ---------------------------------------------------------------------------
# 3. Wrong key → authenticated=False
# ---------------------------------------------------------------------------


def test_wrong_bearer_token(middleware: AuthMiddleware) -> None:
    result = middleware.authenticate({"Authorization": "Bearer totally-wrong-key"})
    assert result.authenticated is False
    assert result.key_id is None
    assert result.error is not None


def test_wrong_x_api_key(middleware: AuthMiddleware) -> None:
    result = middleware.authenticate({"X-API-Key": "not-the-right-key"})
    assert result.authenticated is False
    assert result.error is not None


# ---------------------------------------------------------------------------
# 4. Missing header when require_auth=True → authenticated=False
# ---------------------------------------------------------------------------


def test_missing_header_require_auth_true(middleware: AuthMiddleware) -> None:
    result = middleware.authenticate({})
    assert result.authenticated is False
    assert result.error is not None


def test_missing_header_with_unrelated_headers(middleware: AuthMiddleware) -> None:
    result = middleware.authenticate({"Content-Type": "application/json"})
    assert result.authenticated is False


# ---------------------------------------------------------------------------
# 5. require_auth=False + no header → authenticated=True (dev mode)
# ---------------------------------------------------------------------------


def test_dev_mode_no_header_passes(open_middleware: AuthMiddleware) -> None:
    result = open_middleware.authenticate({})
    assert result.authenticated is True
    assert result.key_id is None
    assert result.error is None


def test_dev_mode_with_valid_key_also_passes(open_middleware: AuthMiddleware) -> None:
    open_middleware.add_key("k", RAW_KEY, frozenset({"read"}))
    result = open_middleware.authenticate({"X-API-Key": RAW_KEY})
    assert result.authenticated is True


# ---------------------------------------------------------------------------
# 6. Scope check: key has {"read"}, requires "write" → False
# ---------------------------------------------------------------------------


def test_scope_check_missing_scope(middleware: AuthMiddleware) -> None:
    """Add a read-only key and verify it fails a write scope check."""
    middleware.add_key("read-only", RAW_KEY_2, frozenset({"read"}))
    result = middleware.authenticate({"X-API-Key": RAW_KEY_2})
    assert result.authenticated is True
    assert middleware.require_scope(result, "write") is False


def test_scope_check_present_scope(middleware: AuthMiddleware) -> None:
    result = middleware.authenticate({"X-API-Key": RAW_KEY})
    assert middleware.require_scope(result, "read") is True
    assert middleware.require_scope(result, "write") is True


def test_scope_check_unauthenticated_result() -> None:
    """require_scope must return False for an unauthenticated result."""
    bad_result = AuthResult(
        authenticated=False,
        key_id=None,
        scopes=frozenset({"read", "write", "admin"}),
        error="Invalid API key.",
    )
    mw = AuthMiddleware(AuthConfig(keys={}, require_auth=True))
    assert mw.require_scope(bad_result, "read") is False


# ---------------------------------------------------------------------------
# 7. Key removal: after remove_key, same key → authenticated=False
# ---------------------------------------------------------------------------


def test_remove_key_revokes_access(middleware: AuthMiddleware) -> None:
    # Confirm it works before removal.
    before = middleware.authenticate({"X-API-Key": RAW_KEY})
    assert before.authenticated is True

    removed = middleware.remove_key("key-1")
    assert removed is True

    after = middleware.authenticate({"X-API-Key": RAW_KEY})
    assert after.authenticated is False


def test_remove_nonexistent_key_returns_false(middleware: AuthMiddleware) -> None:
    assert middleware.remove_key("does-not-exist") is False


def test_has_key(middleware: AuthMiddleware) -> None:
    assert middleware.has_key("key-1") is True
    middleware.remove_key("key-1")
    assert middleware.has_key("key-1") is False


# ---------------------------------------------------------------------------
# 8. Timing safety: comparison does NOT short-circuit
# ---------------------------------------------------------------------------


def test_timing_safe_equal_uses_hmac_compare_digest() -> None:
    """_timing_safe_equal must return False for non-equal strings without
    short-circuiting.  We verify by ensuring the function is backed by
    hmac.compare_digest (qualitative, not timing-based)."""
    import inspect

    source = inspect.getsource(_timing_safe_equal)
    assert "hmac.compare_digest" in source or "compare_digest" in source, (
        "_timing_safe_equal must use hmac.compare_digest"
    )


def test_timing_safe_equal_correctness() -> None:
    assert _timing_safe_equal("abc", "abc") is True
    assert _timing_safe_equal("abc", "abd") is False
    assert _timing_safe_equal("", "") is True
    assert _timing_safe_equal("a", "b") is False


def test_authenticate_always_hashes_supplied_key(middleware: AuthMiddleware) -> None:
    """The middleware must compute the hash of the supplied key before
    comparing — i.e. it never stores or compares raw keys.
    We verify by checking that the stored key_hash equals sha256(RAW_KEY)."""
    expected_hash = hashlib.sha256(RAW_KEY.encode()).hexdigest()
    # Access internal _keys dict (white-box test).
    stored = middleware._keys["key-1"]
    assert stored.key_hash == expected_hash
    assert stored.key_hash != RAW_KEY  # never stored in plaintext


# ---------------------------------------------------------------------------
# 9. Adversarial: empty string key → does not crash
# ---------------------------------------------------------------------------


def test_empty_string_key_does_not_crash(middleware: AuthMiddleware) -> None:
    result = middleware.authenticate({"X-API-Key": ""})
    # An empty key should simply not match any registered key.
    assert isinstance(result, AuthResult)
    assert result.authenticated is False


def test_add_key_empty_string_raw_key(middleware: AuthMiddleware) -> None:
    """Adding an empty-string key must not raise."""
    api_key = middleware.add_key("empty-key", "", frozenset({"read"}))
    assert isinstance(api_key, APIKey)
    # Empty key hash must be sha256("").
    expected = hashlib.sha256(b"").hexdigest()
    assert api_key.key_hash == expected


def test_bearer_empty_after_prefix(middleware: AuthMiddleware) -> None:
    """'Bearer ' followed by nothing should not crash."""
    result = middleware.authenticate({"Authorization": "Bearer "})
    assert isinstance(result, AuthResult)


# ---------------------------------------------------------------------------
# 10. Adversarial: very long key (10 KB) → handles without error
# ---------------------------------------------------------------------------


def test_very_long_key_does_not_crash(middleware: AuthMiddleware) -> None:
    long_key = "x" * 10_240  # 10 KB
    result = middleware.authenticate({"X-API-Key": long_key})
    assert isinstance(result, AuthResult)
    assert result.authenticated is False  # not registered


def test_add_and_authenticate_very_long_key(middleware: AuthMiddleware) -> None:
    long_key = "y" * 10_240
    middleware.add_key("big-key", long_key, frozenset({"read"}))
    result = middleware.authenticate({"X-API-Key": long_key})
    assert result.authenticated is True
    assert result.key_id == "big-key"


# ---------------------------------------------------------------------------
# 11. Module-level registry sanity checks
# ---------------------------------------------------------------------------


def test_auth_middleware_registry_is_dict() -> None:
    assert isinstance(AUTH_MIDDLEWARE_REGISTRY, dict)


def test_default_auth_middleware_is_closed() -> None:
    """DEFAULT_AUTH_MIDDLEWARE must be fail-closed (require_auth=True).

    Regression test for AUR-SEC-2026-0028: the previous default was
    require_auth=False, which left APIs accidentally open in production.
    """
    result = DEFAULT_AUTH_MIDDLEWARE.authenticate({})
    assert result.authenticated is False
    assert result.error is not None


# ---------------------------------------------------------------------------
# 12. add_key returns correct APIKey
# ---------------------------------------------------------------------------


def test_add_key_returns_api_key() -> None:
    mw = AuthMiddleware(AuthConfig(keys={}, require_auth=True))
    api_key = mw.add_key("svc", "mysecret", frozenset({"admin"}), rate_limit_rps=200.0)
    assert isinstance(api_key, APIKey)
    assert api_key.key_id == "svc"
    assert api_key.scopes == frozenset({"admin"})
    assert api_key.rate_limit_rps == 200.0
    assert api_key.key_hash == _sha256_hex("mysecret")
    assert api_key.key_hash != "mysecret"


# ---------------------------------------------------------------------------
# 13. Multiple keys coexist without collision
# ---------------------------------------------------------------------------


def test_multiple_keys_no_collision() -> None:
    mw = AuthMiddleware(AuthConfig(keys={}, require_auth=True))
    mw.add_key("k1", "secret-one", frozenset({"read"}))
    mw.add_key("k2", "secret-two", frozenset({"write"}))

    r1 = mw.authenticate({"X-API-Key": "secret-one"})
    assert r1.authenticated is True
    assert r1.key_id == "k1"

    r2 = mw.authenticate({"X-API-Key": "secret-two"})
    assert r2.authenticated is True
    assert r2.key_id == "k2"

    # Cross-contamination check.
    r_wrong = mw.authenticate({"X-API-Key": "secret-one-extra"})
    assert r_wrong.authenticated is False


# ---------------------------------------------------------------------------
# 14. Bearer takes precedence over X-API-Key when both are present
# ---------------------------------------------------------------------------


def test_bearer_takes_precedence_over_x_api_key() -> None:
    mw = AuthMiddleware(AuthConfig(keys={}, require_auth=True))
    mw.add_key("bearer-key", "bearer-secret", frozenset({"read"}))
    mw.add_key("api-key", "apikey-secret", frozenset({"write"}))

    # Valid Bearer, invalid X-API-Key.
    result = mw.authenticate(
        {
            "Authorization": "Bearer bearer-secret",
            "X-API-Key": "apikey-secret",
        }
    )
    # Bearer is tried first; it succeeds — key_id is "bearer-key".
    assert result.authenticated is True
    assert result.key_id == "bearer-key"
