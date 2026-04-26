"""Tests for message_content_filter — PII, toxicity, policy scanning.

Security surface: STRIDE Information Disclosure / Denial of Service.
"""

from __future__ import annotations

from src.security.message_content_filter import (
    DEFAULT_MESSAGE_FILTER,
    MESSAGE_FILTER_REGISTRY,
    FilterConfig,
    FilterVerdict,
    MessageContentFilter,
)

# ---------------------------------------------------------------------------
# Allow benign text
# ---------------------------------------------------------------------------


def test_benign_text_allowed():
    f = MessageContentFilter()
    result = f.scan("Hello, this is a normal system message.")
    assert result.verdict == FilterVerdict.ALLOW
    assert result.matches == []


# ---------------------------------------------------------------------------
# PII detection
# ---------------------------------------------------------------------------


def test_email_detected_sanitizes():
    f = MessageContentFilter()
    result = f.scan("Contact me at alice@example.com please.")
    assert result.verdict == FilterVerdict.SANITIZE
    assert any(m.rule_name == "email" for m in result.matches)
    assert "[REDACTED]" in result.sanitized


def test_phone_detected_sanitizes():
    f = MessageContentFilter()
    result = f.scan("Call 555-123-4567 for support.")
    assert result.verdict == FilterVerdict.SANITIZE
    assert any(m.rule_name == "phone" for m in result.matches)


def test_ssn_detected_sanitizes():
    f = MessageContentFilter()
    result = f.scan("My SSN is 123-45-6789.")
    assert result.verdict == FilterVerdict.SANITIZE
    assert any(m.rule_name == "ssn" for m in result.matches)


def test_multiple_pii_redacts_all():
    f = MessageContentFilter()
    result = f.scan("Email: bob@test.com, Phone: 555-123-4567")
    assert result.verdict == FilterVerdict.SANITIZE
    assert result.sanitized.count("[REDACTED]") == 2


# ---------------------------------------------------------------------------
# Toxicity detection
# ---------------------------------------------------------------------------


def test_toxic_word_blocks():
    f = MessageContentFilter()
    result = f.scan("I will attack the server tonight.")
    assert result.verdict == FilterVerdict.BLOCK
    assert any(m.category == "toxicity" for m in result.matches)


def test_toxicity_beats_pii():
    """If toxicity and PII both present, BLOCK wins."""
    f = MessageContentFilter()
    result = f.scan("Contact me at alice@example.com to plan the attack.")
    assert result.verdict == FilterVerdict.BLOCK


# ---------------------------------------------------------------------------
# Custom blocklist
# ---------------------------------------------------------------------------


def test_custom_blocklist_blocks():
    f = MessageContentFilter(FilterConfig(custom_blocklist=["proprietary", "secret sauce"]))
    result = f.scan("The secret sauce is in the repo.")
    assert result.verdict == FilterVerdict.BLOCK
    assert any(m.matched_text == "secret sauce" for m in result.matches)


# ---------------------------------------------------------------------------
# Oversized payload — fail closed
# ---------------------------------------------------------------------------


def test_oversized_blocked_by_default():
    f = MessageContentFilter(FilterConfig(max_payload_length=10))
    result = f.scan("x" * 100)
    assert result.verdict == FilterVerdict.BLOCK
    assert any(m.category == "oversized" for m in result.matches)


def test_oversized_allowed_when_configured():
    f = MessageContentFilter(FilterConfig(max_payload_length=10, block_on_oversized=False))
    result = f.scan("x" * 100)
    # Should pass through to other rules; no PII/toxicity so ALLOW
    assert result.verdict == FilterVerdict.ALLOW


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_string_allowed():
    f = MessageContentFilter()
    assert f.scan("").verdict == FilterVerdict.ALLOW


def test_check_convenience_true_for_allow():
    f = MessageContentFilter()
    assert f.check("safe text") is True


def test_check_convenience_false_for_block():
    f = MessageContentFilter()
    assert f.check("attack!") is False


# ---------------------------------------------------------------------------
# Disabled rules
# ---------------------------------------------------------------------------


def test_pii_disabled_allows_email():
    f = MessageContentFilter(FilterConfig(pii_rules_enabled=False))
    result = f.scan("alice@example.com")
    assert result.verdict == FilterVerdict.ALLOW


def test_toxicity_disabled_allows_toxic():
    f = MessageContentFilter(FilterConfig(toxicity_rules_enabled=False))
    result = f.scan("attack!")
    assert result.verdict == FilterVerdict.ALLOW


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in MESSAGE_FILTER_REGISTRY
    assert isinstance(MESSAGE_FILTER_REGISTRY["default"], MessageContentFilter)


def test_default_is_message_filter():
    assert isinstance(DEFAULT_MESSAGE_FILTER, MessageContentFilter)
