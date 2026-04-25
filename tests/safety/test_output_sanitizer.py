"""Tests for output_sanitizer.py — PII scrubbing, URL filtering, secret removal."""

from __future__ import annotations

import pytest

from src.safety.output_sanitizer import (
    OutputSanitizer,
    SanitizationResult,
    SanitizationRule,
)


# ---------------------------------------------------------------------------
# SanitizationRule dataclass fields
# ---------------------------------------------------------------------------

def test_sanitization_rule_has_name():
    rule = SanitizationRule(name="test", pattern=r'\d+', replacement="[NUM]")
    assert rule.name == "test"


def test_sanitization_rule_has_pattern():
    rule = SanitizationRule(name="test", pattern=r'\d+', replacement="[NUM]")
    assert rule.pattern == r'\d+'


def test_sanitization_rule_has_replacement():
    rule = SanitizationRule(name="test", pattern=r'\d+', replacement="[NUM]")
    assert rule.replacement == "[NUM]"


def test_sanitization_rule_enabled_defaults_true():
    rule = SanitizationRule(name="test", pattern=r'\d+', replacement="[NUM]")
    assert rule.enabled is True


def test_sanitization_rule_enabled_can_be_false():
    rule = SanitizationRule(name="test", pattern=r'\d+', replacement="[NUM]", enabled=False)
    assert rule.enabled is False


# ---------------------------------------------------------------------------
# SanitizationResult dataclass fields
# ---------------------------------------------------------------------------

def test_sanitization_result_has_original_length():
    r = SanitizationResult(original_length=10, sanitized_text="hi", rules_applied=[], redaction_count=0)
    assert r.original_length == 10


def test_sanitization_result_has_sanitized_text():
    r = SanitizationResult(original_length=10, sanitized_text="hi", rules_applied=[], redaction_count=0)
    assert r.sanitized_text == "hi"


def test_sanitization_result_has_rules_applied():
    r = SanitizationResult(original_length=10, sanitized_text="hi", rules_applied=["email"], redaction_count=1)
    assert r.rules_applied == ["email"]


def test_sanitization_result_has_redaction_count():
    r = SanitizationResult(original_length=10, sanitized_text="hi", rules_applied=[], redaction_count=5)
    assert r.redaction_count == 5


# ---------------------------------------------------------------------------
# DEFAULT_RULES
# ---------------------------------------------------------------------------

def test_default_rules_has_at_least_5():
    assert len(OutputSanitizer.DEFAULT_RULES) >= 5


def test_default_rules_includes_email():
    names = [r.name for r in OutputSanitizer.DEFAULT_RULES]
    assert "email" in names


def test_default_rules_includes_phone_us():
    names = [r.name for r in OutputSanitizer.DEFAULT_RULES]
    assert "phone_us" in names


def test_default_rules_includes_ssn():
    names = [r.name for r in OutputSanitizer.DEFAULT_RULES]
    assert "ssn" in names


def test_default_rules_includes_api_key():
    names = [r.name for r in OutputSanitizer.DEFAULT_RULES]
    assert "api_key" in names


def test_default_rules_includes_ipv4():
    names = [r.name for r in OutputSanitizer.DEFAULT_RULES]
    assert "ipv4" in names


# ---------------------------------------------------------------------------
# sanitize() — email
# ---------------------------------------------------------------------------

def test_sanitize_email_replaced():
    s = OutputSanitizer()
    result = s.sanitize("Contact us at user@example.com for help.")
    assert "[EMAIL]" in result.sanitized_text
    assert "user@example.com" not in result.sanitized_text


def test_sanitize_email_rule_applied():
    s = OutputSanitizer()
    result = s.sanitize("Send to alice@domain.org please.")
    assert "email" in result.rules_applied


def test_sanitize_email_redaction_count():
    s = OutputSanitizer()
    result = s.sanitize("one@a.com and two@b.com")
    assert result.redaction_count >= 2


# ---------------------------------------------------------------------------
# sanitize() — phone
# ---------------------------------------------------------------------------

def test_sanitize_phone_replaced():
    s = OutputSanitizer()
    result = s.sanitize("Call me at 555-867-5309.")
    assert "[PHONE]" in result.sanitized_text


def test_sanitize_phone_rule_applied():
    s = OutputSanitizer()
    result = s.sanitize("My number is 555.123.4567")
    assert "phone_us" in result.rules_applied


def test_sanitize_phone_dotted_format():
    s = OutputSanitizer()
    result = s.sanitize("Reach me at 800.555.1234")
    assert "[PHONE]" in result.sanitized_text


# ---------------------------------------------------------------------------
# sanitize() — SSN
# ---------------------------------------------------------------------------

def test_sanitize_ssn_replaced():
    s = OutputSanitizer()
    result = s.sanitize("SSN: 123-45-6789")
    assert "[SSN]" in result.sanitized_text


def test_sanitize_ssn_rule_applied():
    s = OutputSanitizer()
    result = s.sanitize("My SSN is 987-65-4321.")
    assert "ssn" in result.rules_applied


def test_sanitize_ssn_original_not_present():
    s = OutputSanitizer()
    result = s.sanitize("SSN: 123-45-6789")
    assert "123-45-6789" not in result.sanitized_text


# ---------------------------------------------------------------------------
# sanitize() — API key
# ---------------------------------------------------------------------------

def test_sanitize_api_key_replaced():
    s = OutputSanitizer()
    result = s.sanitize("Use token sk-abcdefghijklmnopqrstuvwxyz12345 to auth.")
    assert "[API_KEY]" in result.sanitized_text


def test_sanitize_api_key_rule_applied():
    s = OutputSanitizer()
    result = s.sanitize("key_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234")
    assert "api_key" in result.rules_applied


# ---------------------------------------------------------------------------
# sanitize() — IPv4
# ---------------------------------------------------------------------------

def test_sanitize_ipv4_replaced():
    s = OutputSanitizer()
    result = s.sanitize("Server is at 192.168.1.100")
    assert "[IP]" in result.sanitized_text


def test_sanitize_ipv4_rule_applied():
    s = OutputSanitizer()
    result = s.sanitize("Connect to 10.0.0.1")
    assert "ipv4" in result.rules_applied


# ---------------------------------------------------------------------------
# sanitize() — clean text
# ---------------------------------------------------------------------------

def test_sanitize_clean_text_redaction_count_zero():
    s = OutputSanitizer()
    result = s.sanitize("Hello, how are you today?")
    assert result.redaction_count == 0


def test_sanitize_clean_text_rules_applied_empty():
    s = OutputSanitizer()
    result = s.sanitize("The sky is blue.")
    assert result.rules_applied == []


def test_sanitize_clean_text_unchanged():
    s = OutputSanitizer()
    text = "Nothing sensitive here."
    result = s.sanitize(text)
    assert result.sanitized_text == text


def test_sanitize_original_length_recorded():
    s = OutputSanitizer()
    text = "Hello world"
    result = s.sanitize(text)
    assert result.original_length == len(text)


# ---------------------------------------------------------------------------
# add_rule()
# ---------------------------------------------------------------------------

def test_add_rule_new_rule_applied():
    s = OutputSanitizer()
    s.add_rule(SanitizationRule(name="zip", pattern=r'\b\d{5}\b', replacement="[ZIP]"))
    result = s.sanitize("Zip code is 90210.")
    assert "[ZIP]" in result.sanitized_text


def test_add_rule_appears_in_rules_applied():
    s = OutputSanitizer()
    s.add_rule(SanitizationRule(name="zip", pattern=r'\b\d{5}\b', replacement="[ZIP]"))
    result = s.sanitize("Zip: 90210")
    assert "zip" in result.rules_applied


def test_add_rule_does_not_affect_other_sanitizers():
    s1 = OutputSanitizer()
    s2 = OutputSanitizer()
    s1.add_rule(SanitizationRule(name="zip", pattern=r'\b\d{5}\b', replacement="[ZIP]"))
    result = s2.sanitize("Zip: 90210")
    assert "zip" not in result.rules_applied


# ---------------------------------------------------------------------------
# disable_rule()
# ---------------------------------------------------------------------------

def test_disable_rule_returns_true_when_found():
    s = OutputSanitizer()
    assert s.disable_rule("email") is True


def test_disable_rule_returns_false_when_not_found():
    s = OutputSanitizer()
    assert s.disable_rule("nonexistent_rule") is False


def test_disable_rule_rule_no_longer_applied():
    s = OutputSanitizer()
    s.disable_rule("email")
    result = s.sanitize("user@example.com")
    assert "email" not in result.rules_applied
    assert "[EMAIL]" not in result.sanitized_text


def test_disable_rule_other_rules_still_active():
    s = OutputSanitizer()
    s.disable_rule("email")
    result = s.sanitize("Call 555-123-4567")
    assert "phone_us" in result.rules_applied


def test_sanitize_returns_result_type():
    s = OutputSanitizer()
    result = s.sanitize("some text")
    assert isinstance(result, SanitizationResult)
