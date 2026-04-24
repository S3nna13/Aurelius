"""Tests for src.safety.output_filter — ≥28 test cases."""

from __future__ import annotations

import pytest

from src.safety.output_filter import (
    FilterAction,
    FilterResult,
    FilterRule,
    OutputFilter,
    OUTPUT_FILTER_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def filt() -> OutputFilter:
    return OutputFilter()


# ---------------------------------------------------------------------------
# Default rule: email redaction
# ---------------------------------------------------------------------------

def test_email_is_redacted(filt: OutputFilter) -> None:
    result = filt.filter("Contact us at user@example.com for help.")
    assert "[EMAIL]" in result.filtered
    assert "user@example.com" not in result.filtered


def test_email_original_preserved(filt: OutputFilter) -> None:
    text = "Email: info@test.org"
    result = filt.filter(text)
    assert result.original == text


def test_email_triggers_rule_name(filt: OutputFilter) -> None:
    result = filt.filter("hello@world.co")
    assert "pii_email" in result.triggered_rules


def test_email_action_is_redact(filt: OutputFilter) -> None:
    result = filt.filter("reach me at foo@bar.com")
    assert result.action == FilterAction.REDACT


# ---------------------------------------------------------------------------
# Default rule: phone redaction
# ---------------------------------------------------------------------------

def test_phone_is_redacted(filt: OutputFilter) -> None:
    result = filt.filter("Call us at 555-123-4567 anytime.")
    assert "[PHONE]" in result.filtered
    assert "555-123-4567" not in result.filtered


def test_phone_with_dots(filt: OutputFilter) -> None:
    result = filt.filter("My number is 800.555.1234.")
    assert "[PHONE]" in result.filtered


def test_phone_original_preserved(filt: OutputFilter) -> None:
    text = "Reach me at 212-555-6789"
    result = filt.filter(text)
    assert result.original == text


def test_phone_rule_name_in_triggered(filt: OutputFilter) -> None:
    result = filt.filter("555-867-5309")
    assert "pii_phone" in result.triggered_rules


# ---------------------------------------------------------------------------
# Default rule: SSN redaction
# ---------------------------------------------------------------------------

def test_ssn_is_redacted(filt: OutputFilter) -> None:
    result = filt.filter("SSN: 123-45-6789")
    assert "[SSN]" in result.filtered
    assert "123-45-6789" not in result.filtered


def test_ssn_original_preserved(filt: OutputFilter) -> None:
    text = "SSN is 987-65-4321"
    result = filt.filter(text)
    assert result.original == text


def test_ssn_rule_name_in_triggered(filt: OutputFilter) -> None:
    result = filt.filter("Your SSN: 234-56-7890")
    assert "pii_ssn" in result.triggered_rules


# ---------------------------------------------------------------------------
# Clean text — no rules triggered
# ---------------------------------------------------------------------------

def test_clean_text_action_is_allow(filt: OutputFilter) -> None:
    result = filt.filter("The weather is nice today.")
    assert result.action == FilterAction.ALLOW
    assert result.triggered_rules == []
    assert result.filtered == result.original


def test_empty_text_no_rules_triggered(filt: OutputFilter) -> None:
    result = filt.filter("")
    assert result.triggered_rules == []


# ---------------------------------------------------------------------------
# BLOCK action
# ---------------------------------------------------------------------------

def test_block_returns_blocked_text() -> None:
    block_rule = FilterRule(
        name="bad_word",
        pattern=r"\bharmful\b",
        action=FilterAction.BLOCK,
    )
    f = OutputFilter(rules=[block_rule])
    result = f.filter("This is harmful content.")
    assert result.filtered == "[BLOCKED]"
    assert result.action == FilterAction.BLOCK


def test_block_terminates_immediately() -> None:
    block_rule = FilterRule(
        name="block_me",
        pattern=r"badword",
        action=FilterAction.BLOCK,
    )
    redact_rule = FilterRule(
        name="redact_email",
        pattern=r'\b[\w.-]+@[\w.-]+\.\w{2,}\b',
        action=FilterAction.REDACT,
        replacement="[EMAIL]",
    )
    f = OutputFilter(rules=[block_rule, redact_rule])
    result = f.filter("badword and test@example.com")
    # BLOCK fires first; should not have EMAIL redaction applied
    assert result.filtered == "[BLOCKED]"
    assert "block_me" in result.triggered_rules


def test_block_original_still_preserved() -> None:
    block_rule = FilterRule(name="blk", pattern=r"destroy", action=FilterAction.BLOCK)
    f = OutputFilter(rules=[block_rule])
    text = "destroy everything"
    result = f.filter(text)
    assert result.original == text


# ---------------------------------------------------------------------------
# WARN action
# ---------------------------------------------------------------------------

def test_warn_preserves_text() -> None:
    warn_rule = FilterRule(name="warn_me", pattern=r"risky", action=FilterAction.WARN)
    f = OutputFilter(rules=[warn_rule])
    result = f.filter("This might be risky content.")
    assert "risky" in result.filtered
    assert result.filtered == result.original or "risky" in result.filtered


def test_warn_adds_to_triggered_rules() -> None:
    warn_rule = FilterRule(name="warn_me", pattern=r"risky", action=FilterAction.WARN)
    f = OutputFilter(rules=[warn_rule])
    result = f.filter("This might be risky content.")
    assert "warn_me" in result.triggered_rules


def test_warn_does_not_alter_text() -> None:
    warn_rule = FilterRule(name="w", pattern=r"caution", action=FilterAction.WARN)
    f = OutputFilter(rules=[warn_rule])
    text = "Please use caution here."
    result = f.filter(text)
    assert result.filtered == text


# ---------------------------------------------------------------------------
# add_rule
# ---------------------------------------------------------------------------

def test_add_rule_appended(filt: OutputFilter) -> None:
    new_rule = FilterRule(name="custom", pattern=r"secret", action=FilterAction.REDACT)
    filt.add_rule(new_rule)
    result = filt.filter("This is a secret message.")
    assert "custom" in result.triggered_rules


def test_add_rule_redacts_new_pattern(filt: OutputFilter) -> None:
    rule = FilterRule(
        name="credit_card",
        pattern=r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
        action=FilterAction.REDACT,
        replacement="[CARD]",
    )
    filt.add_rule(rule)
    result = filt.filter("Card: 1234-5678-9012-3456")
    assert "[CARD]" in result.filtered


# ---------------------------------------------------------------------------
# remove_rule
# ---------------------------------------------------------------------------

def test_remove_rule_returns_true_when_found(filt: OutputFilter) -> None:
    assert filt.remove_rule("pii_email") is True


def test_remove_rule_returns_false_when_not_found(filt: OutputFilter) -> None:
    assert filt.remove_rule("nonexistent_rule") is False


def test_remove_rule_disables_matching(filt: OutputFilter) -> None:
    filt.remove_rule("pii_email")
    result = filt.filter("Email: user@example.com")
    assert "pii_email" not in result.triggered_rules


# ---------------------------------------------------------------------------
# FilterResult fields
# ---------------------------------------------------------------------------

def test_filter_result_triggered_rules_populated(filt: OutputFilter) -> None:
    result = filt.filter("Contact: dev@foo.bar and SSN: 000-00-0001")
    assert len(result.triggered_rules) >= 2


def test_filter_result_is_not_frozen_instance() -> None:
    # FilterResult is frozen; verify immutability
    result = FilterResult(
        original="x",
        filtered="y",
        action=FilterAction.ALLOW,
        triggered_rules=[],
    )
    with pytest.raises((AttributeError, TypeError)):
        result.filtered = "z"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Multiple rules applied in order
# ---------------------------------------------------------------------------

def test_multiple_redact_rules_both_applied(filt: OutputFilter) -> None:
    result = filt.filter("Call 555-123-4567 or email test@example.com")
    assert "[PHONE]" in result.filtered
    assert "[EMAIL]" in result.filtered


def test_rules_applied_in_order_redact_then_warn() -> None:
    redact_rule = FilterRule(
        name="r1", pattern=r"alpha", action=FilterAction.REDACT, replacement="[A]"
    )
    warn_rule = FilterRule(name="w1", pattern=r"beta", action=FilterAction.WARN)
    f = OutputFilter(rules=[redact_rule, warn_rule])
    result = f.filter("alpha and beta text")
    assert "[A]" in result.filtered
    assert "r1" in result.triggered_rules
    assert "w1" in result.triggered_rules


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_default() -> None:
    assert "default" in OUTPUT_FILTER_REGISTRY


def test_registry_default_is_output_filter_class() -> None:
    cls = OUTPUT_FILTER_REGISTRY["default"]
    instance = cls()
    assert isinstance(instance, OutputFilter)
