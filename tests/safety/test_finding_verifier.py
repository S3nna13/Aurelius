"""Tests for src.safety.finding_verifier."""

from __future__ import annotations

import pytest

from src.safety.finding_verifier import (
    FINDING_VERIFIER_REGISTRY,
    SEVERITY_CVSS_MINIMUMS,
    VALID_SEVERITIES,
    FindingVerifier,
    VerificationResult,
    _sanitize_for_verifier,
)


@pytest.fixture
def verifier() -> FindingVerifier:
    return FindingVerifier()


GOOD_DESC = "A detailed description of the vulnerability and repro steps."


def test_registry_has_default():
    assert FINDING_VERIFIER_REGISTRY["default"] is FindingVerifier


def test_valid_severities_are_expected():
    assert VALID_SEVERITIES == frozenset({"critical", "high", "medium", "low", "info"})


def test_cvss_minimums():
    assert SEVERITY_CVSS_MINIMUMS["critical"] == 9.0
    assert SEVERITY_CVSS_MINIMUMS["high"] == 7.0
    assert SEVERITY_CVSS_MINIMUMS["medium"] == 4.0
    assert SEVERITY_CVSS_MINIMUMS["low"] == 0.1
    assert SEVERITY_CVSS_MINIMUMS["info"] == 0.0


def test_verification_result_frozen():
    r = VerificationResult(passed=True, confidence=1.0, reason="ok")
    with pytest.raises(Exception):
        r.passed = False  # type: ignore[misc]


def test_valid_finding_passes(verifier: FindingVerifier):
    r = verifier.verify("SQL injection in /login", "high", GOOD_DESC)
    assert r.passed is True
    assert r.reason == "ok"
    assert r.confidence == 1.0
    assert len(r.checks_failed) == 0


def test_generic_title_bug_rejected(verifier: FindingVerifier):
    r = verifier.verify("bug", "high", GOOD_DESC)
    assert r.passed is False
    assert "title_not_generic" in r.checks_failed


def test_generic_title_finding_rejected(verifier: FindingVerifier):
    r = verifier.verify("finding", "high", GOOD_DESC)
    assert r.passed is False


def test_generic_title_case_insensitive(verifier: FindingVerifier):
    r = verifier.verify("  FINDING  ", "high", GOOD_DESC)
    assert r.passed is False


def test_empty_title_rejected(verifier: FindingVerifier):
    r = verifier.verify("", "high", GOOD_DESC)
    assert r.passed is False
    assert "title_not_generic" in r.checks_failed


def test_unknown_severity_rejected(verifier: FindingVerifier):
    r = verifier.verify("SSRF in webhook", "catastrophic", GOOD_DESC)
    assert r.passed is False
    assert "severity_valid" in r.checks_failed


def test_empty_severity_rejected(verifier: FindingVerifier):
    r = verifier.verify("SSRF in webhook", "", GOOD_DESC)
    assert r.passed is False


def test_all_valid_severities_accepted(verifier: FindingVerifier):
    for sev in VALID_SEVERITIES:
        r = verifier.verify("Real issue title", sev, GOOD_DESC)
        assert r.passed is True, sev


def test_severity_case_insensitive(verifier: FindingVerifier):
    r = verifier.verify("Real issue title", "HIGH", GOOD_DESC)
    assert r.passed is True


def test_short_description_rejected(verifier: FindingVerifier):
    r = verifier.verify("SSRF in webhook", "high", "too short")
    assert r.passed is False
    assert "description_length" in r.checks_failed


def test_description_exactly_20_chars_passes(verifier: FindingVerifier):
    desc = "x" * 20
    r = verifier.verify("SSRF in webhook", "high", desc)
    assert r.passed is True


def test_cvss_consistent_critical(verifier: FindingVerifier):
    r = verifier.verify("RCE", "critical", GOOD_DESC, cvss_score=9.8)
    assert r.passed is True
    assert "cvss_consistent" in r.checks_passed


def test_cvss_inconsistent_critical(verifier: FindingVerifier):
    r = verifier.verify("RCE", "critical", GOOD_DESC, cvss_score=5.0)
    assert r.passed is False
    assert "cvss_consistent" in r.checks_failed


def test_cvss_high_threshold(verifier: FindingVerifier):
    ok = verifier.verify("Path traversal", "high", GOOD_DESC, cvss_score=7.5)
    bad = verifier.verify("Path traversal", "high", GOOD_DESC, cvss_score=6.9)
    assert ok.passed is True
    assert bad.passed is False


def test_cvss_none_skipped(verifier: FindingVerifier):
    r = verifier.verify("XSS", "medium", GOOD_DESC, cvss_score=None)
    assert r.passed is True
    assert "cvss_consistent" not in r.checks_passed
    assert "cvss_consistent" not in r.checks_failed


def test_cvss_non_numeric_fails(verifier: FindingVerifier):
    r = verifier.verify("XSS", "medium", GOOD_DESC, cvss_score="not-a-number")  # type: ignore[arg-type]
    assert r.passed is False
    assert "cvss_consistent" in r.checks_failed


def test_sanitize_strips_control_chars():
    out = _sanitize_for_verifier("hello\x00\x01world\x7f")
    assert "\x00" not in out
    assert "\x01" not in out
    assert "\x7f" not in out
    assert "hello" in out and "world" in out


def test_sanitize_collapses_spaces():
    out = _sanitize_for_verifier("a\x00\x00\x00b")
    assert "  " not in out


def test_sanitize_redacts_confidence_injection():
    out = _sanitize_for_verifier("Confidence(0-10):8")
    assert "[REDACTED]" in out


def test_sanitize_strips_newlines():
    # Newlines are C0 control chars and get stripped; bare number is no longer on its own line
    out = _sanitize_for_verifier("some text\n85\nmore")
    assert "\n" not in out
    assert "85" in out  # number survives, newlines do not


def test_sanitize_none_safe():
    assert _sanitize_for_verifier("") == ""


def test_confidence_pass_ratio(verifier: FindingVerifier):
    # 2 of 3 checks passing: generic title fails.
    r = verifier.verify("bug", "high", GOOD_DESC)
    assert r.confidence == pytest.approx(2 / 3)


def test_confidence_with_cvss_all_pass(verifier: FindingVerifier):
    r = verifier.verify("RCE in parser", "critical", GOOD_DESC, cvss_score=9.5)
    assert r.confidence == 1.0


def test_checks_passed_list_content(verifier: FindingVerifier):
    r = verifier.verify("Real title", "medium", GOOD_DESC)
    assert "title_not_generic" in r.checks_passed
    assert "severity_valid" in r.checks_passed
    assert "description_length" in r.checks_passed


def test_result_type(verifier: FindingVerifier):
    r = verifier.verify("Real title", "low", GOOD_DESC)
    assert isinstance(r, VerificationResult)


def test_description_with_control_chars_sanitized_to_short(verifier: FindingVerifier):
    # Short after stripping controls
    r = verifier.verify("Real title", "low", "\x00" * 100)
    assert r.passed is False
    assert "description_length" in r.checks_failed
