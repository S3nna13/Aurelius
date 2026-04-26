"""Tests for the LLM output scanner."""

from __future__ import annotations

from dataclasses import fields

import pytest

from src.security.output_scanner import OutputScanner, ScanResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scanner() -> OutputScanner:
    return OutputScanner(max_expected_length=512, toxic_threshold=0.3)


# ---------------------------------------------------------------------------
# Test 1: OutputScanner instantiates without error
# ---------------------------------------------------------------------------


def test_instantiates():
    """OutputScanner instantiates without error using default parameters."""
    s = OutputScanner()
    assert s is not None
    assert s.max_expected_length == 512
    assert s.toxic_threshold == 0.3


# ---------------------------------------------------------------------------
# Test 2: detect_pii_text finds email address
# ---------------------------------------------------------------------------


def test_detect_pii_text_finds_email(scanner):
    """detect_pii_text detects an email address embedded in text."""
    text = "Please send the report to user@example.com by Friday."
    found, pii_types = scanner.detect_pii_text(text)
    assert found is True
    assert "email" in pii_types


# ---------------------------------------------------------------------------
# Test 3: detect_pii_text finds SSN
# ---------------------------------------------------------------------------


def test_detect_pii_text_finds_ssn(scanner):
    """detect_pii_text detects an SSN in 123-45-6789 format."""
    text = "Employee SSN on file: 123-45-6789."
    found, pii_types = scanner.detect_pii_text(text)
    assert found is True
    assert "ssn" in pii_types


# ---------------------------------------------------------------------------
# Test 4: detect_pii_text finds IPv4 address
# ---------------------------------------------------------------------------


def test_detect_pii_text_finds_ipv4(scanner):
    """detect_pii_text detects an IPv4 address in text."""
    text = "Server is running at 192.168.1.100."
    found, pii_types = scanner.detect_pii_text(text)
    assert found is True
    assert "ipv4" in pii_types


# ---------------------------------------------------------------------------
# Test 5: clean text returns has_pii=False
# ---------------------------------------------------------------------------


def test_detect_pii_text_clean_text(scanner):
    """detect_pii_text returns has_pii=False for text without PII."""
    text = "The weather today is sunny and warm with a gentle breeze."
    found, pii_types = scanner.detect_pii_text(text)
    assert found is False
    assert pii_types == []


# ---------------------------------------------------------------------------
# Test 6: 'email' in pii_types when email found
# ---------------------------------------------------------------------------


def test_pii_types_contains_email(scanner):
    """pii_types list contains 'email' when an email address is detected."""
    text = "Contact support@company.org for help."
    found, pii_types = scanner.detect_pii_text(text)
    assert found is True
    assert "email" in pii_types


# ---------------------------------------------------------------------------
# Test 7: toxic_score returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_toxic_score_range(scanner):
    """toxic_score returns a float in [0, 1]."""
    token_ids = [5, 50, 105, 200, 300]
    score = scanner.toxic_score(token_ids)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Test 8: tokens outside toxic ranges → score = 0.0
# ---------------------------------------------------------------------------


def test_toxic_score_clean_tokens(scanner):
    """Tokens entirely outside toxic ranges produce a score of 0.0."""
    # Ranges are 1-20 and 100-110; use tokens well outside these
    token_ids = [50, 60, 70, 80, 90, 200, 500]
    score = scanner.toxic_score(token_ids)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Test 9: tokens in toxic ranges → score > 0
# ---------------------------------------------------------------------------


def test_toxic_score_toxic_tokens(scanner):
    """Tokens entirely within toxic ranges produce a positive score."""
    # All tokens in the range 100-110 (full-severity range)
    token_ids = [100, 101, 102, 103, 104, 105]
    score = scanner.toxic_score(token_ids)
    assert score > 0.0


# ---------------------------------------------------------------------------
# Test 10: scan_text returns ScanResult
# ---------------------------------------------------------------------------


def test_scan_text_returns_scan_result(scanner):
    """scan_text returns a ScanResult instance."""
    result = scanner.scan_text("Hello, world!")
    assert isinstance(result, ScanResult)


# ---------------------------------------------------------------------------
# Test 11: scan_text with long text sets anomalous_length=True
# ---------------------------------------------------------------------------


def test_scan_text_anomalous_length(scanner):
    """scan_text sets anomalous_length=True when text exceeds max_expected_length."""
    long_text = "a" * 600  # longer than default max_expected_length of 512
    result = scanner.scan_text(long_text)
    assert result.anomalous_length is True
    assert result.length == 600


# ---------------------------------------------------------------------------
# Test 12: scan_tokens returns ScanResult
# ---------------------------------------------------------------------------


def test_scan_tokens_returns_scan_result(scanner):
    """scan_tokens returns a ScanResult instance."""
    token_ids = [200, 300, 400, 50, 60]
    result = scanner.scan_tokens(token_ids)
    assert isinstance(result, ScanResult)


# ---------------------------------------------------------------------------
# Test 13: ScanResult has all required fields
# ---------------------------------------------------------------------------


def test_scan_result_has_required_fields():
    """ScanResult dataclass exposes all six required fields."""
    required = {"has_pii", "has_toxic", "anomalous_length", "pii_types", "toxic_score", "length"}
    actual = {f.name for f in fields(ScanResult)}
    assert required.issubset(actual)


# ---------------------------------------------------------------------------
# Test 14: detect_pii_text finds credit card pattern
# ---------------------------------------------------------------------------


def test_detect_pii_text_finds_credit_card(scanner):
    """detect_pii_text detects a credit card number in 1234-5678-9012-3456 format."""
    text = "Card on file: 1234-5678-9012-3456. Please verify."
    found, pii_types = scanner.detect_pii_text(text)
    assert found is True
    assert "credit_card" in pii_types
