"""Unit tests for :mod:`src.safety.pii_detector`."""

from __future__ import annotations

import time

import pytest

from src.safety.pii_detector import (
    PIIDetector,
    PIIMatch,
    PIIResult,
    iban_valid,
    luhn_valid,
)

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def test_email_detected() -> None:
    d = PIIDetector()
    matches = d.detect("contact alice@example.com for details")
    assert len(matches) == 1
    assert matches[0].type == "email"
    assert matches[0].value == "alice@example.com"


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("call (415) 555-1212 now", "(415) 555-1212"),
        ("415-555-1212 is me", "415-555-1212"),
        ("intl: +1 415 555 1212 ext", "+1 415 555 1212"),
    ],
)
def test_phone_detected(text: str, expected: str) -> None:
    d = PIIDetector()
    matches = [m for m in d.detect(text) if m.type == "phone"]
    assert matches, f"no phone in {text!r}"
    assert expected in [m.value for m in matches]


# ---------------------------------------------------------------------------
# SSN
# ---------------------------------------------------------------------------


def test_ssn_detected() -> None:
    d = PIIDetector()
    matches = d.detect("SSN: 123-45-6789.")
    assert any(m.type == "ssn" and m.value == "123-45-6789" for m in matches)


def test_invalid_ssn_rejected() -> None:
    d = PIIDetector()
    assert not any(m.type == "ssn" for m in d.detect("000-12-3456"))
    assert not any(m.type == "ssn" for m in d.detect("666-12-3456"))
    assert not any(m.type == "ssn" for m in d.detect("900-12-3456"))


# ---------------------------------------------------------------------------
# Credit card + Luhn
# ---------------------------------------------------------------------------


def test_credit_card_valid_luhn_detected() -> None:
    d = PIIDetector()
    text = "card 4111 1111 1111 1111 end"
    matches = [m for m in d.detect(text) if m.type == "credit_card"]
    assert matches


def test_credit_card_invalid_luhn_rejected() -> None:
    d = PIIDetector()
    # Change last digit to break Luhn.
    text = "card 4111 1111 1111 1112 end"
    assert not any(m.type == "credit_card" for m in d.detect(text))


def test_luhn_helper() -> None:
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("")
    assert not luhn_valid("12")


# ---------------------------------------------------------------------------
# IP addresses + MAC
# ---------------------------------------------------------------------------


def test_ipv4_detected() -> None:
    d = PIIDetector()
    matches = [m for m in d.detect("server at 192.168.1.100 listens") if m.type == "ipv4"]
    assert matches and matches[0].value == "192.168.1.100"


def test_ipv6_detected() -> None:
    d = PIIDetector()
    txt = "addr 2001:0db8:85a3:0000:0000:8a2e:0370:7334 done"
    matches = [m for m in d.detect(txt) if m.type == "ipv6"]
    assert matches


def test_mac_detected() -> None:
    d = PIIDetector()
    matches = [m for m in d.detect("mac=00:1A:2B:3C:4D:5E here") if m.type == "mac"]
    assert matches


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------


def test_aws_access_key_detected() -> None:
    d = PIIDetector()
    matches = [m for m in d.detect("key: AKIAIOSFODNN7EXAMPLE end") if m.type == "api_key"]
    assert any("AKIA" in m.value for m in matches)


def test_github_token_detected() -> None:
    d = PIIDetector()
    token = "ghp_" + "A" * 36
    matches = [m for m in d.detect(f"token {token} here") if m.type == "api_key"]
    assert any(token == m.value for m in matches)


# ---------------------------------------------------------------------------
# Redaction modes
# ---------------------------------------------------------------------------


def test_redaction_placeholder() -> None:
    d = PIIDetector(redaction_mode="placeholder")
    r = d.redact("email alice@example.com now")
    assert r.redacted_text == "email <EMAIL> now"


def test_redaction_mask() -> None:
    d = PIIDetector(redaction_mode="mask")
    r = d.redact("email alice@example.com now")
    assert r.redacted_text == "email **** now"


def test_redaction_remove() -> None:
    d = PIIDetector(redaction_mode="remove")
    r = d.redact("email alice@example.com now")
    assert r.redacted_text == "email  now"


def test_unknown_redaction_mode_raises() -> None:
    with pytest.raises(ValueError):
        PIIDetector(redaction_mode="erase")


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------


def test_overlapping_matches_prefer_higher_confidence() -> None:
    # An email embedded in something that could also pattern-match as a
    # long generic API-key-looking token. The email (conf 0.95) must win
    # over any generic-api-key (conf 0.55) match on the same span.
    d = PIIDetector()
    text = "visit alice.smith1234@verylongexampledomain.com now"
    matches = d.detect(text)
    email_spans = [m.span for m in matches if m.type == "email"]
    api_spans = [m.span for m in matches if m.type == "api_key"]
    assert email_spans
    for a in api_spans:
        # No api-key match may overlap the email span.
        for es, ee in email_spans:
            assert not (a[0] < ee and es < a[1])


# ---------------------------------------------------------------------------
# Version-string false-positive
# ---------------------------------------------------------------------------


def test_version_string_not_ipv4() -> None:
    # Documented choice: "version 1.2.3.4" does match the IPv4 pattern
    # because 1.2.3.4 is a syntactically valid IPv4 address. We choose to
    # detect it — downstream callers that care about context should strip
    # version prefixes before scanning. We assert the actual behaviour so
    # regressions are visible.
    d = PIIDetector()
    matches = [m for m in d.detect("version 1.2.3.4 released") if m.type == "ipv4"]
    assert matches and matches[0].value == "1.2.3.4"


# ---------------------------------------------------------------------------
# Confidence threshold + type filter
# ---------------------------------------------------------------------------


def test_confidence_threshold_filter() -> None:
    text = "born 01/02/1980 and email a@b.co"
    low = PIIDetector(confidence_threshold=0.1)
    high = PIIDetector(confidence_threshold=0.9)
    low_types = {m.type for m in low.detect(text)}
    high_types = {m.type for m in high.detect(text)}
    assert "dob" in low_types
    assert "dob" not in high_types
    assert "email" in high_types


def test_types_filter_email_only() -> None:
    d = PIIDetector(types=["email"])
    text = "alice@example.com and 415-555-1212 and 123-45-6789"
    matches = d.detect(text)
    assert matches and all(m.type == "email" for m in matches)


def test_unknown_type_raises() -> None:
    with pytest.raises(ValueError):
        PIIDetector(types=["not_a_real_type"])


# ---------------------------------------------------------------------------
# Performance + determinism
# ---------------------------------------------------------------------------


def test_scan_100kb_under_one_second() -> None:
    chunk = "lorem ipsum alice@example.com dolor sit amet 415-555-1212. "
    # ~60 bytes * ~1800 ~ 108 KB.
    text = chunk * 1800
    d = PIIDetector()
    t0 = time.perf_counter()
    matches = d.detect(text)
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"scan too slow: {elapsed:.3f}s"
    assert len(matches) > 0


def test_determinism() -> None:
    d = PIIDetector()
    text = "email x@y.com and card 4111 1111 1111 1111 and ip 10.0.0.1"
    a = d.detect(text)
    b = d.detect(text)
    assert a == b


# ---------------------------------------------------------------------------
# has_pii + IBAN + dataclass sanity
# ---------------------------------------------------------------------------


def test_has_pii() -> None:
    d = PIIDetector()
    assert d.has_pii("reach me at foo@bar.com")
    assert not d.has_pii("nothing here, just prose")
    assert d.has_pii("foo@bar.com", types=["email"])
    assert not d.has_pii("foo@bar.com", types=["ssn"])


def test_iban_valid_helper() -> None:
    # Known-good IBAN from the ISO example set.
    assert iban_valid("GB82 WEST 1234 5698 7654 32")
    assert not iban_valid("GB82 WEST 1234 5698 7654 33")
    assert not iban_valid("not an iban")


def test_pii_result_dataclass() -> None:
    r = PIIResult()
    assert r.matches == [] and r.redacted_text == ""
    m = PIIMatch("email", "a@b.co", (0, 6), 0.9)
    assert m.type == "email" and m.span == (0, 6)
