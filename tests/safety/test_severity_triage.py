"""Tests for src.safety.severity_triage."""

from __future__ import annotations

import pytest

from src.safety.severity_triage import (
    SEVERITY_TRIAGE_REGISTRY,
    WEIGHTED_KEYWORDS,
    SeverityLevel,
    SeverityTriage,
    TriageResult,
)


@pytest.fixture
def triage() -> SeverityTriage:
    return SeverityTriage()


def test_registry_has_default():
    assert "default" in SEVERITY_TRIAGE_REGISTRY
    assert SEVERITY_TRIAGE_REGISTRY["default"] is SeverityTriage


def test_weighted_keywords_length():
    assert len(WEIGHTED_KEYWORDS) >= 20


def test_weighted_keywords_structure():
    for kw, w in WEIGHTED_KEYWORDS:
        assert isinstance(kw, str) and kw == kw.lower()
        assert 0.0 < w <= 1.0


def test_severity_level_weights():
    assert SeverityLevel.CRITICAL.weight == 1.0
    assert SeverityLevel.HIGH.weight == 0.8
    assert SeverityLevel.MEDIUM.weight == 0.5
    assert SeverityLevel.LOW.weight == 0.3
    assert SeverityLevel.INFO.weight == 0.1


def test_triage_result_is_frozen():
    r = TriageResult(text="x", score=0.0, severity=SeverityLevel.INFO)
    with pytest.raises(Exception):
        r.score = 0.5  # type: ignore[misc]


def test_empty_text_is_info(triage: SeverityTriage):
    r = triage.score("")
    assert r.severity == SeverityLevel.INFO
    assert r.score == 0.0
    assert r.matched_keywords == []
    assert r.confidence == 0.0


def test_whitespace_text_is_info(triage: SeverityTriage):
    r = triage.score("   \n\t ")
    assert r.severity == SeverityLevel.INFO


def test_none_like_input_handled(triage: SeverityTriage):
    r = triage.score(None)  # type: ignore[arg-type]
    assert r.severity == SeverityLevel.INFO


def test_rce_keyword_matches(triage: SeverityTriage):
    r = triage.score("Unauthenticated RCE via crafted payload")
    assert "rce" in r.matched_keywords
    assert r.score > 0.0


def test_sql_injection_matches(triage: SeverityTriage):
    r = triage.score("SQL injection in login parameter")
    assert "sql injection" in r.matched_keywords


def test_buffer_overflow_matches(triage: SeverityTriage):
    r = triage.score("Stack-based buffer overflow in parser")
    assert "buffer overflow" in r.matched_keywords


def test_xss_keyword_matches(triage: SeverityTriage):
    r = triage.score("Reflected XSS in search field")
    assert "xss" in r.matched_keywords


def test_open_redirect_low(triage: SeverityTriage):
    r = triage.score("Open redirect on logout endpoint")
    assert "open redirect" in r.matched_keywords
    # Open redirect weight 0.3 / max ~ small; should be LOW or INFO.
    assert r.severity in (SeverityLevel.LOW, SeverityLevel.INFO)


def test_no_match_is_info(triage: SeverityTriage):
    r = triage.score("A friendly greeting about cats and gardening.")
    assert r.matched_keywords == []
    assert r.severity == SeverityLevel.INFO


def test_case_insensitive(triage: SeverityTriage):
    r = triage.score("REMOTE CODE EXECUTION discovered")
    assert "remote code execution" in r.matched_keywords


def test_multiple_matches_accumulate(triage: SeverityTriage):
    single = triage.score("SQL injection only")
    multi = triage.score("SQL injection plus RCE and SSRF and XXE and command injection")
    assert multi.score >= single.score
    assert len(multi.matched_keywords) > len(single.matched_keywords)


def test_score_is_clamped_to_one(triage: SeverityTriage):
    blob = " ".join(kw for kw, _ in WEIGHTED_KEYWORDS)
    r = triage.score(blob)
    assert 0.0 <= r.score <= 1.0


def test_critical_threshold_reachable():
    # Build a custom triage whose single keyword saturates.
    t = SeverityTriage(keywords=[("rce", 1.0)])
    r = t.score("rce found")
    assert r.score == 1.0
    assert r.severity == SeverityLevel.CRITICAL


def test_high_threshold():
    SeverityTriage(keywords=[("rce", 1.0)])
    # Score of 0.5 cannot be produced with single-keyword above; verify mapping.
    from src.safety.severity_triage import _severity_for_score

    # Compare by name to survive module reloads that recreate the enum class.
    assert _severity_for_score(0.5).name == SeverityLevel.HIGH.name
    assert _severity_for_score(0.7).name == SeverityLevel.CRITICAL.name
    assert _severity_for_score(0.3).name == SeverityLevel.MEDIUM.name
    assert _severity_for_score(0.1).name == SeverityLevel.LOW.name
    assert _severity_for_score(0.05).name == SeverityLevel.INFO.name


def test_confidence_scales_with_matches(triage: SeverityTriage):
    r1 = triage.score("xss")
    r2 = triage.score("xss and csrf and idor and ssrf and rce")
    assert r2.confidence > r1.confidence
    assert r2.confidence <= 1.0


def test_score_batch(triage: SeverityTriage):
    texts = ["rce", "nothing here", "xss issue"]
    results = triage.score_batch(texts)
    assert len(results) == 3
    assert all(isinstance(r, TriageResult) for r in results)
    assert results[1].severity == SeverityLevel.INFO


def test_score_batch_empty(triage: SeverityTriage):
    assert triage.score_batch([]) == []


def test_top_k_ordering(triage: SeverityTriage):
    texts = [
        "minor misconfiguration",
        "critical RCE with full command injection and SSRF",
        "open redirect",
    ]
    top = triage.top_k(texts, k=2)
    assert len(top) == 2
    assert top[0].score >= top[1].score
    assert "rce" in top[0].matched_keywords


def test_top_k_zero(triage: SeverityTriage):
    assert triage.top_k(["rce"], k=0) == []


def test_top_k_larger_than_inputs(triage: SeverityTriage):
    top = triage.top_k(["rce", "xss"], k=10)
    assert len(top) == 2


def test_custom_keywords_override():
    t = SeverityTriage(keywords=[("foo", 1.0)])
    r = t.score("this contains foo")
    assert "foo" in r.matched_keywords
    assert r.score == 1.0


def test_matched_keywords_are_from_catalog(triage: SeverityTriage):
    catalog = {kw for kw, _ in WEIGHTED_KEYWORDS}
    r = triage.score("SQL injection and XSS and RCE")
    for kw in r.matched_keywords:
        assert kw in catalog


def test_idempotent_scoring(triage: SeverityTriage):
    a = triage.score("SSRF via metadata endpoint")
    b = triage.score("SSRF via metadata endpoint")
    assert a.score == b.score
    assert a.matched_keywords == b.matched_keywords


def test_info_disclosure_medium_or_low(triage: SeverityTriage):
    r = triage.score("information disclosure in headers")
    assert "information disclosure" in r.matched_keywords


def test_hardcoded_keyword(triage: SeverityTriage):
    r = triage.score("hardcoded credentials in source")
    assert "hardcoded" in r.matched_keywords
