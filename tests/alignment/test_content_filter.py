"""Tests for src/alignment/content_filter.py — 12 tests."""

from __future__ import annotations

import time

from src.alignment.content_filter import (
    AuditLog,
    CategoryScorer,
    ContentFilter,
    EnsembleFilter,
    FilterConfig,
    FilterDecision,
    compute_filter_metrics,
    detect_pii,
)

# ---------------------------------------------------------------------------
# 1. FilterConfig defaults
# ---------------------------------------------------------------------------


def test_filter_config_defaults():
    cfg = FilterConfig()
    assert cfg.categories == ["harmful", "explicit", "hate", "spam", "pii"]
    assert cfg.thresholds == {
        "harmful": 0.7,
        "explicit": 0.8,
        "hate": 0.7,
        "spam": 0.5,
        "pii": 0.9,
    }
    assert cfg.ensemble_method == "max"
    assert cfg.n_ensemble == 3
    assert cfg.log_decisions is True
    assert cfg.appeal_enabled is False


# ---------------------------------------------------------------------------
# 2. FilterDecision fields
# ---------------------------------------------------------------------------


def test_filter_decision_fields():
    now = time.time()
    d = FilterDecision(
        text="hello",
        is_safe=True,
        triggered_categories=[],
        scores={"harmful": 0.1},
        confidence=1.0,
        timestamp=now,
    )
    assert d.text == "hello"
    assert d.is_safe is True
    assert d.triggered_categories == []
    assert d.scores == {"harmful": 0.1}
    assert d.confidence == 1.0
    assert d.timestamp == now


# ---------------------------------------------------------------------------
# 3. CategoryScorer output keys
# ---------------------------------------------------------------------------


def test_category_scorer_output_keys():
    cats = ["harmful", "explicit", "hate", "spam", "pii"]
    scorer = CategoryScorer(d_model=16, categories=cats)
    result = scorer("some test text")
    assert set(result.keys()) == set(cats)


# ---------------------------------------------------------------------------
# 4. CategoryScorer score range [0, 1]
# ---------------------------------------------------------------------------


def test_category_scorer_score_range():
    cats = ["harmful", "explicit", "hate"]
    scorer = CategoryScorer(d_model=16, categories=cats)
    result = scorer("Another piece of text for testing")
    for cat, score in result.items():
        assert 0.0 <= score <= 1.0, f"{cat} score {score} out of range"


# ---------------------------------------------------------------------------
# 5. EnsembleFilter output keys
# ---------------------------------------------------------------------------


def test_ensemble_filter_output_keys():
    cfg = FilterConfig()
    ef = EnsembleFilter(cfg, d_model=16)
    result = ef.score("test text")
    assert set(result.keys()) == set(cfg.categories)


# ---------------------------------------------------------------------------
# 6. EnsembleFilter ensemble_method=max
# ---------------------------------------------------------------------------


def test_ensemble_filter_ensemble_method_max():
    cfg = FilterConfig(ensemble_method="max", n_ensemble=3)
    ef = EnsembleFilter(cfg, d_model=16)
    result = ef.score("test ensemble max")
    # All scores must be in valid range (sanity for max aggregation)
    for score in result.values():
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 7. ContentFilter returns a FilterDecision
# ---------------------------------------------------------------------------


def test_content_filter_returns_decision():
    cfg = FilterConfig()
    cf = ContentFilter(cfg, d_model=16)
    decision = cf.filter("Hello, world!")
    assert isinstance(decision, FilterDecision)


# ---------------------------------------------------------------------------
# 8. ContentFilter is_safe is a bool
# ---------------------------------------------------------------------------


def test_content_filter_is_safe_type():
    cfg = FilterConfig()
    cf = ContentFilter(cfg, d_model=16)
    decision = cf.filter("Some neutral text.")
    assert isinstance(decision.is_safe, bool)


# ---------------------------------------------------------------------------
# 9. AuditLog add and stats
# ---------------------------------------------------------------------------


def test_audit_log_add_and_stats():
    log = AuditLog()
    assert log.statistics()["n_decisions"] == 0

    d = FilterDecision(
        text="hi",
        is_safe=True,
        triggered_categories=[],
        scores={"harmful": 0.1, "spam": 0.2},
        confidence=1.0,
        timestamp=time.time(),
    )
    log.add(d)

    stats = log.statistics()
    assert stats["n_decisions"] == 1
    assert stats["safe_rate"] == 1.0
    assert "category_rates" in stats


# ---------------------------------------------------------------------------
# 10. detect_pii detects email
# ---------------------------------------------------------------------------


def test_detect_pii_email():
    found = detect_pii("Contact me at user@example.com for details.")
    assert "email" in found


# ---------------------------------------------------------------------------
# 11. detect_pii returns empty for clean text
# ---------------------------------------------------------------------------


def test_detect_pii_no_pii():
    found = detect_pii("The quick brown fox jumps over the lazy dog.")
    assert found == []


# ---------------------------------------------------------------------------
# 12. compute_filter_metrics returns required keys
# ---------------------------------------------------------------------------


def test_compute_filter_metrics_keys():
    decisions = [
        FilterDecision(
            text="a",
            is_safe=False,
            triggered_categories=["harmful"],
            scores={"harmful": 0.9},
            confidence=0.8,
            timestamp=time.time(),
        ),
        FilterDecision(
            text="b",
            is_safe=True,
            triggered_categories=[],
            scores={"harmful": 0.1},
            confidence=1.0,
            timestamp=time.time(),
        ),
    ]
    ground_truth = [True, False]
    metrics = compute_filter_metrics(decisions, ground_truth)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "accuracy" in metrics
