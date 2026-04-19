"""Integration tests for the constitutional principles scorer registry entry."""

from __future__ import annotations

from src.safety import HARM_CLASSIFIER_REGISTRY
from src.safety.constitutional_principles_scorer import (
    ConstitutionalPrinciplesScorer,
    ConstitutionalReport,
)


def test_registry_has_constitutional_key() -> None:
    assert "constitutional" in HARM_CLASSIFIER_REGISTRY
    assert HARM_CLASSIFIER_REGISTRY["constitutional"] is ConstitutionalPrinciplesScorer


def test_prior_registry_entries_intact() -> None:
    # Entries registered before constitutional must still be present.
    assert "harm_taxonomy" in HARM_CLASSIFIER_REGISTRY
    assert "refusal" in HARM_CLASSIFIER_REGISTRY


def test_registered_class_scores_sample() -> None:
    cls = HARM_CLASSIFIER_REGISTRY["constitutional"]
    scorer = cls()
    report = scorer.score(
        "Here is a helpful, honest answer. Please be safe. TL;DR: consult a doctor."
    )
    assert isinstance(report, ConstitutionalReport)
    assert 0.0 <= report.overall <= 1.0
    principles = {p.principle for p in report.principle_scores}
    assert {"helpful", "honest", "harmless", "respectful", "concise"}.issubset(principles)
