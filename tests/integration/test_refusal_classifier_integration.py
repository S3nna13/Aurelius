"""Integration tests for the ``refusal`` harm-classifier registration."""

from __future__ import annotations

import pytest


def test_registry_contains_refusal_entry() -> None:
    import src.safety as safety

    assert "refusal" in safety.HARM_CLASSIFIER_REGISTRY
    cls = safety.HARM_CLASSIFIER_REGISTRY["refusal"]
    assert cls is safety.RefusalClassifier


def test_prior_registry_entries_still_present() -> None:
    import src.safety as safety

    # Previously registered HARM classifier.
    assert "harm_taxonomy" in safety.HARM_CLASSIFIER_REGISTRY
    assert (
        safety.HARM_CLASSIFIER_REGISTRY["harm_taxonomy"]
        is safety.HarmTaxonomyClassifier
    )

    # Previously registered SAFETY filters — sanity-check a couple.
    assert "jailbreak" in safety.SAFETY_FILTER_REGISTRY
    assert "prompt_injection" in safety.SAFETY_FILTER_REGISTRY
    assert "pii" in safety.SAFETY_FILTER_REGISTRY


def test_registered_class_is_constructible() -> None:
    import src.safety as safety

    Clf = safety.HARM_CLASSIFIER_REGISTRY["refusal"]
    inst = Clf()
    assert hasattr(inst, "classify")
    assert hasattr(inst, "is_refusal")


def test_three_sample_classification() -> None:
    import src.safety as safety

    Clf = safety.HARM_CLASSIFIER_REGISTRY["refusal"]
    clf = Clf()

    # 1) Refusal.
    refusal = clf.classify("I apologize, but I cannot help with that request.")
    assert refusal.is_refusal is True
    assert refusal.score >= 0.5
    assert len(refusal.signals) >= 1

    # 2) Compliance.
    compliance = clf.classify(
        "Sure! Here is the pseudocode for binary search on a sorted array."
    )
    assert compliance.is_refusal is False
    assert compliance.score < 0.5

    # 3) Ambiguous — short answer, no canonical phrase, long question.
    question = (
        "Could you explain in detail how the transformer's attention "
        "mechanism computes its output?"  # substantive
    )
    ambiguous = clf.classify("Maybe later.", question=question)
    # Ambiguous case triggers the length heuristic but not enough to flag.
    assert ambiguous.is_refusal is False
    assert "length_heuristic" in ambiguous.signals
    assert 0.0 < ambiguous.score < 0.5


def test_refusal_score_has_expected_fields() -> None:
    import src.safety as safety

    Clf = safety.HARM_CLASSIFIER_REGISTRY["refusal"]
    res = Clf().classify("I must decline.")
    assert hasattr(res, "is_refusal")
    assert hasattr(res, "score")
    assert hasattr(res, "signals")
    assert isinstance(res.is_refusal, bool)
    assert isinstance(res.score, float)
    assert isinstance(res.signals, list)
