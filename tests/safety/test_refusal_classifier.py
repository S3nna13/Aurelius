"""Unit tests for ``src.safety.refusal_classifier``."""

from __future__ import annotations

import time

import pytest

from src.safety.refusal_classifier import (
    REFUSAL_PHRASES,
    RefusalClassifier,
    RefusalScore,
)


@pytest.fixture()
def clf() -> RefusalClassifier:
    return RefusalClassifier()


# --------------------------------------------------------------------------- #
# Catalog / structural invariants
# --------------------------------------------------------------------------- #


def test_refusal_phrase_catalog_size() -> None:
    assert len(REFUSAL_PHRASES) >= 30
    assert all(isinstance(p, str) and p.strip() for p in REFUSAL_PHRASES)


def test_weight_sum_is_unity() -> None:
    c = RefusalClassifier
    total = c._W_CANONICAL + c._W_APOLOGY + c._W_POLICY + c._W_LENGTH
    assert abs(total - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Core behaviour
# --------------------------------------------------------------------------- #


def test_obvious_refusal_flags_true(clf: RefusalClassifier) -> None:
    res = clf.classify("I cannot help with that request.")
    assert isinstance(res, RefusalScore)
    assert res.is_refusal is True
    assert res.score >= 0.5
    assert "canonical_phrase" in res.signals


def test_obvious_compliance_is_not_refusal(clf: RefusalClassifier) -> None:
    res = clf.classify(
        "Here's the answer: you sort a Python list with list.sort() or sorted()."
    )
    assert res.is_refusal is False
    assert res.score < 0.5
    assert "canonical_phrase" not in res.signals


def test_apology_negation_pattern_flags(clf: RefusalClassifier) -> None:
    # Use phrasing that isn't itself a canonical phrase so we can assert the
    # apology signal fires independently.
    res = clf.classify("I apologize, but I'm not able to provide that.")
    assert res.is_refusal is True
    assert "apology_negation" in res.signals


def test_policy_reference_signal_fires(clf: RefusalClassifier) -> None:
    res = clf.classify("That request would violate my guidelines.")
    assert "policy_reference" in res.signals


def test_safety_category_mentions_count_as_policy() -> None:
    clf = RefusalClassifier()
    # "illegal" / "harmful" should trip the policy-reference family.
    res = clf.classify("That would be illegal and harmful.")
    assert "policy_reference" in res.signals


# --------------------------------------------------------------------------- #
# Length heuristic
# --------------------------------------------------------------------------- #


def test_length_heuristic_with_short_answer_to_long_question() -> None:
    clf = RefusalClassifier()
    question = (
        "Can you walk me through the detailed reasoning behind quicksort's "
        "average-case complexity?"  # ~100 chars
    )
    res = clf.classify("No.", question=question)
    assert "length_heuristic" in res.signals
    # Length alone is only 0.05 and should NOT flag as refusal by itself.
    assert res.is_refusal is False


def test_length_heuristic_does_not_fire_without_question() -> None:
    clf = RefusalClassifier()
    res = clf.classify("No.")
    assert "length_heuristic" not in res.signals


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_empty_response_is_not_refusal() -> None:
    clf = RefusalClassifier()
    res = clf.classify("")
    assert res.is_refusal is False
    assert res.signals == []
    assert res.score == 0.0


def test_whitespace_only_response_is_not_refusal() -> None:
    clf = RefusalClassifier()
    res = clf.classify("   \n\t  ")
    assert res.is_refusal is False
    assert res.signals == []


def test_case_insensitive_match() -> None:
    clf = RefusalClassifier()
    res_lower = clf.classify("i cannot help with that")
    res_upper = clf.classify("I CANNOT HELP WITH THAT")
    res_mixed = clf.classify("I CaNnOt HeLp with that")
    assert res_lower.is_refusal is True
    assert res_upper.is_refusal is True
    assert res_mixed.is_refusal is True
    assert res_lower.score == res_upper.score == res_mixed.score


def test_custom_phrase_detection() -> None:
    clf = RefusalClassifier(custom_phrases=["zorp forbids this"])
    res = clf.classify("Zorp forbids this action.")
    assert res.is_refusal is True
    assert "canonical_phrase" in res.signals


def test_custom_phrases_rejects_bare_string() -> None:
    with pytest.raises(TypeError):
        RefusalClassifier(custom_phrases="not a list")  # type: ignore[arg-type]


def test_determinism_across_invocations() -> None:
    clf = RefusalClassifier()
    text = "I apologize, but I cannot assist with that illegal request."
    r1 = clf.classify(text)
    r2 = clf.classify(text)
    r3 = clf.classify(text)
    assert r1 == r2 == r3
    assert r1.signals == r2.signals == r3.signals


def test_unicode_nfkc_normalisation() -> None:
    clf = RefusalClassifier()
    # Fullwidth 'I cannot help' — NFKC collapses to ASCII.
    fullwidth = "Ｉ ｃａｎｎｏｔ ｈｅｌｐ with that."
    res = clf.classify(fullwidth)
    assert res.is_refusal is True

    # Zero-width injection inside a canonical phrase should still match:
    # the ZWSP sits between two already-separated tokens and is stripped
    # during NFKC/Cf cleanup, leaving the phrase intact.
    zw = "I \u200bcannot help with that."
    res_zw = clf.classify(zw)
    assert res_zw.is_refusal is True


def test_large_input_performance() -> None:
    clf = RefusalClassifier()
    # 100 KB filler ending in a canonical phrase.
    filler = "lorem ipsum dolor sit amet " * 4000
    payload = filler + " I cannot help with that request."
    assert len(payload) >= 100 * 1024
    t0 = time.perf_counter()
    res = clf.classify(payload)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"classify took {elapsed:.3f}s on 100KB"
    assert res.is_refusal is True


# --------------------------------------------------------------------------- #
# Threshold semantics
# --------------------------------------------------------------------------- #


def test_threshold_inclusive_lower_bound() -> None:
    # Canonical phrase only -> score == 0.60.
    # With threshold exactly 0.60, it should classify as refusal (inclusive).
    clf_tight = RefusalClassifier(threshold=0.60)
    res = clf_tight.classify("I cannot help.")
    assert res.score == pytest.approx(0.60, abs=1e-9)
    assert res.is_refusal is True

    # With threshold strictly above 0.60, same score should NOT classify.
    clf_strict = RefusalClassifier(threshold=0.61)
    res2 = clf_strict.classify("I cannot help.")
    assert res2.score == pytest.approx(0.60, abs=1e-9)
    assert res2.is_refusal is False


def test_threshold_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        RefusalClassifier(threshold=-0.1)
    with pytest.raises(ValueError):
        RefusalClassifier(threshold=1.5)


def test_non_string_response_rejected() -> None:
    clf = RefusalClassifier()
    with pytest.raises(TypeError):
        clf.classify(123)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Over-refusal guard — benign answer should not flag
# --------------------------------------------------------------------------- #


def test_benign_explanation_does_not_flag() -> None:
    clf = RefusalClassifier()
    benign = (
        "Sure! To sort a list in Python you can call list.sort() for an "
        "in-place sort, or sorted(iterable) to get a new sorted list. Both "
        "accept a key= argument for custom comparators."
    )
    res = clf.classify(benign)
    assert res.is_refusal is False
    assert "canonical_phrase" not in res.signals
    assert "apology_negation" not in res.signals


def test_is_refusal_convenience_wrapper() -> None:
    clf = RefusalClassifier()
    assert clf.is_refusal("I must decline this request.") is True
    assert clf.is_refusal("Here you go: the capital of France is Paris.") is False
