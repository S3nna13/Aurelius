"""Tests for src.safety.content_classifier — ≥28 test cases."""

from __future__ import annotations

import pytest

from src.safety.content_classifier import (
    ClassificationResult,
    ContentCategory,
    ContentClassifier,
    CONTENT_CLASSIFIER_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def clf() -> ContentClassifier:
    return ContentClassifier()


# ---------------------------------------------------------------------------
# Safe text
# ---------------------------------------------------------------------------

def test_safe_text_returns_safe_category(clf: ContentClassifier) -> None:
    result = clf.classify("The sky is blue and the weather is nice.")
    assert result.category == ContentCategory.SAFE


def test_safe_text_confidence_is_1(clf: ContentClassifier) -> None:
    result = clf.classify("Hello, how are you today?")
    assert result.confidence == pytest.approx(1.0)


def test_safe_text_evidence_empty(clf: ContentClassifier) -> None:
    result = clf.classify("I love mathematics and science.")
    assert result.evidence == []


def test_empty_text_returns_safe(clf: ContentClassifier) -> None:
    result = clf.classify("")
    assert result.category == ContentCategory.SAFE
    assert result.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Violence detection
# ---------------------------------------------------------------------------

def test_violence_keyword_detected_kill(clf: ContentClassifier) -> None:
    result = clf.classify("I want to kill the process on my computer.")
    assert result.category == ContentCategory.VIOLENCE


def test_violence_keyword_detected_weapon(clf: ContentClassifier) -> None:
    result = clf.classify("The museum has an old weapon on display.")
    assert result.category == ContentCategory.VIOLENCE


def test_violence_keyword_detected_bomb(clf: ContentClassifier) -> None:
    result = clf.classify("The bomb squad arrived at the scene.")
    assert result.category == ContentCategory.VIOLENCE


def test_violence_keyword_in_evidence(clf: ContentClassifier) -> None:
    result = clf.classify("They planned an attack.")
    assert "attack" in result.evidence


# ---------------------------------------------------------------------------
# Hate detection
# ---------------------------------------------------------------------------

def test_hate_keyword_detected_racist(clf: ContentClassifier) -> None:
    result = clf.classify("That was a racist remark.")
    assert result.category == ContentCategory.HATE


def test_hate_keyword_detected_bigot(clf: ContentClassifier) -> None:
    result = clf.classify("He is a bigot who refuses to listen.")
    assert result.category == ContentCategory.HATE


def test_hate_keyword_in_evidence(clf: ContentClassifier) -> None:
    result = clf.classify("The supremacist ideology is dangerous.")
    assert "supremacist" in result.evidence


# ---------------------------------------------------------------------------
# Self-harm detection
# ---------------------------------------------------------------------------

def test_self_harm_detected(clf: ContentClassifier) -> None:
    result = clf.classify("Information about suicide prevention hotlines.")
    assert result.category == ContentCategory.SELF_HARM


def test_self_harm_overdose(clf: ContentClassifier) -> None:
    result = clf.classify("An overdose can be life-threatening.")
    assert result.category == ContentCategory.SELF_HARM


# ---------------------------------------------------------------------------
# Illegal detection
# ---------------------------------------------------------------------------

def test_illegal_cocaine_detected(clf: ContentClassifier) -> None:
    result = clf.classify("Cocaine is a controlled substance.")
    assert result.category == ContentCategory.ILLEGAL


def test_illegal_trafficking_detected(clf: ContentClassifier) -> None:
    result = clf.classify("Human trafficking is a global crisis.")
    assert result.category == ContentCategory.ILLEGAL


# ---------------------------------------------------------------------------
# Spam detection
# ---------------------------------------------------------------------------

def test_spam_detected_click_here(clf: ContentClassifier) -> None:
    result = clf.classify("Click here to claim your prize!")
    assert result.category == ContentCategory.SPAM


def test_spam_detected_win_money(clf: ContentClassifier) -> None:
    result = clf.classify("Win money by participating in this survey!")
    assert result.category == ContentCategory.SPAM


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

def test_confidence_one_match(clf: ContentClassifier) -> None:
    # 1 match / 3 = 0.333...
    result = clf.classify("kill")
    assert result.confidence == pytest.approx(1 / 3.0)


def test_confidence_three_matches_caps_at_1(clf: ContentClassifier) -> None:
    # 3 matches: kill, murder, weapon -> min(3/3, 1.0) = 1.0
    result = clf.classify("kill murder weapon in the story.")
    assert result.confidence == pytest.approx(1.0)


def test_confidence_capped_at_1(clf: ContentClassifier) -> None:
    # All 7 violence keywords -> min(7/3, 1.0) = 1.0
    result = clf.classify("kill murder weapon bomb attack shoot stab")
    assert result.confidence == pytest.approx(1.0)
    assert result.confidence <= 1.0


def test_confidence_two_matches(clf: ContentClassifier) -> None:
    result = clf.classify("cocaine and heroin are illegal drugs")
    assert result.confidence == pytest.approx(2 / 3.0)


# ---------------------------------------------------------------------------
# Evidence: sorted and unique
# ---------------------------------------------------------------------------

def test_evidence_is_sorted(clf: ContentClassifier) -> None:
    result = clf.classify("weapon kill attack")
    assert result.evidence == sorted(result.evidence)


def test_evidence_is_unique(clf: ContentClassifier) -> None:
    # Even if keyword appears multiple times in text, evidence should be unique
    result = clf.classify("kill kill kill kill")
    assert len(result.evidence) == len(set(result.evidence))


def test_evidence_contains_matched_keywords(clf: ContentClassifier) -> None:
    result = clf.classify("shoot and stab were discussed.")
    assert "shoot" in result.evidence
    assert "stab" in result.evidence


# ---------------------------------------------------------------------------
# batch_classify
# ---------------------------------------------------------------------------

def test_batch_classify_length(clf: ContentClassifier) -> None:
    texts = ["Hello world", "kill and murder", "cocaine trafficking"]
    results = clf.batch_classify(texts)
    assert len(results) == len(texts)


def test_batch_classify_returns_list(clf: ContentClassifier) -> None:
    results = clf.batch_classify(["safe text"])
    assert isinstance(results, list)
    assert isinstance(results[0], ClassificationResult)


def test_batch_classify_empty_list(clf: ContentClassifier) -> None:
    results = clf.batch_classify([])
    assert results == []


def test_batch_classify_correct_categories(clf: ContentClassifier) -> None:
    texts = ["nice day", "kill weapon"]
    results = clf.batch_classify(texts)
    assert results[0].category == ContentCategory.SAFE
    assert results[1].category == ContentCategory.VIOLENCE


# ---------------------------------------------------------------------------
# safe_to_serve
# ---------------------------------------------------------------------------

def test_safe_to_serve_safe_category_true(clf: ContentClassifier) -> None:
    result = clf.classify("How do I cook pasta?")
    assert clf.safe_to_serve(result) is True


def test_safe_to_serve_high_confidence_false(clf: ContentClassifier) -> None:
    result = clf.classify("kill murder weapon bomb attack shoot stab")
    # category=VIOLENCE, confidence=1.0 >= 0.5 -> not safe
    assert clf.safe_to_serve(result) is False


def test_safe_to_serve_low_confidence_true(clf: ContentClassifier) -> None:
    # 1 match -> confidence = 1/3 < 0.5 -> safe to serve
    result = clf.classify("kill")
    assert clf.safe_to_serve(result) is True


def test_safe_to_serve_custom_threshold(clf: ContentClassifier) -> None:
    result = clf.classify("cocaine and heroin")  # confidence = 2/3 ≈ 0.667
    assert clf.safe_to_serve(result, threshold=0.8) is True  # 0.667 < 0.8
    assert clf.safe_to_serve(result, threshold=0.5) is False  # 0.667 >= 0.5


# ---------------------------------------------------------------------------
# ClassificationResult frozen
# ---------------------------------------------------------------------------

def test_classification_result_is_frozen() -> None:
    result = ClassificationResult(
        text="hello",
        category=ContentCategory.SAFE,
        confidence=1.0,
        evidence=[],
    )
    with pytest.raises((AttributeError, TypeError)):
        result.confidence = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# text field preserved
# ---------------------------------------------------------------------------

def test_classification_result_text_preserved(clf: ContentClassifier) -> None:
    text = "Some text about a weapon."
    result = clf.classify(text)
    assert result.text == text


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_default() -> None:
    assert "default" in CONTENT_CLASSIFIER_REGISTRY


def test_registry_default_is_content_classifier_class() -> None:
    cls = CONTENT_CLASSIFIER_REGISTRY["default"]
    instance = cls()
    assert isinstance(instance, ContentClassifier)
