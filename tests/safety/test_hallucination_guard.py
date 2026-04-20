"""Unit tests for HallucinationGuard."""

from __future__ import annotations

import random

import pytest

from src.safety.hallucination_guard import (
    Claim,
    Evidence,
    HallucinationGuard,
    ValidationResult,
    extract_key_terms,
    extract_numbers,
)


def test_all_validated_claims() -> None:
    guard = HallucinationGuard()
    claims = [
        Claim(text="Paris is the capital of France", source_model="m", confidence=0.9),
        Claim(text="The Eiffel Tower is in Paris", source_model="m", confidence=0.9),
    ]
    evidence = [
        Evidence(text="Paris, capital of France, lies on the Seine.", source="wiki"),
        Evidence(text="The Eiffel Tower stands in Paris, France.", source="wiki"),
    ]
    result = guard.validate(claims, evidence)
    assert isinstance(result, ValidationResult)
    assert result.is_valid is True
    assert len(result.validated_claims) == 2
    assert not result.unvalidated_claims
    assert not result.contradictions
    assert result.confidence > 0.0


def test_all_unvalidated_claims_with_no_overlap() -> None:
    guard = HallucinationGuard()
    claims = [Claim(text="Quantum gravity unifies everything", confidence=0.8)]
    evidence = [Evidence(text="Today's weather in Rome is sunny.")]
    result = guard.validate(claims, evidence)
    assert result.is_valid is False
    assert len(result.unvalidated_claims) == 1
    assert not result.validated_claims
    # Unvalidated decays confidence
    assert result.confidence <= 0.8 * 0.5 + 1e-9


def test_contradictory_evidence_via_negation() -> None:
    guard = HallucinationGuard()
    claims = [Claim(text="The server is running", confidence=1.0)]
    evidence = [Evidence(text="The server is not running right now")]
    result = guard.validate(claims, evidence)
    assert not result.validated_claims
    assert len(result.contradictions) == 1
    assert result.is_valid is False
    assert result.confidence == 0.0


def test_empty_inputs() -> None:
    guard = HallucinationGuard()
    # Empty both
    r = guard.validate([], [])
    assert r.is_valid is True
    assert not r.validated_claims and not r.unvalidated_claims
    # Claims but no evidence
    r2 = guard.validate([Claim(text="anything")], [])
    assert r2.is_valid is False
    assert len(r2.unvalidated_claims) == 1
    # Evidence but no claims
    r3 = guard.validate([], [Evidence(text="something")])
    assert r3.is_valid is True


def test_batch_of_one() -> None:
    guard = HallucinationGuard()
    claims = [Claim(text="Python is a programming language")]
    evidence = [Evidence(text="Python is a popular programming language used widely")]
    r = guard.validate(claims, evidence)
    assert r.is_valid
    assert len(r.validated_claims) == 1


def test_numeric_match() -> None:
    guard = HallucinationGuard()
    claims = [Claim(text="The build took 42 seconds")]
    evidence = [Evidence(text="Build completed in 42 seconds total")]
    r = guard.validate(claims, evidence)
    assert r.is_valid
    assert r.validated_claims


def test_numeric_mismatch_is_contradiction() -> None:
    guard = HallucinationGuard()
    claims = [Claim(text="The build took 42 seconds")]
    evidence = [Evidence(text="Build completed in 99 seconds total")]
    r = guard.validate(claims, evidence)
    assert not r.validated_claims
    assert len(r.contradictions) == 1


def test_case_insensitivity() -> None:
    guard = HallucinationGuard()
    claims = [Claim(text="TENSORFLOW supports GPU")]
    evidence = [Evidence(text="tensorflow supports gpu acceleration")]
    r = guard.validate(claims, evidence)
    assert r.is_valid


def test_unicode_and_accents() -> None:
    guard = HallucinationGuard()
    claims = [Claim(text="Montréal is in Québec")]
    evidence = [Evidence(text="Montreal sits in the province of Quebec")]
    r = guard.validate(claims, evidence)
    assert r.is_valid, f"expected validation, got {r}"


def test_adversarial_lexical_match_numeric_contradiction() -> None:
    """Claim matches lexically but numbers conflict -> must be contradicted."""
    guard = HallucinationGuard()
    claims = [Claim(text="The model has 1395000000 parameters")]
    evidence = [Evidence(text="The model has 7000000000 parameters")]
    r = guard.validate(claims, evidence)
    assert not r.validated_claims
    assert len(r.contradictions) == 1


def test_determinism_under_seed() -> None:
    random.seed(42)
    guard = HallucinationGuard()
    claims = [Claim(text=f"Fact number {i} about Rome") for i in range(10)]
    evidence = [Evidence(text=f"Rome fact number {i} verified") for i in range(10)]
    r1 = guard.validate(claims, evidence)
    # Shuffle evidence order; result (classification sets) should match.
    random.seed(42)
    shuffled = list(evidence)
    random.shuffle(shuffled)
    r2 = guard.validate(claims, shuffled)
    assert {c.text for c in r1.validated_claims} == {c.text for c in r2.validated_claims}
    assert r1.is_valid == r2.is_valid


def test_confidence_decay_behavior() -> None:
    high = HallucinationGuard(confidence_decay=0.9)
    low = HallucinationGuard(confidence_decay=0.1)
    claims = [Claim(text="Unrelated claim xyz", confidence=1.0)]
    evidence = [Evidence(text="Completely different topic abc")]
    r_high = high.validate(claims, evidence)
    r_low = low.validate(claims, evidence)
    assert r_high.confidence > r_low.confidence
    assert r_high.unvalidated_claims and r_low.unvalidated_claims


def test_mixed_validated_and_unvalidated() -> None:
    guard = HallucinationGuard()
    claims = [
        Claim(text="Linux is an operating system"),
        Claim(text="Atlantis was an advanced civilization"),
    ]
    evidence = [Evidence(text="Linux is a popular operating system used on servers")]
    r = guard.validate(claims, evidence)
    assert len(r.validated_claims) == 1
    assert len(r.unvalidated_claims) == 1
    assert r.is_valid is False


def test_extract_key_terms_removes_stopwords_and_numbers() -> None:
    terms = extract_key_terms("The quick brown fox jumped 42 times over a lazy dog")
    assert "quick" in terms and "fox" in terms and "lazy" in terms
    assert "the" not in terms and "a" not in terms and "over" not in terms
    assert "42" not in terms


def test_extract_numbers_parses_floats_and_ints() -> None:
    nums = extract_numbers("pi is about 3.14 and e is 2.718 with -1 negative and 1e3 big")
    assert 3.14 in nums
    assert 2.718 in nums
    assert -1.0 in nums
    assert 1000.0 in nums


def test_similarity_threshold_knob() -> None:
    strict = HallucinationGuard(similarity_threshold=1.0)
    loose = HallucinationGuard(similarity_threshold=0.1)
    claims = [Claim(text="Aurelius is an agentic coding model built in 2026")]
    evidence = [Evidence(text="Aurelius model")]
    r_strict = strict.validate(claims, evidence)
    r_loose = loose.validate(claims, evidence)
    # Loose should find match; strict should not (evidence lacks most terms).
    assert not r_strict.validated_claims
    # Loose: still needs numeric match; claim has "2026" but evidence doesn't.
    # So this should fall into the "skip" bucket -> unvalidated, not validated.
    assert not r_loose.validated_claims


def test_malformed_inputs_do_not_crash() -> None:
    guard = HallucinationGuard()
    # Non-Claim / non-Evidence items are tolerated with warnings.
    r = guard.validate(
        [Claim(text=""), "not a claim", Claim(text="real fact about dogs")],  # type: ignore[list-item]
        [Evidence(text="dogs are mammals"), 12345],  # type: ignore[list-item]
    )
    assert isinstance(r, ValidationResult)
    assert r.warnings  # accumulated skip warnings
    # The one real claim should still be processed.
    assert len(r.validated_claims) + len(r.unvalidated_claims) + len(r.contradictions) == 1


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError):
        HallucinationGuard(similarity_threshold=0.0)
    with pytest.raises(ValueError):
        HallucinationGuard(similarity_threshold=1.5)
    with pytest.raises(ValueError):
        HallucinationGuard(confidence_decay=-0.1)
    with pytest.raises(ValueError):
        HallucinationGuard(numeric_tolerance=-1.0)
