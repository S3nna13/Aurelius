"""Tests for src/reasoning/step_verifier.py — at least 20 tests."""

from __future__ import annotations

import pytest

from src.reasoning.step_verifier import (
    _MAX_STEP_LEN,
    _MAX_STEPS,
    STEP_VERIFIER,
    StepScore,
    StepVerifier,
    VerificationLabel,
)

# ---------------------------------------------------------------------------
# StepScore dataclass validation
# ---------------------------------------------------------------------------


def test_stepscore_confidence_below_zero_raises():
    with pytest.raises(ValueError, match="confidence"):
        StepScore(step_index=0, content="x", label=VerificationLabel.CORRECT, confidence=-0.1)


def test_stepscore_confidence_above_one_raises():
    with pytest.raises(ValueError, match="confidence"):
        StepScore(step_index=0, content="x", label=VerificationLabel.CORRECT, confidence=1.1)


def test_stepscore_negative_step_index_raises():
    with pytest.raises(ValueError, match="step_index"):
        StepScore(step_index=-1, content="x", label=VerificationLabel.CORRECT, confidence=0.5)


def test_stepscore_valid_boundary_confidence_zero():
    s = StepScore(step_index=0, content="x", label=VerificationLabel.NEUTRAL, confidence=0.0)
    assert s.confidence == 0.0


def test_stepscore_valid_boundary_confidence_one():
    s = StepScore(step_index=0, content="x", label=VerificationLabel.CORRECT, confidence=1.0)
    assert s.confidence == 1.0


# ---------------------------------------------------------------------------
# verify_step: basic label outcomes
# ---------------------------------------------------------------------------


def test_verify_step_short_step_returns_neutral():
    v = StepVerifier()
    score = v.verify_step(0, "ab")  # len < 5
    assert score.label == VerificationLabel.NEUTRAL


def test_verify_step_whitespace_only_returns_neutral():
    v = StepVerifier()
    score = v.verify_step(0, "     ")
    assert score.label == VerificationLabel.NEUTRAL


def test_verify_step_contradiction_keyword_returns_incorrect():
    v = StepVerifier()
    score = v.verify_step(0, "This is wrong because the formula is inverted.")
    assert score.label == VerificationLabel.INCORRECT


def test_verify_step_contradiction_confidence_gte_075():
    v = StepVerifier()
    score = v.verify_step(0, "This is wrong because the formula is inverted.")
    assert score.confidence >= 0.75


def test_verify_step_normal_step_returns_correct():
    v = StepVerifier()
    score = v.verify_step(0, "We substitute x=2 into the equation to get 4.")
    assert score.label == VerificationLabel.CORRECT


def test_verify_step_confidence_in_range():
    v = StepVerifier()
    score = v.verify_step(0, "We substitute x=2 into the equation to get 4.")
    assert 0.0 <= score.confidence <= 1.0


def test_verify_step_negative_index_raises():
    v = StepVerifier()
    with pytest.raises(ValueError, match="step_index"):
        v.verify_step(-1, "valid content here")


def test_verify_step_content_too_long_raises():
    v = StepVerifier()
    with pytest.raises(ValueError):
        v.verify_step(0, "x" * (_MAX_STEP_LEN + 1))


def test_verify_step_step_index_preserved():
    v = StepVerifier()
    score = v.verify_step(7, "We compute the derivative of f(x).")
    assert score.step_index == 7


def test_verify_step_confidence_scales_with_length():
    v = StepVerifier()
    short = v.verify_step(0, "We solve it now.")  # shorter normal step
    long_content = (
        "We carefully derive the solution by expanding the polynomial and "
        "collecting terms, then applying the quadratic formula to obtain roots. " * 3
    )
    long = v.verify_step(1, long_content)
    # longer correct step should have higher or equal confidence (capped at 0.85)
    assert long.confidence >= short.confidence or long.confidence == 0.85


def test_verify_step_confidence_capped_at_085():
    v = StepVerifier()
    # A very long step should be capped at 0.85
    long_content = "This is a valid reasoning step that explains everything. " * 100
    score = v.verify_step(0, long_content)
    assert score.confidence <= 0.85


# ---------------------------------------------------------------------------
# verify_step: adversarial inputs
# ---------------------------------------------------------------------------


def test_verify_step_null_bytes_do_not_crash():
    v = StepVerifier()
    content = "Valid step\x00with null bytes inside."
    score = v.verify_step(0, content)
    assert score.label in (
        VerificationLabel.CORRECT,
        VerificationLabel.NEUTRAL,
        VerificationLabel.INCORRECT,
    )


# ---------------------------------------------------------------------------
# verify_chain
# ---------------------------------------------------------------------------


def test_verify_chain_correct_sequential_indices():
    v = StepVerifier()
    steps = [
        "First we note that x equals 2.",
        "Then we substitute to get 4.",
        "Therefore the answer is 4.",
    ]
    scores = v.verify_chain(steps)
    for i, s in enumerate(scores):
        assert s.step_index == i


def test_verify_chain_too_many_steps_raises():
    v = StepVerifier()
    with pytest.raises(ValueError, match="too many steps"):
        v.verify_chain(["step"] * (_MAX_STEPS + 1))


def test_verify_chain_empty_list_returns_empty():
    v = StepVerifier()
    assert v.verify_chain([]) == []


def test_verify_chain_returns_correct_count():
    v = StepVerifier()
    steps = [
        "Step one is to identify the variable.",
        "Step two is to isolate x.",
        "Step three gives us the final answer.",
    ]
    scores = v.verify_chain(steps)
    assert len(scores) == 3


# ---------------------------------------------------------------------------
# aggregate_score
# ---------------------------------------------------------------------------


def test_aggregate_score_all_correct_above_half():
    v = StepVerifier()
    scores = [
        StepScore(
            step_index=i, content="good step", label=VerificationLabel.CORRECT, confidence=0.8
        )
        for i in range(5)
    ]
    result = v.aggregate_score(scores)
    assert result > 0.5


def test_aggregate_score_incorrect_with_high_confidence_drags_down():
    v = StepVerifier()
    scores = [
        StepScore(step_index=0, content="good", label=VerificationLabel.CORRECT, confidence=0.9),
        StepScore(step_index=1, content="bad", label=VerificationLabel.INCORRECT, confidence=0.95),
        StepScore(step_index=2, content="good", label=VerificationLabel.CORRECT, confidence=0.9),
    ]
    result = v.aggregate_score(scores)
    # An INCORRECT step with label_weight=0.0 → adjusted = 0 * conf + (1-conf) * 0.1
    # This drags the geometric mean toward 0
    assert result < 0.3


def test_aggregate_score_all_neutral_mid_range():
    v = StepVerifier()
    scores = [
        StepScore(step_index=i, content="ok", label=VerificationLabel.NEUTRAL, confidence=0.5)
        for i in range(4)
    ]
    result = v.aggregate_score(scores)
    # label_weight=0.5, confidence=0.5 → adjusted = 0.5*0.5 + 0.5*0.1 = 0.3
    assert 0.0 < result < 0.8


def test_aggregate_score_empty_returns_zero():
    v = StepVerifier()
    assert v.aggregate_score([]) == 0.0


# ---------------------------------------------------------------------------
# Custom contradiction keywords
# ---------------------------------------------------------------------------


def test_custom_contradiction_keyword_triggers_incorrect():
    v = StepVerifier(contradiction_keywords=["banana contradiction"])
    score = v.verify_step(0, "This step has a banana contradiction in its logic.")
    assert score.label == VerificationLabel.INCORRECT


def test_custom_keywords_replace_defaults():
    v = StepVerifier(contradiction_keywords=["banana contradiction"])
    # Default keyword "this is wrong" should NOT trigger anymore
    score = v.verify_step(0, "This is wrong according to my custom verifier check.")
    # Without the default keyword list, this should be CORRECT (no custom keyword present)
    assert score.label == VerificationLabel.CORRECT


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


def test_step_verifier_singleton_is_instance():
    assert isinstance(STEP_VERIFIER, StepVerifier)
