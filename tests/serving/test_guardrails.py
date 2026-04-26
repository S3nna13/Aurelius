"""Tests for src/serving/guardrails.py"""

from src.serving.guardrails import (
    ContentGuardrails,
    GuardrailPolicy,
    GuardrailResult,
)


def test_content_guardrails_instantiates():
    g = ContentGuardrails()
    assert g is not None
    assert isinstance(g.policy, GuardrailPolicy)


def test_clean_input_allowed():
    g = ContentGuardrails()
    result = g.check_input("Hello, how are you?")
    assert result.allowed is True


def test_block_pattern_blocks_input():
    policy = GuardrailPolicy(block_patterns=["badword"])
    g = ContentGuardrails(policy)
    result = g.check_input("This has a badword in it")
    assert result.allowed is False
    assert "badword" in result.reason


def test_over_length_text_blocked():
    policy = GuardrailPolicy(max_length=10)
    g = ContentGuardrails(policy)
    result = g.check_input("A" * 11)
    assert result.allowed is False


def test_truncate_if_needed_shortens():
    policy = GuardrailPolicy(max_length=5)
    g = ContentGuardrails(policy)
    out = g.truncate_if_needed("Hello World")
    assert len(out) == 5
    assert out == "Hello"


def test_harm_score_returns_float_in_range():
    g = ContentGuardrails()
    score = g._harm_score("some benign text")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_harm_score_positive_for_suspicious_text():
    g = ContentGuardrails()
    score = g._harm_score("INJECT this payload")
    assert score > 0.0


def test_check_output_returns_guardrail_result():
    g = ContentGuardrails()
    result = g.check_output("This is a safe output.")
    assert isinstance(result, GuardrailResult)


def test_guardrail_result_has_all_fields():
    result = GuardrailResult(allowed=True, reason="OK", modified_input=None, harm_score=0.5)
    assert hasattr(result, "allowed")
    assert hasattr(result, "reason")
    assert hasattr(result, "modified_input")
    assert hasattr(result, "harm_score")
    assert result.harm_score == 0.5


def test_pattern_check_finds_match():
    policy = GuardrailPolicy(block_patterns=["SECRET"])
    g = ContentGuardrails(policy)
    blocked, matched = g._pattern_check("Please do not share SECRET info")
    assert blocked is True
    assert matched == "SECRET"
