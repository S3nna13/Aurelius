"""Tests for ConstitutionalAIv2Pipeline."""
from __future__ import annotations

import pytest

from src.alignment.cai_v2_pipeline import (
    CAIv2Result,
    ConstitutionalAIv2Pipeline,
    CritiqueResult,
    Principle,
)


@pytest.fixture()
def pipeline() -> ConstitutionalAIv2Pipeline:
    return ConstitutionalAIv2Pipeline()


@pytest.fixture()
def single_principle() -> Principle:
    return Principle(
        principle_id="test_p",
        text="Be safe and helpful.",
        weight=1.0,
        category="safety",
    )


def test_default_principles_loaded(pipeline: ConstitutionalAIv2Pipeline) -> None:
    assert len(pipeline.principles) == 3


def test_add_principle(pipeline: ConstitutionalAIv2Pipeline, single_principle: Principle) -> None:
    pipeline.add_principle(single_principle)
    ids = [p.principle_id for p in pipeline.principles]
    assert "test_p" in ids


def test_remove_principle(pipeline: ConstitutionalAIv2Pipeline) -> None:
    pipeline.remove_principle("hhh_helpful")
    ids = [p.principle_id for p in pipeline.principles]
    assert "hhh_helpful" not in ids
    assert len(pipeline.principles) == 2


def test_remove_nonexistent_principle_is_noop(pipeline: ConstitutionalAIv2Pipeline) -> None:
    before = len(pipeline.principles)
    pipeline.remove_principle("does_not_exist")
    assert len(pipeline.principles) == before


def test_critique_clean_response_zero_score(
    pipeline: ConstitutionalAIv2Pipeline, single_principle: Principle
) -> None:
    result = pipeline.critique("This is a perfectly fine response.", single_principle)
    assert isinstance(result, CritiqueResult)
    assert result.violation_score == 0.0
    assert result.principle_id == "test_p"


def test_critique_flagged_word_raises_score(
    pipeline: ConstitutionalAIv2Pipeline, single_principle: Principle
) -> None:
    result = pipeline.critique("This contains harm and kill and attack words.", single_principle)
    assert result.violation_score > 0.0


def test_critique_score_capped_at_one(
    pipeline: ConstitutionalAIv2Pipeline, single_principle: Principle
) -> None:
    many_flags = " ".join(["harm kill attack exploit illegal dangerous"] * 5)
    result = pipeline.critique(many_flags, single_principle)
    assert result.violation_score <= 1.0


def test_critique_contains_principle_text(
    pipeline: ConstitutionalAIv2Pipeline, single_principle: Principle
) -> None:
    result = pipeline.critique("hello", single_principle)
    assert single_principle.text in result.critique


def test_revise_clean_response_unchanged(pipeline: ConstitutionalAIv2Pipeline) -> None:
    critiques = [
        CritiqueResult("p1", "ok", 0.1, "no revision needed"),
        CritiqueResult("p2", "ok", 0.2, "no revision needed"),
    ]
    original = "A clean response."
    revised = pipeline.revise(original, critiques)
    assert revised == original


def test_revise_flagged_response_gets_prefix(pipeline: ConstitutionalAIv2Pipeline) -> None:
    critiques = [
        CritiqueResult("p1", "bad", 0.5, "revise this"),
    ]
    revised = pipeline.revise("A bad response.", critiques)
    assert revised.startswith("[Revised per principles]")


def test_evaluate_revised_longer_gives_08(pipeline: ConstitutionalAIv2Pipeline) -> None:
    original = "short"
    revised = "[Revised per principles] short response text"
    score = pipeline.evaluate(original, revised)
    assert score == 0.8


def test_evaluate_unchanged_gives_05(pipeline: ConstitutionalAIv2Pipeline) -> None:
    score = pipeline.evaluate("same", "same")
    assert score == 0.5


def test_run_returns_cai_result(pipeline: ConstitutionalAIv2Pipeline) -> None:
    result = pipeline.run("Hello world.")
    assert isinstance(result, CAIv2Result)
    assert result.original_response == "Hello world."


def test_run_critiques_count_matches_principles(pipeline: ConstitutionalAIv2Pipeline) -> None:
    result = pipeline.run("Some response text.")
    assert len(result.critiques) == len(pipeline.principles)


def test_run_harmlessness_score_in_range(pipeline: ConstitutionalAIv2Pipeline) -> None:
    result = pipeline.run("Neutral response without any bad words.")
    assert 0.0 <= result.overall_harmlessness_score <= 1.0


def test_run_batch_processes_all(pipeline: ConstitutionalAIv2Pipeline) -> None:
    responses = ["Hello.", "World.", "Test response."]
    results = pipeline.run_batch(responses)
    assert len(results) == 3
    for i, result in enumerate(results):
        assert result.original_response == responses[i]


def test_run_batch_empty_input(pipeline: ConstitutionalAIv2Pipeline) -> None:
    assert pipeline.run_batch([]) == []


def test_custom_principles_pipeline() -> None:
    principles = [
        Principle("p1", "Be concise.", 2.0, "helpfulness"),
        Principle("p2", "Be accurate.", 1.0, "ethics"),
    ]
    p = ConstitutionalAIv2Pipeline(principles=principles)
    assert len(p.principles) == 2
    result = p.run("A response.")
    assert isinstance(result, CAIv2Result)


def test_weighted_harmlessness_reflects_weights() -> None:
    principles = [
        Principle("p1", "Safety first.", 3.0, "safety"),
        Principle("p2", "Be helpful.", 1.0, "helpfulness"),
    ]
    p = ConstitutionalAIv2Pipeline(principles=principles)
    result = p.run("completely clean text")
    assert result.overall_harmlessness_score > 0.0
