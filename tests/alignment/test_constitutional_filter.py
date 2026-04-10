"""Tests for constitutional_filter.py - automated dataset filtering pipeline."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.alignment.constitutional_filter import (
    SAFETY_PRINCIPLES,
    ConstitutionalFilter,
    ConstitutionalPrinciple,
    FilterScore,
    ModelBasedScorer,
    score_text_heuristic,
)
from src.model.config import AureliusConfig


# ---------------------------------------------------------------------------
# Tiny model fixture for model-based tests
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal model matching the Aurelius API: loss, logits, pkv = model(input_ids)."""

    def __init__(self, vocab_size: int = 256) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, 64)
        self.proj = nn.Linear(64, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)          # (B, T, 64)
        logits = self.proj(x)              # (B, T, vocab_size)
        loss = logits.mean()               # dummy scalar
        return loss, logits, None          # (loss, logits, past_key_values)


@pytest.fixture(scope="module")
def tiny_model() -> _TinyModel:
    model = _TinyModel(vocab_size=256)
    model.eval()
    return model


def _encode(text: str) -> list[int]:
    """Encode text as UTF-8 bytes, clamped to vocab 256."""
    return [b % 256 for b in text.encode("utf-8")][:128]


def _decode(ids: list[int]) -> str:
    """Decode a list of byte-ints back to a string (best-effort)."""
    return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def model_scorer(tiny_model: _TinyModel) -> ModelBasedScorer:
    return ModelBasedScorer(
        model=tiny_model,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        principles=SAFETY_PRINCIPLES,
    )


# ---------------------------------------------------------------------------
# 1. ConstitutionalPrinciple fields
# ---------------------------------------------------------------------------

def test_constitutional_principle_fields() -> None:
    """ConstitutionalPrinciple has name, description, critique_prompt, revision_prompt, weight."""
    p = ConstitutionalPrinciple(
        name="test",
        description="A test principle.",
        critique_prompt="Is this good?",
        revision_prompt="Make this better:",
        weight=2.0,
    )
    assert p.name == "test"
    assert p.description == "A test principle."
    assert p.critique_prompt == "Is this good?"
    assert p.revision_prompt == "Make this better:"
    assert p.weight == 2.0


def test_constitutional_principle_default_weight() -> None:
    """ConstitutionalPrinciple weight defaults to 1.0."""
    p = ConstitutionalPrinciple(
        name="x",
        description="d",
        critique_prompt="c",
        revision_prompt="r",
    )
    assert p.weight == 1.0


# ---------------------------------------------------------------------------
# 2. SAFETY_PRINCIPLES has 4 principles
# ---------------------------------------------------------------------------

def test_safety_principles_count() -> None:
    """SAFETY_PRINCIPLES must contain exactly 4 built-in principles."""
    assert len(SAFETY_PRINCIPLES) == 4


def test_safety_principles_names() -> None:
    """SAFETY_PRINCIPLES includes harmlessness, honesty, helpfulness, coherence."""
    names = {p.name for p in SAFETY_PRINCIPLES}
    assert names == {"harmlessness", "honesty", "helpfulness", "coherence"}


# ---------------------------------------------------------------------------
# 3. FilterScore fields
# ---------------------------------------------------------------------------

def test_filter_score_fields() -> None:
    """FilterScore has principle_name, score, passed, critique."""
    fs = FilterScore(
        principle_name="harmlessness",
        score=0.8,
        passed=True,
        critique="All good.",
    )
    assert fs.principle_name == "harmlessness"
    assert fs.score == 0.8
    assert fs.passed is True
    assert fs.critique == "All good."


def test_filter_score_default_critique() -> None:
    """FilterScore critique defaults to empty string."""
    fs = FilterScore(principle_name="honesty", score=0.6, passed=True)
    assert fs.critique == ""


# ---------------------------------------------------------------------------
# 4. score_text_heuristic harmlessness: harmful text -> score < 0.5
# ---------------------------------------------------------------------------

def test_heuristic_harmlessness_harmful_text() -> None:
    """Harmful text should receive a harmlessness score < 0.5."""
    harmful_text = "I will kill you with a weapon and do something illegal and dangerous."
    principle = next(p for p in SAFETY_PRINCIPLES if p.name == "harmlessness")
    result = score_text_heuristic(harmful_text, principle)
    assert isinstance(result, FilterScore)
    assert result.score < 0.5, f"Expected score < 0.5, got {result.score}"
    assert result.passed is False


# ---------------------------------------------------------------------------
# 5. score_text_heuristic harmlessness: safe text -> score >= 0.5
# ---------------------------------------------------------------------------

def test_heuristic_harmlessness_safe_text() -> None:
    """Safe, benign text should receive a harmlessness score >= 0.5."""
    safe_text = "The sky is blue and the weather is pleasant today."
    principle = next(p for p in SAFETY_PRINCIPLES if p.name == "harmlessness")
    result = score_text_heuristic(safe_text, principle)
    assert result.score >= 0.5, f"Expected score >= 0.5, got {result.score}"
    assert result.passed is True


# ---------------------------------------------------------------------------
# 6. score_text_heuristic helpfulness: short text -> low score
# ---------------------------------------------------------------------------

def test_heuristic_helpfulness_short_text() -> None:
    """Very short text should score low on helpfulness."""
    short_text = "Yes."
    principle = next(p for p in SAFETY_PRINCIPLES if p.name == "helpfulness")
    result = score_text_heuristic(short_text, principle)
    assert result.score < 0.5, f"Expected score < 0.5, got {result.score}"
    assert result.passed is False


# ---------------------------------------------------------------------------
# 7. score_text_heuristic helpfulness: long helpful text -> higher score
# ---------------------------------------------------------------------------

def test_heuristic_helpfulness_long_text() -> None:
    """Long substantive text should score higher on helpfulness."""
    helpful_text = (
        "To answer your question thoroughly, you should consider several factors. "
        "First, understand the context and requirements. "
        "Second, evaluate the available options carefully. "
        "Third, implement the chosen solution systematically."
    )
    principle = next(p for p in SAFETY_PRINCIPLES if p.name == "helpfulness")
    result = score_text_heuristic(helpful_text, principle)
    assert result.score >= 0.5, f"Expected score >= 0.5, got {result.score}"


# ---------------------------------------------------------------------------
# 8. ConstitutionalFilter.filter_sample returns (bool, list)
# ---------------------------------------------------------------------------

def test_filter_sample_return_types() -> None:
    """filter_sample must return a (bool, list) tuple."""
    cf = ConstitutionalFilter()
    result = cf.filter_sample("What is 2+2?", "The answer is 4.")
    assert isinstance(result, tuple)
    assert len(result) == 2
    passed, scores = result
    assert isinstance(passed, bool)
    assert isinstance(scores, list)


# ---------------------------------------------------------------------------
# 9. ConstitutionalFilter.filter_sample scores list length = n_principles
# ---------------------------------------------------------------------------

def test_filter_sample_scores_length() -> None:
    """scores list length should equal number of principles."""
    principles = SAFETY_PRINCIPLES[:3]
    cf = ConstitutionalFilter(principles=principles)
    _, scores = cf.filter_sample("Hello?", "Hello there, how can I help?")
    assert len(scores) == 3


def test_filter_sample_scores_are_filter_score_instances() -> None:
    """Each element of scores should be a FilterScore."""
    cf = ConstitutionalFilter()
    _, scores = cf.filter_sample("test prompt", "test response with enough content here.")
    for s in scores:
        assert isinstance(s, FilterScore)


# ---------------------------------------------------------------------------
# 10. ConstitutionalFilter.filter_dataset returns (list, dict)
# ---------------------------------------------------------------------------

def test_filter_dataset_return_types() -> None:
    """filter_dataset must return (list, dict)."""
    cf = ConstitutionalFilter()
    samples = [
        ("What is AI?", "Artificial intelligence is the simulation of human intelligence."),
        ("How are you?", "I am doing well, thank you for asking!"),
    ]
    result = cf.filter_dataset(samples)
    assert isinstance(result, tuple)
    assert len(result) == 2
    accepted, stats = result
    assert isinstance(accepted, list)
    assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# 11. ConstitutionalFilter.filter_dataset stats has required keys
# ---------------------------------------------------------------------------

def test_filter_dataset_stats_keys() -> None:
    """stats dict must have total, accepted, rejection_rate, per_principle_pass_rate."""
    cf = ConstitutionalFilter()
    samples = [("prompt", "A substantive response with helpful content about the topic.")]
    _, stats = cf.filter_dataset(samples)
    assert "total" in stats
    assert "accepted" in stats
    assert "rejection_rate" in stats
    assert "per_principle_pass_rate" in stats
    assert isinstance(stats["per_principle_pass_rate"], dict)


# ---------------------------------------------------------------------------
# 12. ConstitutionalFilter.filter_dataset acceptance_rate in [0, 1]
# ---------------------------------------------------------------------------

def test_filter_dataset_rejection_rate_range() -> None:
    """rejection_rate must be in [0, 1]."""
    cf = ConstitutionalFilter()
    samples = [
        ("Q", "This is a helpful, safe, and coherent response to your question."),
        ("Q2", "kill weapon illegal harm danger dangerous"),
    ]
    _, stats = cf.filter_dataset(samples)
    assert 0.0 <= stats["rejection_rate"] <= 1.0


def test_filter_dataset_counts_consistent() -> None:
    """accepted + rejected should equal total."""
    cf = ConstitutionalFilter()
    samples = [
        ("q1", "Paris is the capital of France, a beautiful European city."),
        ("q2", "ok"),
        ("q3", "The sky is blue and the weather is lovely today, is it not?"),
    ]
    accepted, stats = cf.filter_dataset(samples)
    assert stats["total"] == 3
    assert stats["accepted"] == len(accepted)


# ---------------------------------------------------------------------------
# 13. ConstitutionalFilter.get_rejection_reasons returns list of strings
# ---------------------------------------------------------------------------

def test_get_rejection_reasons_returns_list_of_strings() -> None:
    """get_rejection_reasons must return a list of strings."""
    cf = ConstitutionalFilter()
    scores = [
        FilterScore("harmlessness", 0.3, False, "harmful content"),
        FilterScore("honesty", 0.8, True, ""),
        FilterScore("helpfulness", 0.2, False, "too short"),
        FilterScore("coherence", 0.9, True, ""),
    ]
    reasons = cf.get_rejection_reasons(scores)
    assert isinstance(reasons, list)
    assert all(isinstance(r, str) for r in reasons)
    assert "harmlessness" in reasons
    assert "helpfulness" in reasons
    assert "honesty" not in reasons
    assert "coherence" not in reasons


def test_get_rejection_reasons_empty_when_all_pass() -> None:
    """get_rejection_reasons returns empty list when all scores pass."""
    cf = ConstitutionalFilter()
    scores = [
        FilterScore("harmlessness", 0.9, True),
        FilterScore("honesty", 0.8, True),
    ]
    assert cf.get_rejection_reasons(scores) == []


# ---------------------------------------------------------------------------
# 14. ModelBasedScorer.score_response returns FilterScore with score in [0,1]
# ---------------------------------------------------------------------------

def test_model_scorer_score_response_returns_filter_score(
    model_scorer: ModelBasedScorer,
) -> None:
    """score_response should return a FilterScore with score in [0, 1]."""
    principle = next(p for p in SAFETY_PRINCIPLES if p.name == "harmlessness")
    result = model_scorer.score_response(
        prompt="What should I do today?",
        response="You could go for a walk or read a book.",
        principle=principle,
    )
    assert isinstance(result, FilterScore)
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.passed, bool)
    assert result.principle_name == "harmlessness"


# ---------------------------------------------------------------------------
# 15. ModelBasedScorer.revise_response returns non-empty string
# ---------------------------------------------------------------------------

def test_model_scorer_revise_response_returns_string(
    model_scorer: ModelBasedScorer,
) -> None:
    """revise_response should return a non-empty string."""
    principle = next(p for p in SAFETY_PRINCIPLES if p.name == "helpfulness")
    revised = model_scorer.revise_response(
        prompt="How can I improve?",
        response="Idk.",
        principle=principle,
        max_new_tokens=16,
    )
    assert isinstance(revised, str)
    assert len(revised) > 0
