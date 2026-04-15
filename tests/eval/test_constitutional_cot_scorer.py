"""Tests for src/eval/constitutional_cot_scorer.py"""
import pytest
import torch
import torch.nn as nn

from src.eval.constitutional_cot_scorer import (
    ConstitutionalPrinciple,
    PrincipleScore,
    ConstitutionalCotResult,
    HeuristicViolationDetector,
    ConstitutionalCoTScorer,
    make_default_principles,
    aggregate_constitutional_scores,
)

VOCAB_SIZE = 128


class MockModel(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        logits = self.proj(x)
        return (None, logits, None)


def encode_fn(text: str):
    return [ord(c) % VOCAB_SIZE for c in text] or [0]


def decode_fn(token_id: int) -> str:
    return chr(token_id % 128)


@pytest.fixture
def principles():
    return make_default_principles()


@pytest.fixture
def heuristic(principles):
    return HeuristicViolationDetector(principles)


@pytest.fixture
def model():
    torch.manual_seed(42)
    return MockModel()


@pytest.fixture
def scorer(model, principles):
    return ConstitutionalCoTScorer(
        model=model,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        principles=principles,
        vocab_size=VOCAB_SIZE,
    )


# ---------------------------------------------------------------------------
# ConstitutionalPrinciple
# ---------------------------------------------------------------------------

def test_principle_creates_correctly():
    p = ConstitutionalPrinciple(
        name="test",
        description="A test principle",
        critique_prompt="Does this violate test?",
        violation_keywords=["bad", "wrong"],
        weight=1.5,
    )
    assert p.name == "test"
    assert p.weight == 1.5
    assert "bad" in p.violation_keywords


# ---------------------------------------------------------------------------
# make_default_principles
# ---------------------------------------------------------------------------

def test_make_default_principles_returns_five():
    ps = make_default_principles()
    assert len(ps) == 5


def test_default_principles_have_names():
    ps = make_default_principles()
    names = {p.name for p in ps}
    assert "harmlessness" in names
    assert "honesty" in names


# ---------------------------------------------------------------------------
# HeuristicViolationDetector
# ---------------------------------------------------------------------------

def test_detect_returns_dict_with_all_principle_names(heuristic, principles):
    result = heuristic.detect("A clean, safe response.")
    assert set(result.keys()) == {p.name for p in principles}


def test_detect_flags_keyword_violation(heuristic):
    result = heuristic.detect("I will kill all pests.")
    assert result["harmlessness"] is True


def test_detect_clean_response_no_violation(heuristic):
    result = heuristic.detect("The weather is sunny today.")
    # None of the default keywords should match
    assert not any(result.values())


def test_any_violation_clean(heuristic):
    assert heuristic.any_violation("The sky is blue.") is False


def test_any_violation_flagged(heuristic):
    assert heuristic.any_violation("How to hack into systems.") is True


# ---------------------------------------------------------------------------
# ConstitutionalCoTScorer
# ---------------------------------------------------------------------------

def test_scorer_instantiates(scorer):
    assert scorer is not None
    assert len(scorer.principles) == 5


def test_build_critique_prompt_contains_principle_name(scorer):
    p = scorer.principles[0]
    prompt = scorer._build_critique_prompt(p, "test response")
    assert p.name in prompt
    assert "test response" in prompt


def test_score_principle_confidence_in_range(scorer):
    principle = scorer.principles[0]
    ps = scorer.score_principle("The sky is blue.", principle)
    assert isinstance(ps, PrincipleScore)
    assert 0.0 <= ps.confidence <= 1.0


def test_logprob_yes_no_are_nonpositive(scorer):
    principle = scorer.principles[0]
    ps = scorer.score_principle("The sky is blue.", principle)
    assert ps.logprob_yes <= 0.0
    assert ps.logprob_no <= 0.0


def test_score_returns_result(scorer):
    result = scorer.score("A helpful, safe response.")
    assert isinstance(result, ConstitutionalCotResult)


def test_safety_score_in_range(scorer):
    result = scorer.score("The weather is nice.")
    assert 0.0 <= result.safety_score <= 1.0


def test_principle_scores_length(scorer, principles):
    result = scorer.score("Hello world.")
    assert len(result.principle_scores) == len(principles)


def test_overall_safe_when_no_violations(scorer):
    result = scorer.score("The sky is blue and the sun is shining.")
    # With heuristic-only checks (skip_heuristic_safe=True), clean response → safe
    assert result.overall_safe is True


def test_batch_score_returns_same_length(scorer):
    responses = ["Hello.", "Goodbye.", "How are you?"]
    results = scorer.batch_score(responses)
    assert len(results) == len(responses)


def test_batch_score_each_is_result(scorer):
    results = scorer.batch_score(["Safe text.", "Also safe."])
    for r in results:
        assert isinstance(r, ConstitutionalCotResult)


# ---------------------------------------------------------------------------
# aggregate_constitutional_scores
# ---------------------------------------------------------------------------

def _make_result(safe: bool, safety_score: float, principles_list):
    ps_list = [
        PrincipleScore(
            principle_name=p.name,
            is_violation=not safe,
            confidence=0.9,
            reasoning="test",
            logprob_yes=-1.0,
            logprob_no=-0.5,
        )
        for p in principles_list
    ]
    return ConstitutionalCotResult(
        response="test",
        principle_scores=ps_list,
        overall_safe=safe,
        safety_score=safety_score,
        weighted_violation_rate=0.0 if safe else 1.0,
    )


def test_aggregate_returns_mean_safety_score(principles):
    r1 = _make_result(True, 1.0, principles)
    r2 = _make_result(False, 0.0, principles)
    agg = aggregate_constitutional_scores([r1, r2])
    assert "mean_safety_score" in agg
    assert abs(agg["mean_safety_score"] - 0.5) < 1e-6


def test_aggregate_violation_rate_in_range(principles):
    results = [_make_result(True, 1.0, principles) for _ in range(5)]
    agg = aggregate_constitutional_scores(results)
    assert 0.0 <= agg["violation_rate"] <= 1.0


def test_aggregate_empty_list():
    agg = aggregate_constitutional_scores([])
    assert "mean_safety_score" in agg
    assert agg["mean_safety_score"] == 1.0
