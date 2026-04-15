"""Tests for G-Eval NLG evaluation framework."""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
from typing import List

from src.eval.g_eval import (
    GEvalCriteria,
    GEvalJudge,
    GEvalResult,
    make_default_criteria,
    summarization_eval,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
D_MODEL = 16


class MockModel(nn.Module):
    """Simple nn.Embedding + nn.Linear model for testing."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        # input_ids: (B, T)
        x = self.embed(input_ids)   # (B, T, d_model)
        logits = self.proj(x)       # (B, T, vocab_size)
        return (None, logits, None)


def _encode(text: str) -> List[int]:
    """Map each char to ord(c) % VOCAB_SIZE."""
    return [ord(c) % VOCAB_SIZE for c in text]


def _decode(token_id: int) -> str:
    """Map token id back to chr(token_id % 128)."""
    return chr(token_id % 128)


def _make_judge(criteria=None) -> GEvalJudge:
    if criteria is None:
        criteria = make_default_criteria()
    return GEvalJudge(
        model=MockModel(),
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        criteria=criteria,
        device="cpu",
    )


# Short strings for fast inference
DOC = "AI helps."
HYP = "AI is useful."

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


# Test 1
def test_g_eval_criteria_defaults():
    """GEvalCriteria dataclass creates with defaults correctly."""
    c = GEvalCriteria(name="Fluency", description="Grammar quality.")
    assert c.name == "Fluency"
    assert c.description == "Grammar quality."
    assert c.scale == 5
    assert c.weight == 1.0


# Test 2
def test_g_eval_result_fields():
    """GEvalResult has correct fields."""
    result = GEvalResult(
        criteria_scores={"Coherence": 3.5},
        composite_score=3.5,
        raw_logprobs={"Coherence": [-0.5, -1.0, -0.8, -1.2, -0.9]},
    )
    assert hasattr(result, "criteria_scores")
    assert hasattr(result, "composite_score")
    assert hasattr(result, "raw_logprobs")
    assert result.composite_score == pytest.approx(3.5)


# Test 3
def test_make_default_criteria_returns_four():
    """make_default_criteria() returns 4 criteria."""
    criteria = make_default_criteria()
    assert len(criteria) == 4
    names = {c.name for c in criteria}
    assert names == {"Coherence", "Fluency", "Consistency", "Relevance"}


# Test 4
def test_g_eval_judge_instantiates():
    """GEvalJudge instantiates without error."""
    judge = _make_judge()
    assert judge is not None
    assert judge.criteria is not None
    assert len(judge.criteria) == 4


# Test 5
def test_build_prompt_contains_criterion_name():
    """_build_prompt() returns non-empty string containing criterion name."""
    judge = _make_judge()
    criterion = GEvalCriteria(name="Coherence", description="Tests coherence.")
    prompt = judge._build_prompt(
        task_description="Evaluate the summary.",
        document=DOC,
        summary=HYP,
        criterion=criterion,
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "Coherence" in prompt


# Test 6
def test_weighted_score_uniform_logprobs_near_middle():
    """_weighted_score() with uniform logprobs returns ~middle of scale (~3 for scale=5)."""
    judge = _make_judge()
    # Uniform log-probs: log(1/5) for each
    uniform_lp = [math.log(1.0 / 5)] * 5
    score = judge._weighted_score(uniform_lp, scale=5)
    # Expected value under uniform: (1+2+3+4+5)/5 = 3.0
    assert score == pytest.approx(3.0, abs=1e-4)


# Test 7
def test_weighted_score_all_weight_on_five():
    """_weighted_score() with all weight on '5' returns ~5.0."""
    judge = _make_judge()
    # Near-zero log-prob for 1..4, near-zero log-prob loss for 5
    # Use very large logit for slot 5 (index 4)
    big = 100.0
    # log-probs that strongly favor index 4 (score=5)
    logprobs = [-big, -big, -big, -big, 0.0]
    score = judge._weighted_score(logprobs, scale=5)
    assert score == pytest.approx(5.0, abs=0.01)


# Test 8
def test_weighted_score_all_weight_on_one():
    """_weighted_score() with all weight on '1' returns ~1.0."""
    judge = _make_judge()
    big = 100.0
    logprobs = [0.0, -big, -big, -big, -big]
    score = judge._weighted_score(logprobs, scale=5)
    assert score == pytest.approx(1.0, abs=0.01)


# Test 9
def test_evaluate_composite_score_in_range():
    """evaluate() returns GEvalResult with composite_score in [1, 5]."""
    judge = _make_judge()
    result = judge.evaluate(document=DOC, hypothesis=HYP)
    assert isinstance(result, GEvalResult)
    assert 1.0 <= result.composite_score <= 5.0


# Test 10
def test_evaluate_returns_per_criterion_scores():
    """evaluate() returns per-criterion scores for all criteria."""
    judge = _make_judge()
    result = judge.evaluate(document=DOC, hypothesis=HYP)
    for criterion in judge.criteria:
        assert criterion.name in result.criteria_scores
        assert 1.0 <= result.criteria_scores[criterion.name] <= 5.0


# Test 11
def test_evaluate_composite_is_weighted_mean():
    """evaluate() composite_score is weighted mean of criteria_scores."""
    criteria = make_default_criteria()
    judge = _make_judge(criteria=criteria)
    result = judge.evaluate(document=DOC, hypothesis=HYP)

    total_weight = sum(c.weight for c in criteria)
    expected_composite = sum(
        result.criteria_scores[c.name] * c.weight for c in criteria
    ) / total_weight

    assert result.composite_score == pytest.approx(expected_composite, abs=1e-5)


# Test 12
def test_batch_evaluate_returns_two_results():
    """batch_evaluate() with 2 pairs returns list of 2 GEvalResults."""
    judge = _make_judge()
    docs = [DOC, "Science advances."]
    hyps = [HYP, "Science grows."]
    results = judge.batch_evaluate(documents=docs, hypotheses=hyps)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, GEvalResult)


# Test 13
def test_summarization_eval_completes():
    """summarization_eval() completes without error."""
    model = MockModel()
    result = summarization_eval(
        model=model,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        document=DOC,
        summary=HYP,
    )
    assert isinstance(result, GEvalResult)
    assert 1.0 <= result.composite_score <= 5.0


# Test 14
def test_custom_weights_affect_composite_score():
    """Criteria with custom weights affects composite_score calculation."""
    # Two criteria: one with weight=1, one with weight=9
    criteria_equal = [
        GEvalCriteria(name="A", description="Criterion A", weight=1.0),
        GEvalCriteria(name="B", description="Criterion B", weight=1.0),
    ]
    criteria_heavy_b = [
        GEvalCriteria(name="A", description="Criterion A", weight=1.0),
        GEvalCriteria(name="B", description="Criterion B", weight=9.0),
    ]

    # Use the same model so scores are deterministic
    torch.manual_seed(0)
    model = MockModel()

    judge_equal = GEvalJudge(
        model=model, tokenizer_encode=_encode, tokenizer_decode=_decode,
        criteria=criteria_equal, device="cpu",
    )
    judge_heavy = GEvalJudge(
        model=model, tokenizer_encode=_encode, tokenizer_decode=_decode,
        criteria=criteria_heavy_b, device="cpu",
    )

    result_equal = judge_equal.evaluate(document=DOC, hypothesis=HYP)
    result_heavy = judge_heavy.evaluate(document=DOC, hypothesis=HYP)

    score_a = result_equal.criteria_scores["A"]
    score_b = result_equal.criteria_scores["B"]

    if abs(score_a - score_b) > 0.001:
        # Composite scores should differ when weights are unequal
        assert result_equal.composite_score != pytest.approx(result_heavy.composite_score, abs=1e-4)
    else:
        # If scores are identical, composites will be equal regardless of weights
        assert result_equal.composite_score == pytest.approx(result_heavy.composite_score, abs=1e-4)

    # Verify heavy-B composite matches manual calculation
    total_w = 1.0 + 9.0
    expected_heavy = (
        result_heavy.criteria_scores["A"] * 1.0
        + result_heavy.criteria_scores["B"] * 9.0
    ) / total_w
    assert result_heavy.composite_score == pytest.approx(expected_heavy, abs=1e-5)
