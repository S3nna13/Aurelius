"""Tests for LLM-as-judge evaluation framework."""

from __future__ import annotations

import pytest
import torch

from src.eval.llm_judge import (
    STANDARD_CRITERIA,
    JudgingCriteria,
    LLMJudge,
    PairwiseJudge,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_model(fixed_output: str = "7"):
    """Return a model whose generate() always returns a fixed decode string."""

    class MockModel(torch.nn.Module):
        def __init__(self, output_text: str):
            super().__init__()
            self._output_text = output_text

        def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32):
            # Return a tensor that includes input_ids plus one extra token
            # The decode function is mocked, so we just append a dummy token.
            dummy = torch.zeros(input_ids.shape[0], 1, dtype=torch.long)
            return torch.cat([input_ids, dummy], dim=-1)

    return MockModel(fixed_output)


def _ENCODE(s):
    return [ord(c) % 100 for c in s[:10]]


def _DECODE(ids):
    return "7"  # always decodes to "7"


def _make_judge(criteria=None) -> LLMJudge:
    model = _make_mock_model("7")
    return LLMJudge(model, _ENCODE, _DECODE, criteria=criteria)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_prompt_contains_question():
    judge = _make_judge()
    criterion = STANDARD_CRITERIA[0]
    prompt = judge.build_prompt("What is AI?", "AI is cool.", criterion)
    assert "What is AI?" in prompt


def test_build_prompt_contains_scale():
    judge = _make_judge()
    criterion = JudgingCriteria("quality", "Overall quality", scale=(1, 10))
    prompt = judge.build_prompt("Q?", "A.", criterion)
    assert "1" in prompt
    assert "10" in prompt


def test_parse_score_valid():
    judge = _make_judge()
    criterion = STANDARD_CRITERIA[0]
    result = judge.parse_score("I give it a 7", criterion)
    assert result == pytest.approx(7.0)


def test_parse_score_clamped():
    judge = _make_judge()
    criterion = JudgingCriteria("test", "test", scale=(1, 10))
    result = judge.parse_score("15", criterion)
    assert result == pytest.approx(10.0)


def test_parse_score_invalid():
    judge = _make_judge()
    criterion = STANDARD_CRITERIA[0]
    result = judge.parse_score("no numbers here", criterion)
    assert result is None


def test_judge_single_keys():
    criteria = [
        JudgingCriteria("helpfulness", "How helpful?"),
        JudgingCriteria("accuracy", "How accurate?"),
    ]
    judge = _make_judge(criteria=criteria)
    result = judge.judge_single("What is 2+2?", "It is 4.")
    assert "helpfulness" in result
    assert "accuracy" in result
    assert "mean_score" in result


def test_compare_responses_winner():
    judge = _make_judge()
    result = judge.compare_responses("Q?", "Response A", "Response B")
    assert "winner" in result
    assert result["winner"] in ("A", "B", "tie")


def test_tournament_sorted():
    judge = PairwiseJudge(_make_mock_model(), _ENCODE, _DECODE)
    responses = ["Response 1", "Response 2", "Response 3"]
    ranking = judge.tournament("What is the best?", responses)
    assert len(ranking) == len(responses)
    win_rates = [wr for _, wr in ranking]
    assert win_rates == sorted(win_rates, reverse=True)
