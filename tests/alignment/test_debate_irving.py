"""Tests for debate-based alignment (Irving et al., 2018)."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.alignment.debate_irving import (
    DebateArgument,
    DebateDataCollector,
    DebateSession,
)


# ---------------------------------------------------------------------------
# Tiny mock model
# ---------------------------------------------------------------------------

class _MockModel(nn.Module):
    """Deterministic tiny mock model for testing.

    generate() always returns 'WINNER: for it is good' as token IDs
    so that the judge deterministically picks 'for'.
    """

    def __init__(self) -> None:
        super().__init__()
        # Single param so next(model.parameters()).device works
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        # Return fixed token IDs that decode to something containing 'winner: for'
        # ASCII codes for "WINNER: for" + padding
        fixed = "WINNER: for this argument is clearly superior."
        ids = [ord(c) % 256 for c in fixed[:max_new_tokens]]
        # Pad to max_new_tokens
        while len(ids) < max_new_tokens:
            ids.append(32)
        new_tokens = torch.tensor([ids], dtype=torch.long)
        return torch.cat([input_ids, new_tokens], dim=1)


@pytest.fixture
def mock_model() -> _MockModel:
    torch.manual_seed(42)
    return _MockModel()


@pytest.fixture
def debate_session(mock_model: _MockModel) -> DebateSession:
    return DebateSession(
        model_for=mock_model,
        model_against=mock_model,
        judge_model=mock_model,
        n_rounds=2,
        max_tokens_per_round=16,
    )


# ---------------------------------------------------------------------------
# DebateArgument dataclass
# ---------------------------------------------------------------------------

def test_debate_argument_dataclass() -> None:
    """DebateArgument stores all fields correctly."""
    arg = DebateArgument(
        position="for",
        content="This is a strong argument.",
        round=1,
        model_id="model_a",
    )
    assert arg.position == "for"
    assert arg.content == "This is a strong argument."
    assert arg.round == 1
    assert arg.model_id == "model_a"


# ---------------------------------------------------------------------------
# DebateSession tests
# ---------------------------------------------------------------------------

def test_debate_session_runs(debate_session: DebateSession) -> None:
    """run_debate returns a dict with all required keys."""
    result = debate_session.run_debate("Python is better than JavaScript.")
    required_keys = {"claim", "arguments", "winner", "judge_reasoning"}
    assert required_keys <= result.keys(), (
        f"Missing keys: {required_keys - result.keys()}"
    )
    assert result["claim"] == "Python is better than JavaScript."
    assert isinstance(result["arguments"], list)
    assert isinstance(result["judge_reasoning"], str)


def test_debate_winner_valid(debate_session: DebateSession) -> None:
    """winner is one of 'for', 'against', or 'tie'."""
    result = debate_session.run_debate("Test claim.")
    assert result["winner"] in ("for", "against", "tie"), (
        f"Invalid winner: {result['winner']!r}"
    )


def test_debate_argument_count(mock_model: _MockModel) -> None:
    """Number of arguments == 2 * n_rounds (one for each side per round)."""
    n_rounds = 3
    session = DebateSession(
        model_for=mock_model,
        model_against=mock_model,
        judge_model=mock_model,
        n_rounds=n_rounds,
        max_tokens_per_round=8,
    )
    result = session.run_debate("AI safety is important.")
    assert len(result["arguments"]) == 2 * n_rounds, (
        f"Expected {2 * n_rounds} arguments, got {len(result['arguments'])}"
    )


# ---------------------------------------------------------------------------
# DebateDataCollector tests
# ---------------------------------------------------------------------------

def _make_debate_result(winner: str = "for") -> dict:
    """Build a minimal debate result dict."""
    args = [
        DebateArgument(position="for", content="Arg for round 1", round=1, model_id="m1"),
        DebateArgument(position="against", content="Arg against round 1", round=1, model_id="m2"),
    ]
    return {
        "claim": "Test claim.",
        "arguments": args,
        "winner": winner,
        "judge_reasoning": "Because of reason X.",
    }


def test_debate_data_collector_record() -> None:
    """record() increases len(debates) by 1 each call."""
    collector = DebateDataCollector()
    assert len(collector.debates) == 0
    collector.record(_make_debate_result("for"))
    assert len(collector.debates) == 1
    collector.record(_make_debate_result("against"))
    assert len(collector.debates) == 2


def test_to_preference_pairs() -> None:
    """to_preference_pairs returns list of dicts with prompt/chosen/rejected keys."""
    collector = DebateDataCollector()
    collector.record(_make_debate_result("for"))
    collector.record(_make_debate_result("against"))
    collector.record(_make_debate_result("tie"))  # should be skipped

    pairs = collector.to_preference_pairs()
    # 'tie' debates are excluded
    assert len(pairs) == 2, f"Expected 2 pairs (tie excluded), got {len(pairs)}"

    for pair in pairs:
        assert "prompt" in pair, f"Missing 'prompt' key in pair: {pair}"
        assert "chosen" in pair, f"Missing 'chosen' key in pair: {pair}"
        assert "rejected" in pair, f"Missing 'rejected' key in pair: {pair}"
        assert isinstance(pair["prompt"], str)
        assert isinstance(pair["chosen"], str)
        assert isinstance(pair["rejected"], str)


def test_collector_stats_keys() -> None:
    """stats() contains all required keys with correct counts."""
    collector = DebateDataCollector()
    collector.record(_make_debate_result("for"))
    collector.record(_make_debate_result("for"))
    collector.record(_make_debate_result("against"))
    collector.record(_make_debate_result("tie"))

    stats = collector.stats()
    assert "n_debates" in stats
    assert "for_wins" in stats
    assert "against_wins" in stats
    assert "ties" in stats

    assert stats["n_debates"] == 4
    assert stats["for_wins"] == 2
    assert stats["against_wins"] == 1
    assert stats["ties"] == 1
