"""Tests for debate-style alignment module."""

from __future__ import annotations

import pytest

from src.alignment.debate import (
    DebateConfig,
    DebateTurn,
    DebateJudge,
    DebateDebater,
    DebateSession,
    aggregate_debate_results,
    format_debate_for_training,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CONFIG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

TOKENIZER_ENCODE = lambda t: list(t.encode("utf-8")[:32])
TOKENIZER_DECODE = lambda ids: bytes(ids).decode("utf-8", errors="replace")


def make_model():
    return AureliusTransformer(CONFIG)


def make_debate_config(**kwargs):
    return DebateConfig(n_debaters=2, n_rounds=2, max_tokens_per_turn=4, **kwargs)


def make_judge(model=None):
    if model is None:
        model = make_model()
    return DebateJudge(model, TOKENIZER_ENCODE, TOKENIZER_DECODE)


def make_debater(debater_id=0, model=None, config=None):
    if model is None:
        model = make_model()
    if config is None:
        config = make_debate_config()
    return DebateDebater(debater_id, model, TOKENIZER_ENCODE, TOKENIZER_DECODE, config)


def make_session(question="Is Python better than Java?", positions=None):
    if positions is None:
        positions = ["yes", "no"]
    config = make_debate_config()
    model = make_model()
    debaters = [
        make_debater(debater_id=i, model=model, config=config)
        for i in range(config.n_debaters)
    ]
    judge = make_judge(model)
    return DebateSession(question, positions, debaters, judge, config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_debate_config_defaults():
    cfg = DebateConfig()
    assert cfg.n_debaters == 2
    assert cfg.n_rounds == 3
    assert cfg.max_tokens_per_turn == 64
    assert cfg.judge_temperature == 0.0
    assert cfg.debater_temperature == 0.7
    assert cfg.use_simultaneous is True


def test_debate_turn_fields():
    turn = DebateTurn(debater_id=1, round_num=2, argument="test arg", score=0.5)
    assert turn.debater_id == 1
    assert turn.round_num == 2
    assert turn.argument == "test arg"
    assert turn.score == pytest.approx(0.5)


def test_debate_judge_score_is_float():
    judge = make_judge()
    score = judge.score_argument("What is 2+2?", "The answer is 4.")
    assert isinstance(score, float)


def test_debate_judge_select_best_valid_index():
    judge = make_judge()
    arguments = ["The sky is blue.", "Water is wet.", "Fire is hot."]
    idx = judge.select_best("What is a fact?", arguments)
    assert 0 <= idx < len(arguments)


def test_debate_debater_generates_string():
    debater = make_debater(debater_id=0)
    result = debater.generate_argument("Is coffee good?", [], "yes")
    assert isinstance(result, str)


def test_debate_session_run_round_count():
    """Each round should produce exactly n_debaters turns."""
    session = make_session()
    turns = session.run_round(round_num=1)
    assert len(turns) == session.config.n_debaters


def test_debate_session_run_returns_winner():
    """Winner returned by run() must be one of the positions."""
    session = make_session(positions=["yes", "no"])
    winner, _ = session.run()
    assert winner in ["yes", "no"]


def test_debate_session_history_length():
    """After run(), history should have n_rounds * n_debaters turns."""
    session = make_session()
    _, all_turns = session.run()
    expected = session.config.n_rounds * session.config.n_debaters
    assert len(all_turns) == expected


def test_debate_transcript_is_string():
    session = make_session()
    session.run()
    transcript = session.transcript()
    assert isinstance(transcript, str)
    assert len(transcript) > 0


def test_aggregate_results_keys():
    """aggregate_debate_results must return dict with required keys."""
    turns = [DebateTurn(debater_id=0, round_num=1, argument="a", score=0.5)]
    sessions = [("yes", turns), ("no", turns), ("yes", turns)]
    result = aggregate_debate_results(sessions)
    assert "win_rates" in result
    assert "mean_rounds" in result
    assert "total_sessions" in result


def test_aggregate_win_rates_sum():
    """Win rates across all positions should sum to ~1.0."""
    turns = [DebateTurn(debater_id=0, round_num=1, argument="x", score=0.3)]
    sessions = [("A", turns), ("B", turns), ("A", turns), ("B", turns)]
    result = aggregate_debate_results(sessions)
    total = sum(result["win_rates"].values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_format_debate_for_training_keys():
    """format_debate_for_training must return dict with prompt, chosen, rejected."""
    turns = [
        DebateTurn(debater_id=0, round_num=1, argument="arg A", score=0.8),
        DebateTurn(debater_id=1, round_num=1, argument="arg B", score=0.2),
    ]
    result = format_debate_for_training(("yes", turns), TOKENIZER_ENCODE)
    assert "prompt" in result
    assert "chosen" in result
    assert "rejected" in result
    assert isinstance(result["prompt"], str)
    assert isinstance(result["chosen"], str)
    assert isinstance(result["rejected"], str)
