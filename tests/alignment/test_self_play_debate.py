"""Tests for self-play debate helpers."""

import pytest

from src.alignment.self_play_debate import (
    DebateMove,
    alternate_agents,
    self_play_margin,
    summarize_claims,
    winning_agent,
)


def make_moves():
    return [
        DebateMove("alice", "claim one", 0.7),
        DebateMove("bob", "reply", 0.2),
        DebateMove("alice", "claim two", 0.5),
    ]


def test_alternate_agents_builds_round_order():
    assert alternate_agents("alice", "bob", 4) == ["alice", "bob", "alice", "bob"]


def test_self_play_margin_computes_agent_advantage():
    assert self_play_margin(make_moves(), "alice") == pytest.approx(1.0)


def test_winning_agent_returns_correct_agent():
    assert winning_agent(make_moves(), "alice", "bob") == "alice"


def test_winning_agent_returns_tie_when_balanced():
    moves = [DebateMove("alice", "a", 1.0), DebateMove("bob", "b", 1.0)]
    assert winning_agent(moves, "alice", "bob") == "tie"


def test_summarize_claims_returns_top_k_claims():
    assert summarize_claims(make_moves(), top_k=2) == ["claim one", "claim two"]


def test_alternate_agents_rejects_bad_round_count():
    with pytest.raises(ValueError):
        alternate_agents("alice", "bob", -1)


def test_summarize_claims_rejects_bad_top_k():
    with pytest.raises(ValueError):
        summarize_claims(make_moves(), top_k=-1)
