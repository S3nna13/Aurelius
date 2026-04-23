"""Tests for src/multiagent/debate_framework.py"""
import pytest
from src.multiagent.debate_framework import (
    DebateConfig,
    DebateRound,
    DebateSession,
    DEBATE_REGISTRY,
)


def make_agent(answer: str):
    def agent_fn(question: str, others: list[str]) -> str:
        return answer
    return agent_fn


def test_debate_session_runs_correct_number_of_rounds():
    cfg = DebateConfig(n_agents=3, n_rounds=2)
    session = DebateSession(cfg)
    session.run("What is 2+2?", make_agent("4"))
    assert len(session.history) == 3  # rounds 0, 1, 2


def test_debate_history_records_all_agent_responses():
    cfg = DebateConfig(n_agents=3, n_rounds=2)
    session = DebateSession(cfg)
    session.run("Q?", make_agent("A"))
    for dr in session.history:
        assert isinstance(dr, DebateRound)
        assert len(dr.responses) == 3


def test_debate_round_indices():
    cfg = DebateConfig(n_agents=2, n_rounds=1)
    session = DebateSession(cfg)
    session.run("Q?", make_agent("X"))
    assert [dr.round_idx for dr in session.history] == [0, 1]


def test_debate_reset_clears_history():
    cfg = DebateConfig(n_agents=2, n_rounds=1)
    session = DebateSession(cfg)
    session.run("Q?", make_agent("X"))
    assert len(session.history) > 0
    session.reset()
    assert session.history == []


def test_debate_final_answer_majority():
    agents_answers = ["yes", "no", "yes"]
    idx = 0

    def rotating_agent(question, others):
        nonlocal idx
        answer = agents_answers[idx % len(agents_answers)]
        idx += 1
        return answer

    cfg = DebateConfig(n_agents=3, n_rounds=0, summarize=True)
    session = DebateSession(cfg)
    result = session.run("Q?", rotating_agent)
    assert result == "yes"


def test_debate_no_summarize_returns_first():
    cfg = DebateConfig(n_agents=3, n_rounds=0, summarize=False)
    session = DebateSession(cfg)
    call_count = 0

    def agent_fn(q, others):
        nonlocal call_count
        call_count += 1
        return f"answer_{call_count}"

    result = session.run("Q?", agent_fn)
    assert result == "answer_1"


def test_debate_round_is_frozen():
    dr = DebateRound(round_idx=0, responses=("a", "b"))
    with pytest.raises((AttributeError, TypeError)):
        dr.round_idx = 1  # type: ignore[misc]


def test_debate_registry_has_default():
    assert "default" in DEBATE_REGISTRY
    assert DEBATE_REGISTRY["default"] is DebateSession
