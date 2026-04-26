"""Tests for src/agent/action_scorer.py"""

from __future__ import annotations

import pytest

from src.agent.action_scorer import AGENT_REGISTRY, ActionScore, ActionScorer, ScoreLambdas

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer() -> ActionScorer:
    return ActionScorer()


@pytest.fixture
def default_lambdas() -> ScoreLambdas:
    return ScoreLambdas()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_action_scorer():
    assert "action_scorer" in AGENT_REGISTRY
    assert isinstance(AGENT_REGISTRY["action_scorer"], ActionScorer)


# ---------------------------------------------------------------------------
# estimate_utility
# ---------------------------------------------------------------------------


def test_utility_full_match(scorer):
    action = {"type": "search"}
    state = {"goals": ["search", "retrieve"]}
    assert scorer.estimate_utility(action, state) == 1.0


def test_utility_partial_match(scorer):
    action = {"type": "search"}
    state = {"goals": ["deep_search_result"]}
    assert scorer.estimate_utility(action, state) == 0.5


def test_utility_no_match(scorer):
    action = {"type": "wait"}
    state = {"goals": ["search", "retrieve"]}
    assert scorer.estimate_utility(action, state) == 0.1


def test_utility_no_goals(scorer):
    action = {"type": "search"}
    state = {}
    assert scorer.estimate_utility(action, state) == 0.1


# ---------------------------------------------------------------------------
# estimate_info_gain
# ---------------------------------------------------------------------------


def test_info_gain_info_action(scorer):
    for t in ["search", "retrieve", "query", "fetch", "lookup"]:
        assert scorer.estimate_info_gain({"type": t}, {}) == 1.0


def test_info_gain_write(scorer):
    assert scorer.estimate_info_gain({"type": "write"}, {}) == 0.3


def test_info_gain_create(scorer):
    assert scorer.estimate_info_gain({"type": "create"}, {}) == 0.3


def test_info_gain_wait(scorer):
    assert scorer.estimate_info_gain({"type": "wait"}, {}) == 0.05


def test_info_gain_unknown(scorer):
    assert scorer.estimate_info_gain({"type": "unknown_action"}, {}) == 0.2


# ---------------------------------------------------------------------------
# estimate_risk
# ---------------------------------------------------------------------------


def test_risk_high(scorer):
    for t in ["delete", "write_file", "execute", "shell", "deploy", "drop", "rm"]:
        assert scorer.estimate_risk({"type": t}, {}) == 0.8


def test_risk_medium(scorer):
    for t in ["read_file", "api_call", "network", "http"]:
        assert scorer.estimate_risk({"type": t}, {}) == 0.4


def test_risk_low(scorer):
    assert scorer.estimate_risk({"type": "summarise"}, {}) == 0.1


def test_risk_policy_deny_list(scorer):
    action = {"type": "summarise"}
    policy = {"deny_list": ["summarise"]}
    assert scorer.estimate_risk(action, {}, policy) == 1.0


def test_risk_policy_none(scorer):
    # No policy → falls through to category check
    assert scorer.estimate_risk({"type": "delete"}, {}, None) == 0.8


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


def test_cost_zero_tokens(scorer):
    assert scorer.estimate_cost({}) == 0.0


def test_cost_1000_tokens(scorer):
    assert scorer.estimate_cost({"estimated_tokens": 1000}) == pytest.approx(1.0)


def test_cost_clamped_above_1(scorer):
    assert scorer.estimate_cost({"estimated_tokens": 5000}) == 1.0


def test_cost_partial(scorer):
    assert scorer.estimate_cost({"estimated_tokens": 500}) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def test_score_returns_action_score(scorer):
    result = scorer.score({"type": "search"}, {"goals": ["search"]})
    assert isinstance(result, ActionScore)


def test_score_total_formula(scorer):
    lam = ScoreLambdas(utility=1.0, info_gain=0.5, risk=-1.5, cost=-0.2)
    action = {"type": "search", "estimated_tokens": 0}
    state = {"goals": ["search"]}
    result = scorer.score(action, state, lam)
    expected = 1.0 * 1.0 + 0.5 * 1.0 + (-1.5) * 0.1 + (-0.2) * 0.0
    assert result.total == pytest.approx(expected)


def test_score_default_lambdas(scorer):
    # Should not raise; default lambdas are used
    result = scorer.score({"type": "wait"}, {})
    assert isinstance(result.total, float)


# ---------------------------------------------------------------------------
# rank_actions
# ---------------------------------------------------------------------------


def test_rank_actions_descending(scorer):
    actions = [
        {"type": "wait"},
        {"type": "search"},
        {"type": "delete"},
    ]
    state = {"goals": ["search"]}
    ranked = scorer.rank_actions(actions, state)
    totals = [s.total for _, s in ranked]
    assert totals == sorted(totals, reverse=True)


def test_rank_actions_returns_all(scorer):
    actions = [{"type": str(i)} for i in range(5)]
    ranked = scorer.rank_actions(actions, {})
    assert len(ranked) == 5


def test_rank_actions_empty(scorer):
    assert scorer.rank_actions([], {}) == []
