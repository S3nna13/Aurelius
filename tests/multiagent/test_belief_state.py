"""Tests for src/multiagent/belief_state.py."""

from __future__ import annotations

import pytest

from src.multiagent.belief_state import (
    BELIEF_STATE_REGISTRY,
    BeliefState,
    BeliefUpdate,
    Proposition,
)


def _bs() -> BeliefState:
    return BeliefState("agent1")


def test_registry_has_default():
    assert "default" in BELIEF_STATE_REGISTRY
    assert BELIEF_STATE_REGISTRY["default"] is BeliefState


def test_init_agent_id():
    b = BeliefState("alice")
    assert b.agent_id == "alice"
    assert b.all_propositions() == []


def test_believe_creates_proposition():
    b = _bs()
    p = b.believe("rain", 0.7)
    assert isinstance(p, Proposition)
    assert p.name == "rain"
    assert p.probability == 0.7


def test_believe_default_confidence():
    b = _bs()
    p = b.believe("rain", 0.5)
    assert p.confidence == 1.0


def test_believe_with_confidence_and_evidence():
    b = _bs()
    p = b.believe("rain", 0.5, confidence=0.8, evidence=["cloud"])
    assert p.confidence == 0.8
    assert p.evidence == ["cloud"]


def test_believe_clamps_probability_high():
    b = _bs()
    p = b.believe("x", 1.5)
    assert p.probability == 1.0


def test_believe_clamps_probability_low():
    b = _bs()
    p = b.believe("x", -0.5)
    assert p.probability == 0.0


def test_believe_clamps_confidence():
    b = _bs()
    p = b.believe("x", 0.5, confidence=2.0)
    assert p.confidence == 1.0


def test_believe_overwrites_same_name():
    b = _bs()
    b.believe("x", 0.3)
    b.believe("x", 0.9)
    assert b.get("x").probability == 0.9
    assert len(b.all_propositions()) == 1


def test_update_existing_returns_update():
    b = _bs()
    b.believe("rain", 0.3)
    upd = b.update("rain", 0.8, reason="new data")
    assert isinstance(upd, BeliefUpdate)
    assert upd.proposition_name == "rain"
    assert upd.old_probability == 0.3
    assert upd.new_probability == 0.8
    assert upd.reason == "new data"


def test_update_changes_stored_probability():
    b = _bs()
    b.believe("rain", 0.3)
    b.update("rain", 0.9)
    assert b.get("rain").probability == 0.9


def test_update_unknown_returns_none():
    b = _bs()
    assert b.update("missing", 0.5) is None


def test_update_clamps_probability():
    b = _bs()
    b.believe("x", 0.5)
    upd = b.update("x", 1.7)
    assert upd.new_probability == 1.0


def test_update_preserves_confidence():
    b = _bs()
    b.believe("x", 0.5, confidence=0.7)
    b.update("x", 0.9)
    assert b.get("x").confidence == 0.7


def test_get_existing():
    b = _bs()
    b.believe("x", 0.4)
    p = b.get("x")
    assert p is not None
    assert p.name == "x"


def test_get_missing_returns_none():
    b = _bs()
    assert b.get("ghost") is None


def test_most_certain_orders_by_product():
    b = _bs()
    b.believe("a", 0.9, confidence=0.9)  # 0.81
    b.believe("b", 0.5, confidence=0.5)  # 0.25
    b.believe("c", 1.0, confidence=1.0)  # 1.0
    top = b.most_certain(k=3)
    assert [p.name for p in top] == ["c", "a", "b"]


def test_most_certain_top_k():
    b = _bs()
    for i in range(10):
        b.believe(f"p{i}", i / 10.0, confidence=1.0)
    out = b.most_certain(k=3)
    assert len(out) == 3


def test_most_certain_empty():
    b = _bs()
    assert b.most_certain() == []


def test_contradictions_detects_not_x():
    b = _bs()
    b.believe("rain", 0.8)
    b.believe("not_rain", 0.2)
    out = b.contradictions()
    assert len(out) == 1
    assert set(out[0]) == {"rain", "not_rain"}


def test_contradictions_detects_no_x():
    b = _bs()
    b.believe("fly", 0.7)
    b.believe("no_fly", 0.3)
    out = b.contradictions()
    assert len(out) == 1
    assert set(out[0]) == {"fly", "no_fly"}


def test_contradictions_none_when_unrelated():
    b = _bs()
    b.believe("rain", 0.8)
    b.believe("cold", 0.4)
    assert b.contradictions() == []


def test_contradictions_no_match_when_not_consistent():
    b = _bs()
    b.believe("rain", 0.9)
    b.believe("not_rain", 0.9)  # probabilities don't sum close to 1
    assert b.contradictions() == []


def test_contradictions_requires_both():
    b = _bs()
    b.believe("not_rain", 0.3)
    assert b.contradictions() == []


def test_bayesian_update_basic_formula():
    b = _bs()
    b.believe("disease", 0.01)
    # P(E|H)=0.9, P(E)=0.1, prior=0.01 -> 0.09
    upd = b.bayesian_update("disease", likelihood_given_true=0.9, prior_evidence_prob=0.1)
    assert upd is not None
    assert abs(upd.new_probability - 0.09) < 1e-9
    assert upd.reason == "bayesian_update"


def test_bayesian_update_clamps_to_one():
    b = _bs()
    b.believe("x", 0.5)
    upd = b.bayesian_update("x", likelihood_given_true=1.0, prior_evidence_prob=0.1)
    assert upd.new_probability == 1.0


def test_bayesian_update_unknown_returns_none():
    b = _bs()
    assert b.bayesian_update("nope", 0.5, 0.5) is None


def test_bayesian_update_zero_evidence_returns_none():
    b = _bs()
    b.believe("x", 0.5)
    assert b.bayesian_update("x", 0.5, 0.0) is None


def test_bayesian_update_stores_new_probability():
    b = _bs()
    b.believe("x", 0.2)
    b.bayesian_update("x", 0.8, 0.4)
    # 0.8 * 0.2 / 0.4 = 0.4
    assert abs(b.get("x").probability - 0.4) < 1e-9


def test_all_propositions_returns_all():
    b = _bs()
    b.believe("a", 0.1)
    b.believe("b", 0.2)
    b.believe("c", 0.3)
    assert len(b.all_propositions()) == 3


def test_proposition_frozen():
    p = Proposition(name="x", probability=0.5)
    with pytest.raises(Exception):
        p.probability = 0.9  # type: ignore[misc]


def test_belief_update_frozen():
    u = BeliefUpdate(
        proposition_name="x",
        old_probability=0.1,
        new_probability=0.2,
        reason="r",
    )
    with pytest.raises(Exception):
        u.reason = "other"  # type: ignore[misc]
