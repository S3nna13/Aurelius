"""Tests for src/multiagent/negotiation_protocol.py."""

from __future__ import annotations

import pytest

from src.multiagent.negotiation_protocol import (
    NEGOTIATION_PROTOCOL_REGISTRY,
    NegotiationProtocol,
    NegotiationSession,
    NegotiationState,
    Offer,
)


def _proto() -> NegotiationProtocol:
    return NegotiationProtocol()


def test_registry_has_default():
    assert "default" in NEGOTIATION_PROTOCOL_REGISTRY
    assert NEGOTIATION_PROTOCOL_REGISTRY["default"] is NegotiationProtocol


def test_state_values():
    assert NegotiationState.OPEN.value == "open"
    assert NegotiationState.AGREED.value == "agreed"
    assert NegotiationState.FAILED.value == "failed"


def test_create_session_basic():
    p = _proto()
    s = p.create_session(["a", "b"])
    assert isinstance(s, NegotiationSession)
    assert s.participants == ["a", "b"]
    assert s.state == NegotiationState.OPEN
    assert s.offers == []
    assert s.max_rounds == 10


def test_create_session_custom_max_rounds():
    p = _proto()
    s = p.create_session(["a", "b"], max_rounds=3)
    assert s.max_rounds == 3


def test_create_session_auto_id_len_8():
    p = _proto()
    s = p.create_session(["a"])
    assert len(s.session_id) == 8


def test_session_ids_unique():
    p = _proto()
    ids = {p.create_session(["a"]).session_id for _ in range(20)}
    assert len(ids) == 20


def test_submit_offer_returns_offer():
    p = _proto()
    s = p.create_session(["a", "b"])
    o = p.submit_offer(s, "a", 100.0)
    assert isinstance(o, Offer)
    assert o.agent_id == "a"
    assert o.value == 100.0
    assert o.round_num == 0


def test_submit_offer_appends_to_session():
    p = _proto()
    s = p.create_session(["a", "b"])
    p.submit_offer(s, "a", 10.0)
    assert len(s.offers) == 1


def test_submit_offer_transitions_to_in_progress():
    p = _proto()
    s = p.create_session(["a", "b"])
    assert s.state == NegotiationState.OPEN
    p.submit_offer(s, "a", 1.0)
    assert s.state == NegotiationState.IN_PROGRESS


def test_submit_offer_round_tracking():
    p = _proto()
    s = p.create_session(["a", "b"])
    o1 = p.submit_offer(s, "a", 10.0)
    o2 = p.submit_offer(s, "b", 11.0)
    o3 = p.submit_offer(s, "a", 12.0)
    assert o1.round_num == 0
    assert o2.round_num == 0
    assert o3.round_num == 1


def test_submit_offer_with_terms():
    p = _proto()
    s = p.create_session(["a", "b"])
    o = p.submit_offer(s, "a", 5.0, terms={"currency": "USD"})
    assert o.terms == {"currency": "USD"}


def test_submit_offer_default_terms_empty():
    p = _proto()
    s = p.create_session(["a", "b"])
    o = p.submit_offer(s, "a", 5.0)
    assert o.terms == {}


def test_current_round_empty_session():
    p = _proto()
    s = p.create_session(["a", "b"])
    assert p.current_round(s) == 0


def test_current_round_after_partial():
    p = _proto()
    s = p.create_session(["a", "b"])
    p.submit_offer(s, "a", 1.0)
    assert p.current_round(s) == 0


def test_current_round_after_full_round():
    p = _proto()
    s = p.create_session(["a", "b"])
    p.submit_offer(s, "a", 1.0)
    p.submit_offer(s, "b", 2.0)
    assert p.current_round(s) == 1


def test_current_round_empty_participants():
    p = _proto()
    s = NegotiationSession(participants=[])
    assert p.current_round(s) == 0


def test_latest_offers_empty():
    p = _proto()
    s = p.create_session(["a", "b"])
    assert p.latest_offers(s) == []


def test_latest_offers_partial_round():
    p = _proto()
    s = p.create_session(["a", "b"])
    p.submit_offer(s, "a", 10.0)
    out = p.latest_offers(s)
    assert len(out) == 1
    assert out[0].agent_id == "a"


def test_latest_offers_full_round():
    p = _proto()
    s = p.create_session(["a", "b"])
    p.submit_offer(s, "a", 10.0)
    p.submit_offer(s, "b", 11.0)
    out = p.latest_offers(s)
    assert len(out) == 2


def test_latest_offers_second_round_partial():
    p = _proto()
    s = p.create_session(["a", "b"])
    p.submit_offer(s, "a", 10.0)
    p.submit_offer(s, "b", 11.0)
    p.submit_offer(s, "a", 100.0)
    out = p.latest_offers(s)
    assert len(out) == 1
    assert out[0].round_num == 1


def test_evaluate_offers_none_when_incomplete():
    p = _proto()
    s = p.create_session(["a", "b"])
    p.submit_offer(s, "a", 100.0)
    assert p.evaluate_offers(s) is None


def test_evaluate_offers_agreement_within_5pct():
    p = _proto()
    s = p.create_session(["a", "b"])
    p.submit_offer(s, "a", 100.0)
    p.submit_offer(s, "b", 102.0)
    result = p.evaluate_offers(s)
    assert result is not None
    assert s.state == NegotiationState.AGREED
    assert s.agreed_offer is result


def test_evaluate_offers_no_agreement_far_apart():
    p = _proto()
    s = p.create_session(["a", "b"], max_rounds=5)
    p.submit_offer(s, "a", 100.0)
    p.submit_offer(s, "b", 200.0)
    result = p.evaluate_offers(s)
    assert result is None
    assert s.state == NegotiationState.IN_PROGRESS


def test_evaluate_offers_failed_on_max_rounds():
    p = _proto()
    s = p.create_session(["a", "b"], max_rounds=1)
    p.submit_offer(s, "a", 10.0)
    p.submit_offer(s, "b", 100.0)
    p.evaluate_offers(s)
    assert s.state == NegotiationState.FAILED


def test_evaluate_offers_picks_closest_to_mean():
    p = _proto()
    s = p.create_session(["a", "b", "c"])
    p.submit_offer(s, "a", 100.0)
    p.submit_offer(s, "b", 101.0)
    p.submit_offer(s, "c", 102.0)
    result = p.evaluate_offers(s)
    assert result is not None
    assert result.agent_id == "b"


def test_evaluate_offers_empty_returns_none():
    p = _proto()
    s = p.create_session(["a", "b"])
    assert p.evaluate_offers(s) is None


def test_offer_auto_id_len_8():
    p = _proto()
    s = p.create_session(["a"])
    o = p.submit_offer(s, "a", 1.0)
    assert len(o.offer_id) == 8


def test_offer_frozen():
    o = Offer(agent_id="a", value=1.0)
    with pytest.raises(Exception):
        o.value = 2.0  # type: ignore[misc]


def test_three_party_negotiation_flow():
    p = _proto()
    s = p.create_session(["a", "b", "c"], max_rounds=5)
    # Round 0 - far apart
    p.submit_offer(s, "a", 10.0)
    p.submit_offer(s, "b", 50.0)
    p.submit_offer(s, "c", 100.0)
    assert p.evaluate_offers(s) is None
    # Round 1 - converged
    p.submit_offer(s, "a", 50.0)
    p.submit_offer(s, "b", 51.0)
    p.submit_offer(s, "c", 52.0)
    agreed = p.evaluate_offers(s)
    assert agreed is not None
    assert s.state == NegotiationState.AGREED
