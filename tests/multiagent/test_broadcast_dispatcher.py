"""Tests for broadcast_dispatcher — topic-based multi-cast on MessageBus."""

from __future__ import annotations

from src.multiagent.broadcast_dispatcher import (
    BROADCAST_DISPATCHER_REGISTRY,
    DEFAULT_BROADCAST_DISPATCHER,
    BroadcastDispatcher,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_dispatcher() -> BroadcastDispatcher:
    return BroadcastDispatcher()


# ---------------------------------------------------------------------------
# Subscription management
# ---------------------------------------------------------------------------


def test_subscribe_adds_agent_to_topic():
    bd = make_dispatcher()
    bd.subscribe("alice", "alerts")
    assert "alice" in bd.subscribers("alerts")


def test_unsubscribe_removes_agent():
    bd = make_dispatcher()
    bd.subscribe("alice", "alerts")
    assert bd.unsubscribe("alice", "alerts") is True
    assert "alice" not in bd.subscribers("alerts")


def test_unsubscribe_unknown_returns_false():
    bd = make_dispatcher()
    assert bd.unsubscribe("alice", "alerts") is False


def test_unsubscribe_cleans_empty_topic():
    bd = make_dispatcher()
    bd.subscribe("alice", "alerts")
    bd.unsubscribe("alice", "alerts")
    assert "alerts" not in bd.topics()


# ---------------------------------------------------------------------------
# Broadcast publish
# ---------------------------------------------------------------------------


def test_publish_delivers_to_all_subscribers():
    bd = make_dispatcher()
    bd.subscribe("alice", "alerts")
    bd.subscribe("bob", "alerts")
    count = bd.publish("alerts", {"severity": "high"}, sender="monitor")
    assert count == 2
    alice_msgs = bd.receive("alice")
    bob_msgs = bd.receive("bob")
    assert len(alice_msgs) == 1
    assert len(bob_msgs) == 1
    assert alice_msgs[0].payload == {"severity": "high"}
    assert alice_msgs[0].sender == "monitor"


def test_publish_returns_zero_when_no_subscribers():
    bd = make_dispatcher()
    assert bd.publish("alerts", "data") == 0


def test_publish_does_not_deliver_to_unsubscribed_agents():
    bd = make_dispatcher()
    bd.subscribe("alice", "alerts")
    bd.publish("alerts", "data")
    assert bd.receive("bob") == []


def test_publish_preserves_topic_as_msg_type():
    bd = make_dispatcher()
    bd.subscribe("alice", "alerts")
    bd.publish("alerts", "data")
    msgs = bd.receive("alice")
    assert msgs[0].msg_type == "alerts"


# ---------------------------------------------------------------------------
# Topics snapshot
# ---------------------------------------------------------------------------


def test_topics_returns_only_nonempty_topics():
    bd = make_dispatcher()
    bd.subscribe("alice", "alerts")
    bd.subscribe("bob", "metrics")
    topics = bd.topics()
    assert "alerts" in topics
    assert "metrics" in topics
    assert len(topics) == 2


# ---------------------------------------------------------------------------
# Pending count delegation
# ---------------------------------------------------------------------------


def test_pending_count_delegates_to_bus():
    bd = make_dispatcher()
    bd.subscribe("alice", "alerts")
    bd.publish("alerts", "ping")
    assert bd.pending_count("alice") == 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in BROADCAST_DISPATCHER_REGISTRY
    assert isinstance(BROADCAST_DISPATCHER_REGISTRY["default"], BroadcastDispatcher)


def test_default_is_broadcast_dispatcher():
    assert isinstance(DEFAULT_BROADCAST_DISPATCHER, BroadcastDispatcher)
