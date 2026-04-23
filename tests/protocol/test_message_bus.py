"""Tests for src/protocol/message_bus.py (~50 tests)."""
import pytest
from src.protocol.message_bus import (
    AgentMessage,
    MessageBus,
    MessageEnvelope,
    MESSAGE_BUS,
)


# ---------------------------------------------------------------------------
# MessageEnvelope
# ---------------------------------------------------------------------------

def make_envelope(**kwargs):
    defaults = dict(sender="alice", recipient="bob", topic="chat", payload={})
    defaults.update(kwargs)
    return MessageEnvelope(**defaults)


class TestMessageEnvelope:
    def test_id_auto_generated(self):
        e = make_envelope()
        assert e.id is not None

    def test_id_is_8_chars(self):
        e = make_envelope()
        assert len(e.id) == 8

    def test_id_is_hex(self):
        e = make_envelope()
        int(e.id, 16)  # raises if not valid hex

    def test_two_envelopes_have_different_ids(self):
        e1 = make_envelope()
        e2 = make_envelope()
        assert e1.id != e2.id

    def test_timestamp_is_string(self):
        e = make_envelope()
        assert isinstance(e.timestamp, str)

    def test_timestamp_nonempty(self):
        e = make_envelope()
        assert len(e.timestamp) > 0

    def test_timestamp_iso_format(self):
        from datetime import datetime
        e = make_envelope()
        # Should parse without error
        datetime.fromisoformat(e.timestamp)

    def test_sender_stored(self):
        e = make_envelope(sender="s1")
        assert e.sender == "s1"

    def test_recipient_stored(self):
        e = make_envelope(recipient="r1")
        assert e.recipient == "r1"

    def test_topic_stored(self):
        e = make_envelope(topic="events")
        assert e.topic == "events"

    def test_payload_stored(self):
        p = {"key": "value"}
        e = make_envelope(payload=p)
        assert e.payload == p

    def test_priority_default_zero(self):
        e = make_envelope()
        assert e.priority == 0

    def test_priority_can_be_set(self):
        e = make_envelope(priority=5)
        assert e.priority == 5

    def test_explicit_id_override(self):
        e = make_envelope()
        e2 = MessageEnvelope(sender="a", recipient="b", topic="t", payload={}, id="abcd1234")
        assert e2.id == "abcd1234"


# ---------------------------------------------------------------------------
# AgentMessage
# ---------------------------------------------------------------------------

def make_message(**kwargs):
    envelope = make_envelope()
    return AgentMessage(envelope=envelope, **kwargs)


class TestAgentMessage:
    def test_envelope_stored(self):
        env = make_envelope()
        msg = AgentMessage(envelope=env)
        assert msg.envelope is env

    def test_ack_required_default_false(self):
        msg = make_message()
        assert msg.ack_required is False

    def test_ack_required_can_be_set(self):
        msg = make_message(ack_required=True)
        assert msg.ack_required is True

    def test_ttl_default_60(self):
        msg = make_message()
        assert msg.ttl_seconds == 60.0

    def test_ttl_can_be_set(self):
        msg = make_message(ttl_seconds=120.0)
        assert msg.ttl_seconds == 120.0


# ---------------------------------------------------------------------------
# MessageBus
# ---------------------------------------------------------------------------

class TestMessageBus:
    def setup_method(self):
        self.bus = MessageBus()

    def test_subscribe_and_subscribers(self):
        self.bus.subscribe("topic_a", "sub1")
        assert "sub1" in self.bus.subscribers("topic_a")

    def test_subscribers_empty_for_unknown_topic(self):
        assert self.bus.subscribers("nonexistent") == []

    def test_multiple_subscribers_on_same_topic(self):
        self.bus.subscribe("t", "s1")
        self.bus.subscribe("t", "s2")
        subs = self.bus.subscribers("t")
        assert "s1" in subs
        assert "s2" in subs

    def test_duplicate_subscribe_no_duplication(self):
        self.bus.subscribe("t", "s1")
        self.bus.subscribe("t", "s1")
        assert self.bus.subscribers("t").count("s1") == 1

    def test_unsubscribe_returns_true_when_was_subscribed(self):
        self.bus.subscribe("t", "s1")
        result = self.bus.unsubscribe("t", "s1")
        assert result is True

    def test_unsubscribe_returns_false_when_not_subscribed(self):
        result = self.bus.unsubscribe("t", "nonexistent")
        assert result is False

    def test_unsubscribe_removes_subscriber(self):
        self.bus.subscribe("t", "s1")
        self.bus.unsubscribe("t", "s1")
        assert "s1" not in self.bus.subscribers("t")

    def test_publish_delivers_to_subscriber(self):
        self.bus.subscribe("news", "reader1")
        env = MessageEnvelope(sender="pub", recipient="all", topic="news", payload={"x": 1})
        msg = AgentMessage(envelope=env)
        self.bus.publish(msg)
        received = self.bus.receive("reader1")
        assert len(received) == 1
        assert received[0] is msg

    def test_publish_returns_subscriber_ids(self):
        self.bus.subscribe("alerts", "s1")
        self.bus.subscribe("alerts", "s2")
        env = MessageEnvelope(sender="pub", recipient="all", topic="alerts", payload={})
        msg = AgentMessage(envelope=env)
        delivered = self.bus.publish(msg)
        assert set(delivered) == {"s1", "s2"}

    def test_publish_returns_empty_for_no_subscribers(self):
        env = MessageEnvelope(sender="pub", recipient="all", topic="empty_topic", payload={})
        msg = AgentMessage(envelope=env)
        delivered = self.bus.publish(msg)
        assert delivered == []

    def test_publish_delivers_to_all_topic_subscribers(self):
        self.bus.subscribe("multi", "a")
        self.bus.subscribe("multi", "b")
        self.bus.subscribe("multi", "c")
        env = MessageEnvelope(sender="src", recipient="all", topic="multi", payload={})
        msg = AgentMessage(envelope=env)
        self.bus.publish(msg)
        for sub in ["a", "b", "c"]:
            assert len(self.bus.receive(sub)) == 1

    def test_publish_does_not_deliver_to_different_topic_subscribers(self):
        self.bus.subscribe("topic_x", "sx")
        self.bus.subscribe("topic_y", "sy")
        env = MessageEnvelope(sender="src", recipient="all", topic="topic_x", payload={})
        msg = AgentMessage(envelope=env)
        self.bus.publish(msg)
        assert self.bus.receive("sy") == []

    def test_receive_drains_mailbox(self):
        self.bus.subscribe("t", "s1")
        env = MessageEnvelope(sender="a", recipient="b", topic="t", payload={})
        msg = AgentMessage(envelope=env)
        self.bus.publish(msg)
        self.bus.receive("s1")
        assert self.bus.receive("s1") == []

    def test_receive_unknown_subscriber_returns_empty(self):
        assert self.bus.receive("nobody") == []

    def test_pending_count_zero_initially(self):
        self.bus.subscribe("t", "s1")
        assert self.bus.pending_count("s1") == 0

    def test_pending_count_increases_with_publish(self):
        self.bus.subscribe("t", "s1")
        for _ in range(3):
            env = MessageEnvelope(sender="a", recipient="b", topic="t", payload={})
            msg = AgentMessage(envelope=env)
            self.bus.publish(msg)
        assert self.bus.pending_count("s1") == 3

    def test_pending_count_resets_after_receive(self):
        self.bus.subscribe("t", "s1")
        env = MessageEnvelope(sender="a", recipient="b", topic="t", payload={})
        msg = AgentMessage(envelope=env)
        self.bus.publish(msg)
        self.bus.receive("s1")
        assert self.bus.pending_count("s1") == 0

    def test_pending_count_unknown_subscriber_is_zero(self):
        assert self.bus.pending_count("nobody") == 0

    def test_multiple_messages_in_order(self):
        self.bus.subscribe("ordered", "s1")
        msgs = []
        for i in range(5):
            env = MessageEnvelope(sender="a", recipient="b", topic="ordered", payload={"i": i})
            msg = AgentMessage(envelope=env)
            self.bus.publish(msg)
            msgs.append(msg)
        received = self.bus.receive("s1")
        for i, msg in enumerate(msgs):
            assert received[i] is msg

    def test_subscribers_returns_list(self):
        self.bus.subscribe("t", "s1")
        result = self.bus.subscribers("t")
        assert isinstance(result, list)

    def test_message_bus_singleton_exists(self):
        assert MESSAGE_BUS is not None
        assert isinstance(MESSAGE_BUS, MessageBus)
