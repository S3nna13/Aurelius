"""Tests for message bus."""
from __future__ import annotations

import pytest

from src.multiagent.message_bus import MessageBus, AgentMessage


class TestMessageBus:
    def test_send_and_receive(self):
        mb = MessageBus()
        mb.send(AgentMessage("alice", "bob", "greet", "hello"))
        msgs = mb.receive("bob")
        assert len(msgs) == 1
        assert msgs[0].payload == "hello"

    def test_receive_clears_inbox(self):
        mb = MessageBus()
        mb.send(AgentMessage("a", "b", "t", "data"))
        mb.receive("b")
        assert mb.pending_count("b") == 0

    def test_pending_count(self):
        mb = MessageBus()
        mb.send(AgentMessage("a", "b", "t"))
        mb.send(AgentMessage("c", "d", "t"))
        assert mb.pending_count() == 2