"""Tests for conversation tracker."""

from __future__ import annotations

from src.chat.conversation_tracker import ConversationTracker, Turn


class TestConversationTracker:
    def test_start_and_add_turn(self):
        ct = ConversationTracker()
        ct.start("conv1")
        ct.add_turn("conv1", Turn("user", "hello"))
        conv = ct.get("conv1")
        assert conv is not None
        assert len(conv.turns) == 1

    def test_auto_creates_conversation(self):
        ct = ConversationTracker()
        ct.add_turn("new_conv", Turn("assistant", "hi"))
        conv = ct.get("new_conv")
        assert conv is not None
        assert conv.conversation_id == "new_conv"

    def test_last_n(self):
        ct = ConversationTracker()
        ct.add_turn("c", Turn("user", "1"))
        ct.add_turn("c", Turn("user", "2"))
        ct.add_turn("c", Turn("user", "3"))
        conv = ct.get("c")
        assert len(conv.last_n(2)) == 2

    def test_token_count(self):
        ct = ConversationTracker()
        ct.add_turn("c", Turn("user", "hello world"))
        conv = ct.get("c")
        assert conv.token_count() == 2
