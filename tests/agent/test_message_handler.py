"""Tests for message_handler — agent-bound message routing."""

from __future__ import annotations

import pytest

from src.agent.message_handler import (
    DEFAULT_MESSAGE_HANDLER,
    MESSAGE_HANDLER_REGISTRY,
    MessageHandler,
)
from src.multiagent.message_bus import AgentMessage

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_and_handle():
    mh = MessageHandler()
    calls = []
    mh.register("alert", lambda msg: calls.append(msg.payload))
    mh.handle(AgentMessage("sys", "agent", "alert", "fire"))
    assert calls == ["fire"]


def test_unregister_returns_true_when_present():
    mh = MessageHandler()
    mh.register("x", lambda msg: None)
    assert mh.unregister("x") is True
    assert mh.can_handle("x") is False


def test_unregister_returns_false_when_missing():
    mh = MessageHandler()
    assert mh.unregister("x") is False


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_handle_unknown_type_raises():
    mh = MessageHandler()
    with pytest.raises(ValueError, match="No handler registered"):
        mh.handle(AgentMessage("sys", "agent", "unknown", "data"))


def test_can_handle_true_for_registered():
    mh = MessageHandler()
    mh.register("x", lambda msg: None)
    assert mh.can_handle("x") is True


def test_known_types_snapshot():
    mh = MessageHandler()
    mh.register("a", lambda msg: None)
    mh.register("b", lambda msg: None)
    assert set(mh.known_types()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Handler return values
# ---------------------------------------------------------------------------


def test_handle_returns_handler_result():
    mh = MessageHandler()
    mh.register("add", lambda msg: msg.payload + 1)
    result = mh.handle(AgentMessage("sys", "agent", "add", 5))
    assert result == 6


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in MESSAGE_HANDLER_REGISTRY
    assert isinstance(MESSAGE_HANDLER_REGISTRY["default"], MessageHandler)


def test_default_is_message_handler():
    assert isinstance(DEFAULT_MESSAGE_HANDLER, MessageHandler)
