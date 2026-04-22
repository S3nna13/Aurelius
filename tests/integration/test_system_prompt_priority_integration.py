"""Integration: system-prompt priority encoder chat registry."""

from __future__ import annotations

import src.chat as chat
from src.model.config import AureliusConfig


def test_encoder_in_template_registry():
    assert "system_prompt_priority" in chat.CHAT_TEMPLATE_REGISTRY
    assert isinstance(
        chat.CHAT_TEMPLATE_REGISTRY["system_prompt_priority"],
        chat.SystemPromptPriorityEncoder,
    )


def test_fragment_in_message_format_registry():
    assert chat.MESSAGE_FORMAT_REGISTRY["system_prompt_fragment"] is (
        chat.SystemPromptFragment
    )


def test_config_default_off():
    assert AureliusConfig().chat_system_prompt_priority_enabled is False


def test_config_flag_toggleable():
    cfg = AureliusConfig(chat_system_prompt_priority_enabled=True)
    assert cfg.chat_system_prompt_priority_enabled is True


def test_chatml_still_registered():
    assert "chatml" in chat.CHAT_TEMPLATE_REGISTRY


def test_smoke_merge():
    enc = chat.SystemPromptPriorityEncoder()
    frags = [
        chat.SystemPromptFragment(
            priority=chat.SystemPromptPriority.DEVELOPER,
            content="dev",
            source_id="d",
        ),
        chat.SystemPromptFragment(
            priority=chat.SystemPromptPriority.USER,
            content="user",
            source_id="u",
        ),
    ]
    out = enc.merge(frags)
    assert "dev" in out and "user" in out
