"""Integration tests for the threat-intel persona in the chat surface."""

from __future__ import annotations

from src.chat import (
    CHAT_TEMPLATE_REGISTRY,
    MESSAGE_FORMAT_REGISTRY,
)
from src.chat.threat_intel_persona import (
    THREAT_INTEL_SYSTEM_PROMPT,
    ThreatIntelPersona,
)
from src.model.config import AureliusConfig


def test_threat_intel_persona_registered_in_chat_template_registry() -> None:
    assert "threat_intel_persona" in CHAT_TEMPLATE_REGISTRY
    persona = CHAT_TEMPLATE_REGISTRY["threat_intel_persona"]
    assert isinstance(persona, ThreatIntelPersona)


def test_threat_intel_persona_registered_in_message_format_registry() -> None:
    assert "threat_intel_persona" in MESSAGE_FORMAT_REGISTRY
    assert MESSAGE_FORMAT_REGISTRY["threat_intel_persona"] is ThreatIntelPersona


def test_registered_persona_can_classify_and_build() -> None:
    persona = CHAT_TEMPLATE_REGISTRY["threat_intel_persona"]
    assert persona.classify_query("CVE-2021-44228") == "cve"
    msgs = persona.build_messages("CVE-2021-44228")
    assert msgs[0]["content"] == THREAT_INTEL_SYSTEM_PROMPT
    assert msgs[-1]["content"] == "CVE-2021-44228"


def test_config_flag_defaults_off() -> None:
    cfg = AureliusConfig()
    assert hasattr(cfg, "chat_threat_intel_persona_enabled")
    assert cfg.chat_threat_intel_persona_enabled is False


def test_config_flag_is_toggleable() -> None:
    cfg = AureliusConfig(chat_threat_intel_persona_enabled=True)
    assert cfg.chat_threat_intel_persona_enabled is True


def test_other_registered_templates_unchanged() -> None:
    # Additive registration: existing keys still present.
    for key in ("chatml", "llama3", "harmony"):
        assert key in CHAT_TEMPLATE_REGISTRY
