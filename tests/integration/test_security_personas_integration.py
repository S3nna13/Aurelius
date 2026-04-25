"""Integration tests for security personas in the chat surface."""

from __future__ import annotations

from src.chat import CHAT_TEMPLATE_REGISTRY, MESSAGE_FORMAT_REGISTRY
from src.chat.security_personas import (
    BLUE_TEAM_PERSONA,
    DEFAULT_SECURITY_PERSONA_REGISTRY,
    PURPLE_TEAM_PERSONA,
    RED_TEAM_PERSONA,
    SecurityPersona,
    SecurityPersonaRegistry,
)
from src.model.config import AureliusConfig
from src.runtime.feature_flags import FeatureFlag, FEATURE_FLAG_REGISTRY


def test_security_personas_registered_in_chat_template_registry() -> None:
    assert "security_personas" in CHAT_TEMPLATE_REGISTRY
    reg = CHAT_TEMPLATE_REGISTRY["security_personas"]
    assert isinstance(reg, SecurityPersonaRegistry)


def test_security_personas_registered_in_message_format_registry() -> None:
    assert "security_persona" in MESSAGE_FORMAT_REGISTRY
    assert MESSAGE_FORMAT_REGISTRY["security_persona"] is SecurityPersona


def test_registered_registry_contains_three_personas() -> None:
    reg = CHAT_TEMPLATE_REGISTRY["security_personas"]
    ids = {p.id for p in reg.all()}
    assert ids == {"red_team", "blue_team", "purple_team"}


def test_registered_registry_build_messages() -> None:
    reg = CHAT_TEMPLATE_REGISTRY["security_personas"]
    msgs = reg.build_messages("red_team", "enumerate auth01.lab.local")
    assert msgs[0]["content"] == RED_TEAM_PERSONA.system_prompt
    assert msgs[-1]["content"] == "enumerate auth01.lab.local"


def test_default_registry_identity() -> None:
    # The registry placed in CHAT_TEMPLATE_REGISTRY is the module default.
    assert CHAT_TEMPLATE_REGISTRY["security_personas"] is DEFAULT_SECURITY_PERSONA_REGISTRY


def test_config_flag_defaults_off() -> None:
    cfg = AureliusConfig()
    assert hasattr(cfg, "chat_security_personas_enabled")
    assert cfg.chat_security_personas_enabled is False


def test_config_flag_is_toggleable() -> None:
    FEATURE_FLAG_REGISTRY.register(FeatureFlag(name="chat.security_personas", enabled=True))
    cfg = AureliusConfig()
    assert cfg.chat_security_personas_enabled is True


def test_other_registered_templates_unchanged() -> None:
    for key in ("chatml", "llama3", "harmony", "threat_intel_persona"):
        assert key in CHAT_TEMPLATE_REGISTRY


def test_personas_exposed_as_module_constants() -> None:
    assert RED_TEAM_PERSONA.id == "red_team"
    assert BLUE_TEAM_PERSONA.id == "blue_team"
    assert PURPLE_TEAM_PERSONA.id == "purple_team"
