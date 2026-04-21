"""Integration: FiveStateController wired through AGENT_LIFECYCLE_REGISTRY."""

from __future__ import annotations

from src.agent import AGENT_LIFECYCLE_REGISTRY
from src.agent.five_state_controller import AgentState, FiveStateController
from src.model.config import AureliusConfig


def test_controller_registered_in_lifecycle_registry():
    assert "five_state" in AGENT_LIFECYCLE_REGISTRY
    assert AGENT_LIFECYCLE_REGISTRY["five_state"] is FiveStateController


def test_config_flag_defaults_off():
    cfg = AureliusConfig()
    assert cfg.agent_five_state_controller_enabled is False


def test_config_flag_settable():
    cfg = AureliusConfig(agent_five_state_controller_enabled=True)
    assert cfg.agent_five_state_controller_enabled is True


def test_construct_and_initial_state_idle():
    def _stub_backend(messages, control):
        yield "ok"

    ctl = FiveStateController(backend_fn=_stub_backend)
    assert ctl.state == AgentState.IDLE


def test_feature_flag_off_preserves_default_behavior():
    cfg = AureliusConfig()
    assert cfg.agent_five_state_controller_enabled is False
    # Registry remains inspectable regardless of flag value.
    assert "five_state" in AGENT_LIFECYCLE_REGISTRY
