"""Tests for src/multiagent/role_play_manager.py"""

import pytest

from src.multiagent.role_play_manager import (
    ROLE_PLAY_REGISTRY,
    AgentRole,
    RolePlayConfig,
    RolePlayManager,
    Utterance,
)


def make_manager(*roles: AgentRole, **kwargs) -> RolePlayManager:
    cfg = RolePlayConfig(roles=list(roles), **kwargs)
    return RolePlayManager(cfg)


def test_add_role_sorts_by_turn_order():
    mgr = RolePlayManager()
    mgr.add_role(AgentRole("B", "persona B", turn_order=2))
    mgr.add_role(AgentRole("A", "persona A", turn_order=1))
    mgr.add_role(AgentRole("C", "persona C", turn_order=0))
    names = [r.name for r in mgr._config.roles]
    assert names == ["C", "A", "B"]


def test_current_role_returns_none_when_no_roles():
    mgr = RolePlayManager()
    assert mgr.current_role() is None


def test_current_role_cycles():
    role_a = AgentRole("A", "persona A", turn_order=0)
    role_b = AgentRole("B", "persona B", turn_order=1)
    mgr = make_manager(role_a, role_b)
    assert mgr.current_role().name == "A"
    mgr.record("A", "hello")
    assert mgr.current_role().name == "B"
    mgr.record("B", "hi")
    assert mgr.current_role().name == "A"


def test_record_appends_utterance():
    mgr = RolePlayManager()
    mgr.record("Alice", "Hello")
    mgr.record("Bob", "World")
    t = mgr.transcript()
    assert len(t) == 2
    assert t[0].role_name == "Alice"
    assert t[1].content == "World"


def test_record_increments_turn():
    mgr = RolePlayManager()
    mgr.record("Alice", "msg1")
    mgr.record("Bob", "msg2")
    t = mgr.transcript()
    assert t[0].turn == 0
    assert t[1].turn == 1


def test_transcript_length():
    mgr = RolePlayManager()
    for i in range(5):
        mgr.record("R", f"msg{i}")
    assert len(mgr.transcript()) == 5


def test_reset_clears_transcript_and_turn():
    mgr = RolePlayManager()
    mgr.record("X", "content")
    mgr.reset()
    assert mgr.transcript() == []
    assert mgr._turn == 0


def test_format_for_agent_general_audience():
    mgr = RolePlayManager(RolePlayConfig(audience="general"))
    role = AgentRole("Narrator", "Tell the story.")
    prompt = mgr.format_for_agent(role)
    assert len(prompt) > 0
    assert "Narrator" in prompt


def test_format_for_agent_expert_audience():
    mgr = RolePlayManager(RolePlayConfig(audience="expert"))
    role = AgentRole("Scientist", "Analyze data.")
    prompt = mgr.format_for_agent(role)
    assert (
        "expert" in prompt.lower() or "technical" in prompt.lower() or "precise" in prompt.lower()
    )


def test_format_for_agent_child_audience():
    mgr = RolePlayManager(RolePlayConfig(audience="child"))
    role = AgentRole("Teacher", "Explain math.")
    prompt = mgr.format_for_agent(role)
    assert "child" in prompt.lower() or "simple" in prompt.lower() or "friendly" in prompt.lower()


def test_utterance_is_frozen():
    u = Utterance(role_name="A", content="hello", turn=0)
    with pytest.raises((AttributeError, TypeError)):
        u.content = "changed"  # type: ignore[misc]


def test_role_play_registry_has_default():
    assert "default" in ROLE_PLAY_REGISTRY
    assert ROLE_PLAY_REGISTRY["default"] is RolePlayManager
