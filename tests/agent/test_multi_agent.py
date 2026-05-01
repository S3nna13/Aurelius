"""Tests for src.agent.multi_agent."""

from __future__ import annotations

from src.agent.agent_runtime import AgentRuntime, AgentSpec
from src.agent.multi_agent import SupervisorCoordinator, SwarmCoordinator


def _agent(name: str) -> AgentSpec:
    return AgentSpec(name=name, role="worker")


def test_delegate_returns_only_current_task_messages() -> None:
    runtime = AgentRuntime()
    coord = SupervisorCoordinator(runtime)
    coord.set_supervisor(_agent("supervisor"))
    coord.add_worker(_agent("worker"))

    first = coord.delegate("task one")
    second = coord.delegate("task two")

    assert first == ["task one"]
    assert second == ["task two"]


def test_broadcast_returns_only_current_broadcast_messages() -> None:
    runtime = AgentRuntime()
    coord = SwarmCoordinator(runtime)
    sender = coord.add_agent(_agent("sender"))
    recipient = coord.add_agent(_agent("recipient"))

    first = coord.broadcast("hello", sender)
    second = coord.broadcast("world", sender)

    assert first == [f"{recipient}: hello"]
    assert second == [f"{recipient}: world"]
