"""Integration tests for AgentSwarm via AGENT_LOOP_REGISTRY.

Verifies:
1. "agent_swarm" key is present in AGENT_LOOP_REGISTRY.
2. Constructing AgentSwarm from the registry and dispatching 3 mock tasks
   yields 3 results.
3. A pre-existing AGENT_LOOP_REGISTRY key ("react") is still present — no
   accidental registry mutation.
"""

from __future__ import annotations

import pytest

from src.agent import AGENT_LOOP_REGISTRY
from src.agent.agent_swarm import AgentSwarm, SubAgentResult


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_agent_swarm_in_registry():
    """'agent_swarm' must be a key in AGENT_LOOP_REGISTRY."""
    assert "agent_swarm" in AGENT_LOOP_REGISTRY


def test_registry_construct_and_dispatch_three_tasks():
    """Construct AgentSwarm from registry; dispatch 3 tasks; get 3 results."""
    SwarmCls = AGENT_LOOP_REGISTRY["agent_swarm"]
    swarm = SwarmCls(orchestrator_max_steps=10, subagent_max_steps=50)

    def mock_subagent(task, max_steps: int) -> SubAgentResult:
        return SubAgentResult(
            task_id=task["id"],
            result=f"done:{task['id']}",
            steps_used=3,
            status="completed",
        )

    tasks = [{"id": 1}, {"id": 2}, {"id": 3}]
    results = swarm.dispatch(tasks, mock_subagent)

    assert len(results) == 3
    assert all(isinstance(r, SubAgentResult) for r in results)
    assert [r.task_id for r in results] == [1, 2, 3]
    assert all(r.status == "completed" for r in results)


def test_existing_registry_key_still_present():
    """Pre-existing 'react' key must remain in AGENT_LOOP_REGISTRY."""
    assert "react" in AGENT_LOOP_REGISTRY
