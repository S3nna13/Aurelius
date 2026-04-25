"""Tests for src/multiagent/agent_pool.py."""
from __future__ import annotations

import pytest

from src.multiagent.agent_pool import (
    AgentPool,
    PoolAgent,
    PoolAgentStatus,
    AGENT_POOL,
)


# ---------------------------------------------------------------------------
# PoolAgentStatus enum
# ---------------------------------------------------------------------------

def test_pool_agent_status_values():
    assert PoolAgentStatus.IDLE == "idle"
    assert PoolAgentStatus.BUSY == "busy"
    assert PoolAgentStatus.DRAINING == "draining"
    assert PoolAgentStatus.RETIRED == "retired"


def test_pool_agent_status_count():
    assert len(PoolAgentStatus) == 4


def test_pool_agent_status_is_str():
    assert isinstance(PoolAgentStatus.IDLE, str)


# ---------------------------------------------------------------------------
# PoolAgent dataclass
# ---------------------------------------------------------------------------

def test_pool_agent_auto_id():
    agent = PoolAgent(name="worker")
    assert agent.agent_id
    assert len(agent.agent_id) == 8


def test_pool_agent_unique_ids():
    a = PoolAgent(name="a")
    b = PoolAgent(name="b")
    assert a.agent_id != b.agent_id


def test_pool_agent_default_status():
    agent = PoolAgent(name="worker")
    assert agent.status == PoolAgentStatus.IDLE


def test_pool_agent_default_counters():
    agent = PoolAgent(name="worker")
    assert agent.tasks_completed == 0
    assert agent.tasks_failed == 0


def test_pool_agent_custom_fields():
    agent = PoolAgent(name="w", status=PoolAgentStatus.BUSY, tasks_completed=5)
    assert agent.status == PoolAgentStatus.BUSY
    assert agent.tasks_completed == 5


# ---------------------------------------------------------------------------
# AgentPool.spawn()
# ---------------------------------------------------------------------------

def test_spawn_returns_pool_agent():
    pool = AgentPool()
    agent = pool.spawn("worker")
    assert isinstance(agent, PoolAgent)


def test_spawn_agent_is_idle():
    pool = AgentPool()
    agent = pool.spawn("worker")
    assert agent.status == PoolAgentStatus.IDLE


def test_spawn_increments_pool_size():
    pool = AgentPool()
    pool.spawn("w1")
    pool.spawn("w2")
    assert pool.pool_size() == 2


def test_spawn_at_max_returns_none():
    pool = AgentPool(min_size=1, max_size=2)
    pool.spawn("w1")
    pool.spawn("w2")
    result = pool.spawn("w3")
    assert result is None


def test_spawn_below_max_returns_agent():
    pool = AgentPool(max_size=5)
    for i in range(5):
        assert pool.spawn(f"w{i}") is not None


# ---------------------------------------------------------------------------
# retire()
# ---------------------------------------------------------------------------

def test_retire_sets_status_retired():
    pool = AgentPool()
    agent = pool.spawn("worker")
    pool.retire(agent.agent_id)
    assert agent.status == PoolAgentStatus.RETIRED


def test_retire_returns_true():
    pool = AgentPool()
    agent = pool.spawn("worker")
    assert pool.retire(agent.agent_id) is True


def test_retire_unknown_returns_false():
    pool = AgentPool()
    assert pool.retire("no-such-id") is False


def test_retire_reduces_pool_size():
    pool = AgentPool(max_size=5)
    agent = pool.spawn("worker")
    size_before = pool.pool_size()
    pool.retire(agent.agent_id)
    assert pool.pool_size() == size_before - 1


# ---------------------------------------------------------------------------
# active_agents()
# ---------------------------------------------------------------------------

def test_active_agents_excludes_retired():
    pool = AgentPool(max_size=5)
    a1 = pool.spawn("w1")
    a2 = pool.spawn("w2")
    pool.retire(a1.agent_id)
    active = pool.active_agents()
    assert a1 not in active
    assert a2 in active


def test_active_agents_excludes_draining():
    pool = AgentPool(max_size=5)
    agent = pool.spawn("worker")
    # Manually set to draining
    agent.status = PoolAgentStatus.DRAINING
    active = pool.active_agents()
    assert agent not in active


def test_active_agents_includes_idle_and_busy():
    pool = AgentPool(max_size=5)
    a1 = pool.spawn("w1")
    a2 = pool.spawn("w2")
    pool.assign_task(a2.agent_id)
    active = pool.active_agents()
    assert a1 in active
    assert a2 in active


# ---------------------------------------------------------------------------
# assign_task()
# ---------------------------------------------------------------------------

def test_assign_task_sets_status_busy():
    pool = AgentPool()
    agent = pool.spawn("worker")
    pool.assign_task(agent.agent_id)
    assert agent.status == PoolAgentStatus.BUSY


def test_assign_task_returns_true_for_idle():
    pool = AgentPool()
    agent = pool.spawn("worker")
    assert pool.assign_task(agent.agent_id) is True


def test_assign_task_returns_false_for_non_idle():
    pool = AgentPool()
    agent = pool.spawn("worker")
    pool.assign_task(agent.agent_id)  # now BUSY
    assert pool.assign_task(agent.agent_id) is False


def test_assign_task_returns_false_for_unknown():
    pool = AgentPool()
    assert pool.assign_task("no-such-id") is False


def test_assign_task_returns_false_for_retired():
    pool = AgentPool(max_size=5)
    agent = pool.spawn("worker")
    pool.retire(agent.agent_id)
    assert pool.assign_task(agent.agent_id) is False


# ---------------------------------------------------------------------------
# complete_task()
# ---------------------------------------------------------------------------

def test_complete_task_sets_status_idle():
    pool = AgentPool()
    agent = pool.spawn("worker")
    pool.assign_task(agent.agent_id)
    pool.complete_task(agent.agent_id)
    assert agent.status == PoolAgentStatus.IDLE


def test_complete_task_increments_completed():
    pool = AgentPool()
    agent = pool.spawn("worker")
    pool.assign_task(agent.agent_id)
    pool.complete_task(agent.agent_id, success=True)
    assert agent.tasks_completed == 1


def test_complete_task_increments_failed():
    pool = AgentPool()
    agent = pool.spawn("worker")
    pool.assign_task(agent.agent_id)
    pool.complete_task(agent.agent_id, success=False)
    assert agent.tasks_failed == 1


def test_complete_task_returns_true():
    pool = AgentPool()
    agent = pool.spawn("worker")
    assert pool.complete_task(agent.agent_id) is True


def test_complete_task_unknown_returns_false():
    pool = AgentPool()
    assert pool.complete_task("no-such-id") is False


# ---------------------------------------------------------------------------
# scale_to()
# ---------------------------------------------------------------------------

def test_scale_to_increases_pool():
    pool = AgentPool(min_size=0, max_size=10)
    pool.scale_to(3)
    assert pool.pool_size() == 3


def test_scale_to_decreases_pool():
    pool = AgentPool(min_size=0, max_size=10)
    for i in range(6):
        pool.spawn(f"w{i}")
    pool.scale_to(3)
    assert pool.pool_size() == 3


def test_scale_to_respects_max():
    pool = AgentPool(min_size=0, max_size=5)
    result = pool.scale_to(20)
    assert result == 5


def test_scale_to_respects_min():
    pool = AgentPool(min_size=2, max_size=10)
    for i in range(5):
        pool.spawn(f"w{i}")
    result = pool.scale_to(0)
    assert result == 2


def test_scale_to_returns_actual_size():
    pool = AgentPool(min_size=1, max_size=10)
    size = pool.scale_to(4)
    assert size == pool.pool_size()


# ---------------------------------------------------------------------------
# pool_size()
# ---------------------------------------------------------------------------

def test_pool_size_zero_initially():
    pool = AgentPool()
    assert pool.pool_size() == 0


def test_pool_size_excludes_retired():
    pool = AgentPool(max_size=5)
    a1 = pool.spawn("w1")
    pool.spawn("w2")
    pool.retire(a1.agent_id)
    assert pool.pool_size() == 1


# ---------------------------------------------------------------------------
# utilization()
# ---------------------------------------------------------------------------

def test_utilization_zero_when_empty():
    pool = AgentPool()
    assert pool.utilization() == 0.0


def test_utilization_zero_when_all_idle():
    pool = AgentPool(max_size=5)
    pool.spawn("w1")
    pool.spawn("w2")
    assert pool.utilization() == 0.0


def test_utilization_one_when_all_busy():
    pool = AgentPool(max_size=5)
    a1 = pool.spawn("w1")
    a2 = pool.spawn("w2")
    pool.assign_task(a1.agent_id)
    pool.assign_task(a2.agent_id)
    assert pool.utilization() == 1.0


def test_utilization_fraction():
    pool = AgentPool(max_size=5)
    a1 = pool.spawn("w1")
    pool.spawn("w2")
    pool.assign_task(a1.agent_id)
    assert pool.utilization() == pytest.approx(0.5)


def test_utilization_excludes_retired():
    pool = AgentPool(max_size=5)
    a1 = pool.spawn("w1")
    a2 = pool.spawn("w2")
    pool.assign_task(a1.agent_id)
    pool.retire(a2.agent_id)
    # only a1 is active (BUSY), utilization = 1/1 = 1.0
    assert pool.utilization() == 1.0


# ---------------------------------------------------------------------------
# AGENT_POOL singleton
# ---------------------------------------------------------------------------

def test_agent_pool_singleton_exists():
    assert AGENT_POOL is not None
    assert isinstance(AGENT_POOL, AgentPool)
