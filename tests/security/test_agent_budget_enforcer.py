"""Tests for agent_budget_enforcer — hard limits on agent execution.

Security surface: STRIDE Denial of Service.
"""
from __future__ import annotations

import time

import pytest

from src.security.agent_budget_enforcer import (
    AgentBudget,
    AgentBudgetEnforcer,
    BudgetExhausted,
    BUDGET_ENFORCER_REGISTRY,
    DEFAULT_BUDGET_ENFORCER,
)


# ---------------------------------------------------------------------------
# Step budget
# ---------------------------------------------------------------------------


def test_step_budget_not_exhausted():
    e = AgentBudgetEnforcer(AgentBudget(max_steps=5))
    snap = e.check(additional_steps=3)
    assert snap.steps_used == 3
    assert snap.exhausted is False


def test_step_budget_exhausted_raises():
    e = AgentBudgetEnforcer(AgentBudget(max_steps=2))
    e.check(additional_steps=2)
    with pytest.raises(BudgetExhausted, match="step_budget_exhausted"):
        e.check(additional_steps=1)


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


def test_token_budget_exhausted_raises():
    e = AgentBudgetEnforcer(AgentBudget(max_tokens=100))
    e.check(additional_tokens=100)
    with pytest.raises(BudgetExhausted, match="token_budget_exhausted"):
        e.check(additional_tokens=1)


# ---------------------------------------------------------------------------
# Time budget
# ---------------------------------------------------------------------------


def test_time_budget_exhausted_raises():
    e = AgentBudgetEnforcer(AgentBudget(max_elapsed_seconds=0.05))
    time.sleep(0.1)
    with pytest.raises(BudgetExhausted, match="time_budget_exhausted"):
        e.check()


# ---------------------------------------------------------------------------
# No budget = unlimited
# ---------------------------------------------------------------------------


def test_no_budget_never_exhausted():
    e = AgentBudgetEnforcer(AgentBudget())
    for _ in range(1000):
        e.check(additional_steps=1)
    assert e.snapshot().steps_used == 1000


# ---------------------------------------------------------------------------
# Snapshot without side effects
# ---------------------------------------------------------------------------


def test_snapshot_does_not_increment():
    e = AgentBudgetEnforcer(AgentBudget(max_steps=10))
    e.check(additional_steps=3)
    snap1 = e.snapshot()
    snap2 = e.snapshot()
    assert snap1.steps_used == snap2.steps_used == 3


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_exhausted_state():
    e = AgentBudgetEnforcer(AgentBudget(max_steps=1))
    with pytest.raises(BudgetExhausted):
        e.check(additional_steps=2)
    with pytest.raises(BudgetExhausted):
        e.check()
    e.reset()
    e.check(additional_steps=1)  # should not raise


# ---------------------------------------------------------------------------
# Re-check after exhaustion
# ---------------------------------------------------------------------------


def test_recheck_raises_immediately():
    e = AgentBudgetEnforcer(AgentBudget(max_steps=1))
    with pytest.raises(BudgetExhausted):
        e.check(additional_steps=5)
    with pytest.raises(BudgetExhausted):
        e.check()  # should raise even with 0 increment


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in BUDGET_ENFORCER_REGISTRY
    assert isinstance(BUDGET_ENFORCER_REGISTRY["default"], AgentBudgetEnforcer)


def test_default_is_budget_enforcer():
    assert isinstance(DEFAULT_BUDGET_ENFORCER, AgentBudgetEnforcer)
