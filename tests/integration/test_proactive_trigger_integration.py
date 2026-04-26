"""Integration tests for ProactiveTriggerRegistry wired into AGENT_LOOP_REGISTRY.

Scenario: 3 triggers (2 interval, 1 condition) across a simulated time
progression.  Verifies expected fire counts and that the registry is
properly wired into AGENT_LOOP_REGISTRY["proactive_trigger"].
"""

from __future__ import annotations

from src.agent import AGENT_LOOP_REGISTRY
from src.agent.proactive_trigger import (
    ProactiveTriggerRegistry,
    condition_trigger,
    interval_trigger,
)

# ---------------------------------------------------------------------------
# 1. Registry wired into AGENT_LOOP_REGISTRY
# ---------------------------------------------------------------------------


def test_proactive_trigger_in_agent_loop_registry():
    assert "proactive_trigger" in AGENT_LOOP_REGISTRY, (
        "'proactive_trigger' must be registered in AGENT_LOOP_REGISTRY"
    )


# ---------------------------------------------------------------------------
# 2. Class from registry is constructable
# ---------------------------------------------------------------------------


def test_registry_class_constructable():
    cls = AGENT_LOOP_REGISTRY["proactive_trigger"]
    reg = cls()
    assert isinstance(reg, ProactiveTriggerRegistry)


# ---------------------------------------------------------------------------
# 3. Full time-progression scenario
#
#   Triggers
#   --------
#   interval_a  — interval=60s, action="task_a"
#   interval_b  — interval=60s, action="task_b"
#   cond_c      — condition (initially True), action="task_c", cooldown=60s
#
#   Timeline
#   --------
#   t=0   → interval_a fires, interval_b fires, cond_c fires  (3 actions)
#   t=30  → cooldown not elapsed for any → 0 actions
#   t=65  → elapsed ≥ 60 for all three → 3 actions again
#
# ---------------------------------------------------------------------------


def test_time_progression_scenario():
    reg = ProactiveTriggerRegistry()

    reg.register(interval_trigger("interval_a", interval_s=60.0, action="task_a"))
    reg.register(interval_trigger("interval_b", interval_s=60.0, action="task_b"))

    condition_active = [True]
    reg.register(
        condition_trigger(
            "cond_c",
            condition_fn=lambda: condition_active[0],
            action="task_c",
            cooldown_s=60.0,
        )
    )

    # t=0 — all three should fire (first evaluation; no cooldown history).
    fired_t0 = reg.check_all(current_time=0.0)
    assert set(fired_t0) == {"task_a", "task_b", "task_c"}, (
        f"Expected all 3 to fire at t=0, got {fired_t0}"
    )

    # t=30 — cooldown not elapsed → nothing fires.
    fired_t30 = reg.check_all(current_time=30.0)
    assert fired_t30 == [], f"Expected 0 fires at t=30 (cooldown), got {fired_t30}"

    # t=65 — 65 ≥ 60 → all three fire again.
    fired_t65 = reg.check_all(current_time=65.0)
    assert set(fired_t65) == {"task_a", "task_b", "task_c"}, (
        f"Expected all 3 to fire at t=65, got {fired_t65}"
    )

    # fire_count verification
    assert reg.get_trigger("interval_a").fire_count == 2
    assert reg.get_trigger("interval_b").fire_count == 2
    assert reg.get_trigger("cond_c").fire_count == 2


# ---------------------------------------------------------------------------
# 4. Regression guard — existing AGENT_LOOP_REGISTRY keys intact
# ---------------------------------------------------------------------------


def test_existing_agent_loop_registry_keys_intact():
    expected = {
        "react",
        "safe_dispatch",
        "beam_plan",
        "task_decompose",
        "dispatch_task",
        "budget_bounded",
        "agent_swarm",
        "plugin_hook",
    }
    for key in expected:
        assert key in AGENT_LOOP_REGISTRY, (
            f"Regression: existing key {key!r} missing from AGENT_LOOP_REGISTRY"
        )
