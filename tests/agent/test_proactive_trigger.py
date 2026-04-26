"""Unit tests for src/agent/proactive_trigger.py — ProactiveTriggerRegistry.

10–16 tests covering: config defaults, register/unregister, enable/disable,
check_all firing logic, cooldown, max_fires, fire_count, reset, and both
factory helpers.
"""

from __future__ import annotations

import pytest

from src.agent.proactive_trigger import (
    ProactiveTriggerConfig,
    ProactiveTriggerRegistry,
    TriggerSpec,
    condition_trigger,
    interval_trigger,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _always_true(t: float) -> bool:
    return True


def _always_false(t: float) -> bool:
    return False


def _fresh(cooldown: float = 0.0) -> ProactiveTriggerRegistry:
    """Return a fresh registry; default cooldown 0 so tests don't fight it."""
    cfg = ProactiveTriggerConfig(default_cooldown_s=cooldown)
    return ProactiveTriggerRegistry(config=cfg)


def _spec(
    name: str = "t",
    trigger_fn=_always_true,
    action: str = "do_thing",
    cooldown_s: float = 0.0,
    **kwargs,
) -> TriggerSpec:
    return TriggerSpec(
        name=name,
        trigger_fn=trigger_fn,
        action=action,
        cooldown_s=cooldown_s,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = ProactiveTriggerConfig()
    assert cfg.default_cooldown_s == 60.0
    assert cfg.max_triggers == 256


# ---------------------------------------------------------------------------
# 2. test_register_basic
# ---------------------------------------------------------------------------


def test_register_basic():
    reg = _fresh()
    reg.register(_spec("alpha"))
    assert "alpha" in reg.list_triggers()


# ---------------------------------------------------------------------------
# 3. test_register_duplicate_raises
# ---------------------------------------------------------------------------


def test_register_duplicate_raises():
    reg = _fresh()
    reg.register(_spec("alpha"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_spec("alpha"))


# ---------------------------------------------------------------------------
# 4. test_unregister
# ---------------------------------------------------------------------------


def test_unregister():
    reg = _fresh()
    reg.register(_spec("alpha"))
    result = reg.unregister("alpha")
    assert result is True
    assert "alpha" not in reg.list_triggers()


# ---------------------------------------------------------------------------
# 5. test_unregister_nonexistent
# ---------------------------------------------------------------------------


def test_unregister_nonexistent():
    reg = _fresh()
    assert reg.unregister("ghost") is False


# ---------------------------------------------------------------------------
# 6. test_enable_disable
# ---------------------------------------------------------------------------


def test_enable_disable():
    reg = _fresh()
    reg.register(_spec("alpha"))

    # Disable → should not fire.
    reg.disable("alpha")
    fired = reg.check_all(current_time=0.0)
    assert fired == []

    # Re-enable → should fire.
    reg.enable("alpha")
    fired = reg.check_all(current_time=1.0)
    assert fired == ["do_thing"]


# ---------------------------------------------------------------------------
# 7. test_check_all_fires
# ---------------------------------------------------------------------------


def test_check_all_fires():
    reg = _fresh()
    reg.register(_spec("alpha", cooldown_s=0.0))
    fired = reg.check_all(current_time=0.0)
    assert fired == ["do_thing"]


# ---------------------------------------------------------------------------
# 8. test_check_all_cooldown
# ---------------------------------------------------------------------------


def test_check_all_cooldown():
    reg = _fresh()
    reg.register(_spec("alpha", cooldown_s=60.0))
    # First call fires.
    reg.check_all(current_time=0.0)
    # Second call at t=30 — cooldown not elapsed.
    fired = reg.check_all(current_time=30.0)
    assert fired == []


# ---------------------------------------------------------------------------
# 9. test_check_all_cooldown_elapsed
# ---------------------------------------------------------------------------


def test_check_all_cooldown_elapsed():
    reg = _fresh()
    reg.register(_spec("alpha", cooldown_s=60.0))
    reg.check_all(current_time=0.0)  # fires; last_fired_at=0
    reg.check_all(current_time=30.0)  # not yet
    fired = reg.check_all(current_time=65.0)  # elapsed ≥ 60 → fires again
    assert fired == ["do_thing"]


# ---------------------------------------------------------------------------
# 10. test_max_fires_limit
# ---------------------------------------------------------------------------


def test_max_fires_limit():
    reg = _fresh()
    reg.register(_spec("alpha", cooldown_s=0.0, max_fires=2))

    reg.check_all(current_time=0.0)  # fire 1
    reg.check_all(current_time=1.0)  # fire 2
    fired = reg.check_all(current_time=2.0)  # should NOT fire
    assert fired == []
    assert reg.get_trigger("alpha").fire_count == 2


# ---------------------------------------------------------------------------
# 11. test_max_fires_unlimited
# ---------------------------------------------------------------------------


def test_max_fires_unlimited():
    reg = _fresh()
    reg.register(_spec("alpha", cooldown_s=0.0, max_fires=-1))

    for i in range(20):
        fired = reg.check_all(current_time=float(i))
        assert "do_thing" in fired

    assert reg.get_trigger("alpha").fire_count == 20


# ---------------------------------------------------------------------------
# 12. test_disabled_does_not_fire
# ---------------------------------------------------------------------------


def test_disabled_does_not_fire():
    reg = _fresh()
    reg.register(_spec("alpha", cooldown_s=0.0, enabled=False))
    for t in range(5):
        assert reg.check_all(current_time=float(t)) == []


# ---------------------------------------------------------------------------
# 13. test_fire_count_increments
# ---------------------------------------------------------------------------


def test_fire_count_increments():
    reg = _fresh()
    reg.register(_spec("alpha", cooldown_s=0.0))

    for i in range(1, 6):
        reg.check_all(current_time=float(i))
        assert reg.get_trigger("alpha").fire_count == i


# ---------------------------------------------------------------------------
# 14. test_reset
# ---------------------------------------------------------------------------


def test_reset():
    reg = _fresh()
    reg.register(_spec("alpha", cooldown_s=0.0))

    reg.check_all(current_time=0.0)
    reg.check_all(current_time=1.0)
    assert reg.get_trigger("alpha").fire_count == 2

    result = reg.reset("alpha")
    assert result is True
    spec = reg.get_trigger("alpha")
    assert spec.fire_count == 0
    assert spec.last_fired_at is None


# ---------------------------------------------------------------------------
# 15. test_interval_trigger_factory
# ---------------------------------------------------------------------------


def test_interval_trigger_factory():
    reg = _fresh()
    spec = interval_trigger("heartbeat", interval_s=10.0, action="ping")
    reg.register(spec)

    # t=0 → first fire
    fired = reg.check_all(current_time=0.0)
    assert fired == ["ping"]

    # t=5 → cooldown not elapsed (interval=10)
    fired = reg.check_all(current_time=5.0)
    assert fired == []

    # t=10 → interval elapsed → fires again
    fired = reg.check_all(current_time=10.0)
    assert fired == ["ping"]


# ---------------------------------------------------------------------------
# 16. test_condition_trigger_factory
# ---------------------------------------------------------------------------


def test_condition_trigger_factory():
    flag = [False]

    def cond() -> bool:
        return flag[0]

    reg = _fresh()
    spec = condition_trigger("cond_t", condition_fn=cond, action="react", cooldown_s=0.0)
    reg.register(spec)

    # Condition False → no fire.
    assert reg.check_all(current_time=0.0) == []

    # Condition True → fires.
    flag[0] = True
    fired = reg.check_all(current_time=1.0)
    assert fired == ["react"]

    # Back to False → no fire.
    flag[0] = False
    assert reg.check_all(current_time=2.0) == []
