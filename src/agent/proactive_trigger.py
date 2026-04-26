"""Proactive Agent Trigger — schedule/event-based agent activation.

Inspired by Kimi K2.6's 24/7 background-agent capability.  Implements a
trigger registry that evaluates firing conditions against a *virtual clock*
passed in at check time, making the module fully deterministic and
thread-free.

All firing logic is pure Python; no PyTorch, no threading, no external libs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TriggerSpec:
    """Specification for a single proactive trigger."""

    name: str
    trigger_fn: Callable[[float], bool]  # (current_time) -> should_fire
    action: str  # label / task name to submit when fired
    cooldown_s: float = 60.0  # minimum seconds between firings
    max_fires: int = -1  # -1 = unlimited
    enabled: bool = True
    last_fired_at: float | None = None
    fire_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ProactiveTriggerConfig:
    """Configuration for a ProactiveTriggerRegistry."""

    default_cooldown_s: float = 60.0
    max_triggers: int = 256


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ProactiveTriggerRegistry:
    """Registry that evaluates proactive triggers against a virtual clock.

    Usage::

        reg = ProactiveTriggerRegistry()
        spec = interval_trigger("heartbeat", interval_s=30, action="ping")
        reg.register(spec)

        fired_actions = reg.check_all(current_time=100.0)
    """

    def __init__(self, config: ProactiveTriggerConfig | None = None) -> None:
        self._config = config or ProactiveTriggerConfig()
        # Ordered dict preserves insertion order (Python 3.7+).
        self._triggers: dict[str, TriggerSpec] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def register(self, spec: TriggerSpec) -> None:
        """Register a trigger.

        Raises:
            ValueError: if *name* is already registered, or the registry is
                at capacity (``max_triggers``).
        """
        if spec.name in self._triggers:
            raise ValueError(
                f"Trigger {spec.name!r} is already registered. "
                "Unregister it first or use a different name."
            )
        if len(self._triggers) >= self._config.max_triggers:
            raise ValueError(f"Registry is at capacity ({self._config.max_triggers} triggers).")
        self._triggers[spec.name] = spec

    def unregister(self, name: str) -> bool:
        """Remove a trigger by name.  Returns True if it existed, False otherwise."""
        if name in self._triggers:
            del self._triggers[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable(self, name: str) -> bool:
        """Enable a trigger.  Returns True if trigger found, False otherwise."""
        if name in self._triggers:
            self._triggers[name].enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable a trigger.  Returns True if trigger found, False otherwise."""
        if name in self._triggers:
            self._triggers[name].enabled = False
            return True
        return False

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def check_all(self, current_time: float) -> list[str]:
        """Evaluate all triggers at *current_time*.

        A trigger fires when ALL of the following hold:

        * ``enabled`` is True
        * ``trigger_fn(current_time)`` returns True
        * cooldown has elapsed since ``last_fired_at`` (or never fired)
        * ``max_fires`` not reached (``-1`` means unlimited)

        For each fired trigger ``last_fired_at`` and ``fire_count`` are updated.

        Returns:
            List of ``action`` labels for triggers that fired (in registration
            order).
        """
        fired_actions: list[str] = []
        for spec in self._triggers.values():
            if not spec.enabled:
                continue

            # max_fires guard
            if spec.max_fires != -1 and spec.fire_count >= spec.max_fires:
                continue

            # cooldown guard
            if spec.last_fired_at is not None:
                elapsed = current_time - spec.last_fired_at
                if elapsed < spec.cooldown_s:
                    continue

            # user predicate
            try:
                should_fire = spec.trigger_fn(current_time)
            except Exception:
                # Never let a broken trigger_fn crash the whole loop.
                logging.getLogger(__name__).warning(
                    "Trigger %s predicate raised an exception; skipping.",
                    spec.name,
                    exc_info=True,
                )
                continue

            if not should_fire:
                continue

            # --- fire ---
            spec.last_fired_at = current_time
            spec.fire_count += 1
            fired_actions.append(spec.action)

        return fired_actions

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_trigger(self, name: str) -> TriggerSpec | None:
        """Return the TriggerSpec for *name*, or None if not registered."""
        return self._triggers.get(name)

    def list_triggers(self) -> list[str]:
        """Return names of all registered triggers (in insertion order)."""
        return list(self._triggers.keys())

    def reset(self, name: str) -> bool:
        """Reset fire_count and last_fired_at for *name*.

        Returns True if the trigger was found, False otherwise.
        """
        if name in self._triggers:
            spec = self._triggers[name]
            spec.fire_count = 0
            spec.last_fired_at = None
            return True
        return False


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def interval_trigger(
    name: str,
    interval_s: float,
    action: str,
    **kwargs: Any,
) -> TriggerSpec:
    """Create a trigger that fires every *interval_s* seconds.

    The first fire occurs at ``t=0`` (or whenever ``last_fired_at`` is None
    and the trigger is first evaluated).  Subsequent fires are spaced
    *interval_s* apart.

    All extra keyword arguments are forwarded to :class:`TriggerSpec`.
    """

    def _trigger_fn(current_time: float) -> bool:  # noqa: ANN001
        # Always returns True; the cooldown mechanism in check_all enforces
        # the interval.  On the very first call (last_fired_at is None) the
        # cooldown check passes automatically.
        return True

    # The interval is enforced via cooldown_s; the trigger_fn always returns
    # True so that the very first evaluation at t=0 fires immediately.
    kwargs.setdefault("cooldown_s", interval_s)
    return TriggerSpec(
        name=name,
        trigger_fn=_trigger_fn,
        action=action,
        **kwargs,
    )


def condition_trigger(
    name: str,
    condition_fn: Callable[[], bool],
    action: str,
    **kwargs: Any,
) -> TriggerSpec:
    """Create a trigger that fires whenever *condition_fn()* returns True.

    Subject to the normal ``cooldown_s`` / ``max_fires`` rules.
    All extra keyword arguments are forwarded to :class:`TriggerSpec`.
    """

    def _trigger_fn(current_time: float) -> bool:  # noqa: ANN001
        return condition_fn()

    return TriggerSpec(
        name=name,
        trigger_fn=_trigger_fn,
        action=action,
        **kwargs,
    )
