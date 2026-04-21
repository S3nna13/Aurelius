"""Tiered compaction trigger manager (Mythos System Card, pp.189, 192).

Decides *when* to invoke a compaction pass across a long-running agent
context. The actual compaction algorithm lives in
``src.longcontext.context_compaction`` -- this module is only the trigger
policy: multiple tiers with distinct token thresholds and target ratios
fire at their own cadence, and a total budget caps the lifetime context
size via repeated compaction passes.

Mythos defaults:

* ``fast``  tier: threshold 50_000 tokens, target ratio 0.3
  (HLE-like tasks -- frequent compaction).
* ``slow``  tier: threshold 200_000 tokens, target ratio 0.5
  (BrowseComp-like tasks -- less frequent compaction).
* ``total_budget``: 3_000_000 tokens lifetime ceiling.

Pure stdlib: only ``dataclasses`` and ``time``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class CompactionTier:
    """A single compaction tier.

    ``threshold_tokens`` is both the *first-fire* threshold and the
    re-trigger cadence for this tier (next fire at
    ``last_fired + threshold_tokens``).

    ``ratio`` is the target compaction ratio handed to the compaction
    function (e.g., 0.3 = compress to 30% of input).
    """

    name: str
    threshold_tokens: int
    ratio: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("CompactionTier.name must be a non-empty str")
        if not isinstance(self.threshold_tokens, int) or self.threshold_tokens <= 0:
            raise ValueError("CompactionTier.threshold_tokens must be a positive int")
        if not isinstance(self.ratio, (int, float)):
            raise TypeError("CompactionTier.ratio must be numeric")
        if not (0.0 <= float(self.ratio) <= 1.0):
            raise ValueError("CompactionTier.ratio must be in [0, 1]")


@dataclass
class CompactionEvent:
    """Record of one compaction firing."""

    tier_name: str
    at_tokens: int
    before_tokens: int
    after_tokens: int
    elapsed_s: float
    note: str = ""


DEFAULT_TIERS: tuple[CompactionTier, ...] = (
    CompactionTier("fast", 50_000, 0.3),
    CompactionTier("slow", 200_000, 0.5),
)


class CompactionTriggerManager:
    """Tiered compaction trigger manager.

    Iterates tiers in ascending threshold order on each ``observe`` call
    and fires the *largest* tier whose threshold is <= current tokens and
    whose per-tier next-fire mark has also been reached. Exactly one tier
    fires per observe call (the one selected); other tiers' schedules are
    untouched.
    """

    def __init__(
        self,
        tiers: tuple[CompactionTier, ...],
        compaction_fn: Callable[[list[dict], float], list[dict]],
        total_budget: int = 3_000_000,
    ) -> None:
        if not isinstance(tiers, tuple) or not tiers:
            raise ValueError("tiers must be a non-empty tuple of CompactionTier")
        for t in tiers:
            if not isinstance(t, CompactionTier):
                raise TypeError("tiers must contain CompactionTier instances")
        if not callable(compaction_fn):
            raise TypeError("compaction_fn must be callable")
        if not isinstance(total_budget, int) or total_budget <= 0:
            raise ValueError("total_budget must be a positive int")

        # Sort ascending by threshold; enforce uniqueness of names.
        names = [t.name for t in tiers]
        if len(set(names)) != len(names):
            raise ValueError("tier names must be unique")

        self._tiers: tuple[CompactionTier, ...] = tuple(
            sorted(tiers, key=lambda t: t.threshold_tokens)
        )
        self._compaction_fn = compaction_fn
        self._total_budget = int(total_budget)
        self._last_fired: dict[str, int] = {t.name: 0 for t in self._tiers}
        self._fire_counts: dict[str, int] = {t.name: 0 for t in self._tiers}
        self._events: list[CompactionEvent] = []

    # ---- public API -----------------------------------------------------

    @property
    def tiers(self) -> tuple[CompactionTier, ...]:
        return self._tiers

    @property
    def total_budget(self) -> int:
        return self._total_budget

    def observe(
        self,
        current_tokens: int,
        message_history: list[dict],
    ) -> CompactionEvent | None:
        """Observe current context size; maybe fire a compaction.

        Parameters
        ----------
        current_tokens:
            Total token count of the live context.
        message_history:
            Sequence of message dicts passed through to ``compaction_fn``.

        Returns
        -------
        CompactionEvent if a tier fired, else None.
        """
        if not isinstance(current_tokens, int) or current_tokens < 0:
            raise ValueError("current_tokens must be a non-negative int")
        if not isinstance(message_history, list):
            raise TypeError("message_history must be a list of dicts")
        if current_tokens > self._total_budget:
            raise RuntimeError(
                f"CompactionTriggerManager: total_budget exceeded "
                f"({current_tokens} > {self._total_budget})"
            )

        # Pick the largest tier eligible to fire.
        chosen: CompactionTier | None = None
        for tier in self._tiers:  # ascending order
            if current_tokens < tier.threshold_tokens:
                continue
            next_fire_at = self._last_fired[tier.name] + tier.threshold_tokens
            if current_tokens >= next_fire_at:
                chosen = tier  # keep climbing; prefer slower (larger) tier

        if chosen is None:
            return None

        before_tokens = current_tokens
        t0 = time.perf_counter()
        try:
            new_history = self._compaction_fn(message_history, chosen.ratio)
        except Exception as exc:  # noqa: BLE001 -- bubble with context
            raise RuntimeError(
                f"compaction_fn raised while firing tier={chosen.name!r} "
                f"at_tokens={current_tokens}: {type(exc).__name__}: {exc}"
            ) from exc
        elapsed = time.perf_counter() - t0

        if not isinstance(new_history, list):
            raise TypeError(
                f"compaction_fn must return list[dict]; got {type(new_history).__name__}"
            )

        after_tokens = int(round(before_tokens * float(chosen.ratio)))
        self._last_fired[chosen.name] = current_tokens
        self._fire_counts[chosen.name] += 1

        event = CompactionEvent(
            tier_name=chosen.name,
            at_tokens=current_tokens,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            elapsed_s=elapsed,
            note=f"tier={chosen.name} ratio={chosen.ratio}",
        )
        self._events.append(event)
        return event

    def reset(self) -> None:
        """Clear per-tier state and event log."""
        self._last_fired = {t.name: 0 for t in self._tiers}
        self._fire_counts = {t.name: 0 for t in self._tiers}
        self._events = []

    def metrics(self) -> dict[str, Any]:
        """Return a snapshot of trigger metrics (pure dict; safe to log)."""
        return {
            "total_budget": self._total_budget,
            "tiers": [
                {
                    "name": t.name,
                    "threshold_tokens": t.threshold_tokens,
                    "ratio": t.ratio,
                }
                for t in self._tiers
            ],
            "fire_counts": dict(self._fire_counts),
            "last_fired": dict(self._last_fired),
            "total_fires": sum(self._fire_counts.values()),
        }
