"""Multi-resource budget accounting ledger for agent loops.

This module provides :class:`BudgetLedger`, a transactional accounting
primitive that tracks independent spend against several resources
simultaneously (tokens, wall-clock, cost, tool-calls, errors, ...).  It is
the counterpart of :mod:`src.agent.budget_bounded_loop`, which only tracks
a single scalar cap.  Production agents need soft (advisory) and hard
(fatal) limits on each resource; exceeding a hard limit is transactional
— the spend that would push the ledger over is rejected without mutating
the ledger's state.

Design notes
------------
* Pure stdlib; no foreign imports.
* Deterministic: the monotonic clock is injectable via ``clock_ns`` so tests
  can fast-forward time without sleeping.
* The ``wall_ms`` resource is auto-inserted if the caller does not supply
  one (default: ``math.inf`` hard limit, so it never trips).  Step updates
  against ``wall_ms`` are silent — they flip the severity to HARD when
  elapsed time exceeds the configured hard limit but never raise.  Callers
  inspect :meth:`BudgetLedger.is_exhausted` / :meth:`worst_severity` after
  :meth:`step` to react.
* Severity thresholds are strict-greater-than:
    * OK    when ``spent <= soft_fraction * hard_limit``
    * SOFT  when ``soft_fraction * hard_limit < spent <= hard_limit``
    * HARD  when ``spent > hard_limit``
  :meth:`charge` raises :class:`BudgetExhaustedError` before any state
  mutation whenever the requested charge would push the spend strictly
  above ``hard_limit``.
* Single-owner ledger; external locking is required for multi-threaded use.

References
----------
Inspired by budget-bounded acting (Yao et al., ReAct, arXiv:2210.03629) and
token-accounting patterns in production LLM orchestrators.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BudgetExhaustedError(Exception):
    """Raised when a ledger transaction would exceed a hard limit.

    Raised *before* any state mutation so callers can retry after
    choosing a smaller amount or freeing a different resource.
    """


class LedgerError(Exception):
    """Raised on bad ledger API usage (e.g. negative spend, unknown resource)."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class BudgetSeverity(str, Enum):
    """Severity of a resource's spend relative to its configured thresholds."""

    OK = "ok"
    SOFT = "soft"
    HARD = "hard"


_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


@dataclass(frozen=True)
class ResourceLimit:
    """Immutable specification of a single budget resource."""

    name: str
    hard_limit: float
    soft_fraction: float = 0.8

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise LedgerError("ResourceLimit.name must be a non-empty string")
        if not math.isfinite(self.soft_fraction):
            raise LedgerError("ResourceLimit.soft_fraction must be finite")
        if self.soft_fraction <= 0.0 or self.soft_fraction > 1.0:
            raise LedgerError(
                "ResourceLimit.soft_fraction must be in the half-open "
                "interval (0, 1]"
            )
        if isinstance(self.hard_limit, bool):
            raise LedgerError("ResourceLimit.hard_limit must be a real number")
        if not isinstance(self.hard_limit, (int, float)):
            raise LedgerError("ResourceLimit.hard_limit must be numeric")
        if math.isnan(self.hard_limit):
            raise LedgerError("ResourceLimit.hard_limit must not be NaN")
        if self.hard_limit <= 0.0:
            raise LedgerError("ResourceLimit.hard_limit must be positive")


@dataclass(frozen=True)
class LedgerSnapshot:
    """Point-in-time view of a :class:`BudgetLedger`."""

    resources: dict[str, float]
    limits: dict[str, float]
    severities: dict[str, BudgetSeverity]
    remaining: dict[str, float]
    step_count: int
    started_ns: int
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


_WALL_MS = "wall_ms"


class BudgetLedger:
    """Transactional multi-resource spend accumulator.

    Parameters
    ----------
    limits:
        Non-empty tuple of :class:`ResourceLimit` entries.  Duplicate names
        are rejected.  A ``wall_ms`` limit is auto-inserted if the caller
        did not supply one; the auto-inserted limit has ``hard_limit =
        math.inf`` so it never trips.
    clock_ns:
        Optional monotonic clock returning nanoseconds.  Injectable for
        deterministic tests.  Defaults to :func:`time.monotonic_ns`.

    Notes
    -----
    This ledger is a *single-owner* structure.  It performs no locking and
    callers sharing it across threads must wrap access externally.
    """

    def __init__(
        self,
        limits: tuple[ResourceLimit, ...],
        *,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(limits, tuple):
            raise LedgerError("limits must be a tuple of ResourceLimit entries")
        if len(limits) == 0:
            raise LedgerError("limits must be non-empty")
        for entry in limits:
            if not isinstance(entry, ResourceLimit):
                raise LedgerError(
                    "limits entries must be ResourceLimit instances"
                )

        names = [lim.name for lim in limits]
        if len(set(names)) != len(names):
            raise LedgerError("duplicate resource names are not allowed")

        if clock_ns is None:
            self._clock_ns: Callable[[], int] = time.monotonic_ns
        else:
            if not callable(clock_ns):
                raise LedgerError("clock_ns must be callable")
            self._clock_ns = clock_ns

        lim_list = list(limits)
        if _WALL_MS not in names:
            lim_list.append(
                ResourceLimit(
                    name=_WALL_MS,
                    hard_limit=math.inf,
                    soft_fraction=1.0,
                )
            )

        self._limits: dict[str, ResourceLimit] = {
            lim.name: lim for lim in lim_list
        }
        self._spent: dict[str, float] = {name: 0.0 for name in self._limits}
        self._step_count: int = 0
        self._started_ns: int = int(self._clock_ns())

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Number of :meth:`step` calls since construction / last ``reset``."""
        return self._step_count

    @property
    def started_ns(self) -> int:
        """Monotonic clock reading captured at construction / last ``reset``."""
        return self._started_ns

    def resource_names(self) -> tuple[str, ...]:
        """Ordered tuple of configured resource names."""
        return tuple(self._limits.keys())

    def spent(self, resource: str) -> float:
        """Amount spent against *resource* (zero for fresh ledgers)."""
        self._require_known(resource)
        return self._spent[resource]

    def remaining(self, resource: str) -> float:
        """Return ``hard_limit - spent`` for *resource*.

        Always returns a non-negative value; if spent has been driven past
        the hard limit (only possible through silent ``wall_ms`` updates)
        the result is clamped to 0.
        """
        self._require_known(resource)
        rem = self._limits[resource].hard_limit - self._spent[resource]
        if rem < 0.0:
            return 0.0
        return rem

    def severity(self, resource: str) -> BudgetSeverity:
        """Return the current severity of *resource*."""
        self._require_known(resource)
        return self._severity_of(resource, self._spent[resource])

    def worst_severity(self) -> BudgetSeverity:
        """Severity across all resources (HARD > SOFT > OK)."""
        worst = BudgetSeverity.OK
        for name in self._limits:
            sev = self._severity_of(name, self._spent[name])
            if sev == BudgetSeverity.HARD:
                return BudgetSeverity.HARD
            if sev == BudgetSeverity.SOFT and worst == BudgetSeverity.OK:
                worst = BudgetSeverity.SOFT
        return worst

    def is_exhausted(self) -> bool:
        """True iff any resource is currently at HARD severity."""
        for name in self._limits:
            if self._severity_of(name, self._spent[name]) == BudgetSeverity.HARD:
                return True
        return False

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def charge(self, resource: str, amount: float) -> BudgetSeverity:
        """Record *amount* spent against *resource*.

        Returns the resulting severity (OK or SOFT).  Raises
        :class:`BudgetExhaustedError` without mutating state if the charge
        would push spend strictly above ``hard_limit``.  Raises
        :class:`LedgerError` for negative amounts, non-numeric amounts,
        NaN amounts, or unknown resources.
        """
        self._require_known(resource)
        if isinstance(amount, bool):
            raise LedgerError("charge amount must be a real number, not bool")
        if not isinstance(amount, (int, float)):
            raise LedgerError("charge amount must be numeric")
        amt = float(amount)
        if math.isnan(amt):
            raise LedgerError("charge amount must not be NaN")
        if amt < 0.0:
            raise LedgerError(
                f"charge amount must be non-negative (got {amt!r})"
            )

        limit = self._limits[resource]
        prospective = self._spent[resource] + amt
        if prospective > limit.hard_limit:
            remaining = max(limit.hard_limit - self._spent[resource], 0.0)
            raise BudgetExhaustedError(
                f"resource={resource!r} "
                f"requested={amt} "
                f"remaining={remaining} "
                f"hard_limit={limit.hard_limit}"
            )

        self._spent[resource] = prospective
        return self._severity_of(resource, prospective)

    def step(self) -> None:
        """Increment step counter and update ``wall_ms`` silently.

        ``wall_ms`` updates never raise — even when the configured
        ``wall_ms`` hard limit is finite and elapsed time has crossed it.
        Callers should check :meth:`is_exhausted` / :meth:`worst_severity`
        after stepping to react to a silent wall-clock exhaustion.
        """
        self._step_count += 1
        self._update_wall_ms()

    def reset(self) -> None:
        """Zero all counters and reset ``started_ns`` to ``clock_ns()``."""
        for name in self._spent:
            self._spent[name] = 0.0
        self._step_count = 0
        self._started_ns = int(self._clock_ns())

    def snapshot(self) -> LedgerSnapshot:
        """Return a point-in-time :class:`LedgerSnapshot`."""
        self._update_wall_ms()
        resources: dict[str, float] = {}
        limits: dict[str, float] = {}
        severities: dict[str, BudgetSeverity] = {}
        remaining: dict[str, float] = {}
        for name, lim in self._limits.items():
            spent = self._spent[name]
            resources[name] = spent
            limits[name] = lim.hard_limit
            severities[name] = self._severity_of(name, spent)
            rem = lim.hard_limit - spent
            remaining[name] = rem if rem > 0.0 else 0.0
        elapsed_ms = (int(self._clock_ns()) - self._started_ns) / 1e6
        if elapsed_ms < 0.0:
            elapsed_ms = 0.0
        return LedgerSnapshot(
            resources=resources,
            limits=limits,
            severities=severities,
            remaining=remaining,
            step_count=self._step_count,
            started_ns=self._started_ns,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "BudgetLedger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self._update_wall_ms()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_known(self, resource: str) -> None:
        if not isinstance(resource, str):
            raise LedgerError("resource must be a string")
        if resource not in self._limits:
            raise LedgerError(f"unknown resource {resource!r}")

    def _severity_of(self, resource: str, spent: float) -> BudgetSeverity:
        lim = self._limits[resource]
        hard = lim.hard_limit
        soft_threshold = lim.soft_fraction * hard
        if spent > hard:
            return BudgetSeverity.HARD
        if spent > soft_threshold:
            return BudgetSeverity.SOFT
        return BudgetSeverity.OK

    def _update_wall_ms(self) -> None:
        """Silently refresh the ``wall_ms`` spend.

        Never raises.  Clamps spend to ``hard_limit`` when exceeded so the
        severity helper can report HARD.  Updates are always absolute —
        elapsed time since ``started_ns`` — so repeated calls are
        idempotent on the resulting spend value.
        """
        if _WALL_MS not in self._limits:
            return
        elapsed_ms = (int(self._clock_ns()) - self._started_ns) / 1e6
        if elapsed_ms < 0.0:
            elapsed_ms = 0.0
        limit = self._limits[_WALL_MS]
        if math.isfinite(limit.hard_limit) and elapsed_ms > limit.hard_limit:
            self._spent[_WALL_MS] = elapsed_ms
        else:
            self._spent[_WALL_MS] = elapsed_ms


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


#: Module-level registry of named ledgers for long-running agents.
BUDGET_LEDGER_REGISTRY: dict[str, BudgetLedger] = {}


def _validate_registry_name(name: str) -> None:
    if not isinstance(name, str):
        raise LedgerError("ledger name must be a string")
    if not name:
        raise LedgerError("ledger name must be non-empty")
    if not _VALID_NAME_RE.match(name):
        raise LedgerError(
            f"ledger name {name!r} contains invalid characters "
            "(allowed: A-Z a-z 0-9 _ -)"
        )


def register_ledger(name: str, ledger: BudgetLedger) -> None:
    """Register *ledger* under *name* in :data:`BUDGET_LEDGER_REGISTRY`.

    Raises :class:`LedgerError` if the name is invalid or already taken,
    or if *ledger* is not a :class:`BudgetLedger` instance.
    """
    _validate_registry_name(name)
    if not isinstance(ledger, BudgetLedger):
        raise LedgerError("ledger must be a BudgetLedger instance")
    if name in BUDGET_LEDGER_REGISTRY:
        raise LedgerError(f"ledger {name!r} already registered")
    BUDGET_LEDGER_REGISTRY[name] = ledger


def get_ledger(name: str) -> BudgetLedger:
    """Return the ledger registered under *name* or raise :class:`LedgerError`."""
    _validate_registry_name(name)
    if name not in BUDGET_LEDGER_REGISTRY:
        raise LedgerError(f"no ledger registered under {name!r}")
    return BUDGET_LEDGER_REGISTRY[name]


def list_ledgers() -> tuple[str, ...]:
    """Return a tuple of currently-registered ledger names."""
    return tuple(BUDGET_LEDGER_REGISTRY.keys())


def make_default_ledger(
    *,
    token_budget: int = 128_000,
    wall_ms_budget: float = 120_000,
    cost_usd_budget: float = 1.0,
    tool_call_budget: int = 32,
    error_budget: int = 8,
    clock_ns: Callable[[], int] | None = None,
) -> BudgetLedger:
    """Return a :class:`BudgetLedger` with sensible production defaults.

    The returned ledger tracks the canonical five resources expected of an
    agent loop: ``tokens``, ``wall_ms``, ``cost_usd``, ``tool_calls``,
    ``errors``.
    """
    for label, value in (
        ("token_budget", token_budget),
        ("wall_ms_budget", wall_ms_budget),
        ("cost_usd_budget", cost_usd_budget),
        ("tool_call_budget", tool_call_budget),
        ("error_budget", error_budget),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LedgerError(f"{label} must be numeric")
        if value <= 0:
            raise LedgerError(f"{label} must be positive")

    limits = (
        ResourceLimit(name="tokens", hard_limit=float(token_budget)),
        ResourceLimit(name=_WALL_MS, hard_limit=float(wall_ms_budget)),
        ResourceLimit(name="cost_usd", hard_limit=float(cost_usd_budget)),
        ResourceLimit(name="tool_calls", hard_limit=float(tool_call_budget)),
        ResourceLimit(name="errors", hard_limit=float(error_budget)),
    )
    return BudgetLedger(limits, clock_ns=clock_ns)


__all__ = [
    "BUDGET_LEDGER_REGISTRY",
    "BudgetExhaustedError",
    "BudgetLedger",
    "BudgetSeverity",
    "LedgerError",
    "LedgerSnapshot",
    "ResourceLimit",
    "get_ledger",
    "list_ledgers",
    "make_default_ledger",
    "register_ledger",
]
