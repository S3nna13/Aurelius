"""KV cache eviction-policy primitives for Aurelius inference.

This module defines a *policy layer* sitting above any concrete KV/cache
backend. It takes no dependency on Torch, NumPy, or any other foreign
runtime — eviction here is a bookkeeping decision, not a tensor op.

The surface is intentionally minimal:

    engine = EvictionEngine(EvictionPolicy.LRU, capacity_bytes=1024)
    engine.admit(CacheEntry(key="a", size_bytes=256))
    engine.touch("a")
    decision = engine.admit(CacheEntry(key="b", size_bytes=900))
    # decision.evicted -> tuple of keys the engine pushed out

All policies share the same container so a factory (or a higher-level
manager) can swap strategies at runtime without rebuilding state.

Correctness contract
--------------------
* Every violation raises :class:`EvictionError`. There are no silent
  fallbacks and no partial mutations — :meth:`EvictionEngine.admit`
  is transactional: if the requested eviction is infeasible the cache
  state is untouched.
* Ordering is deterministic. Timestamps come from a pluggable
  ``clock_ns`` callable; ties are broken by stable insertion order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import heapq
import time


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class EvictionError(Exception):
    """Raised when eviction-policy invariants are violated."""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class EvictionPolicy(str, Enum):
    """Eviction strategies supported by :class:`EvictionEngine`."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    WEIGHTED = "weighted"
    SINK_PRESERVING = "sink_preserving"


@dataclass(frozen=True)
class CacheEntry:
    """Immutable description of a single cache slot.

    ``weight`` is consumed by the ``WEIGHTED`` policy (higher = more
    valuable per byte). ``is_sink`` is a per-entry hint honored by
    ``LRU`` — the ``SINK_PRESERVING`` policy instead uses the engine's
    ``sink_count`` to pin the first N inserted keys unconditionally.
    """

    key: str
    size_bytes: int
    weight: float = 1.0
    is_sink: bool = False


@dataclass(frozen=True)
class EvictionDecision:
    """Outcome of a single :meth:`EvictionEngine.admit` call."""

    evicted: Tuple[str, ...]
    freed_bytes: int
    policy: EvictionPolicy
    reason: str


# ---------------------------------------------------------------------------
# Pure victim-selection
# ---------------------------------------------------------------------------

def _ordinal(key: str, order: Tuple[str, ...]) -> int:
    """Insertion rank of ``key`` (stable tiebreak)."""

    try:
        return order.index(key)
    except ValueError:  # pragma: no cover - defensive, callers filter first
        raise EvictionError(f"unknown_key: {key!r}")


def select_victims(
    policy: EvictionPolicy,
    entries: Tuple[CacheEntry, ...],
    uses: Dict[str, int],
    last_touch_ns: Dict[str, int],
    order: Tuple[str, ...],
    sink_count: int,
    bytes_needed: int,
) -> Tuple[str, ...]:
    """Return the keys the engine should evict, in eviction order.

    Pure function — no I/O, no global state. Reusable by callers that
    maintain their own cache accounting but want Aurelius's selection
    logic.

    Parameters
    ----------
    policy:
        Which strategy to apply.
    entries:
        Current cache entries. Order is not significant for selection;
        :paramref:`order` carries insertion order.
    uses / last_touch_ns:
        Per-key counters consulted by ``LFU`` / ``LRU``.
    order:
        Insertion order (oldest first). Used by ``FIFO`` and as a
        deterministic tiebreak for the other policies.
    sink_count:
        First ``sink_count`` keys in ``order`` are pinned for
        ``SINK_PRESERVING`` — they can never be evicted.
    bytes_needed:
        Bytes that must be freed. Non-positive means "nothing to do".

    Returns
    -------
    tuple[str, ...]
        Keys to evict, in the order they should be removed. Empty tuple
        if ``bytes_needed <= 0``.

    Raises
    ------
    EvictionError
        If the non-pinned footprint is smaller than ``bytes_needed``.
    """

    if bytes_needed <= 0:
        return ()

    by_key: Dict[str, CacheEntry] = {e.key: e for e in entries}

    # Determine which keys are eligible for eviction.
    if policy is EvictionPolicy.SINK_PRESERVING:
        pinned = set(order[:sink_count])
        candidates = [k for k in order if k not in pinned]
    elif policy is EvictionPolicy.LRU:
        # LRU is the one policy the spec says honors the per-entry
        # is_sink flag — those entries are untouchable.
        candidates = [k for k in order if not by_key[k].is_sink]
    else:
        candidates = list(order)

    if policy is EvictionPolicy.LRU or policy is EvictionPolicy.SINK_PRESERVING:
        # Ascending last-touch; tie-break by insertion order.
        ordering = sorted(
            candidates,
            key=lambda k: (last_touch_ns.get(k, 0), _ordinal(k, order)),
        )
    elif policy is EvictionPolicy.LFU:
        # Fewest uses; tie-break by LRU then insertion order.
        ordering = sorted(
            candidates,
            key=lambda k: (
                uses.get(k, 0),
                last_touch_ns.get(k, 0),
                _ordinal(k, order),
            ),
        )
    elif policy is EvictionPolicy.FIFO:
        ordering = list(candidates)  # already in insertion order
    elif policy is EvictionPolicy.WEIGHTED:
        # Lowest weight-per-byte first. A zero-byte entry has effectively
        # infinite density; keep it last. Tie-break by insertion order
        # so behavior is reproducible.
        def _density(k: str) -> Tuple[float, int]:
            entry = by_key[k]
            if entry.size_bytes <= 0:
                return (float("inf"), _ordinal(k, order))
            return (entry.weight / entry.size_bytes, _ordinal(k, order))

        ordering = sorted(candidates, key=_density)
    else:  # pragma: no cover - enum is exhaustive
        raise EvictionError(f"unsupported_policy: {policy!r}")

    freed = 0
    chosen: List[str] = []
    for k in ordering:
        if freed >= bytes_needed:
            break
        chosen.append(k)
        freed += by_key[k].size_bytes

    if freed < bytes_needed:
        evictable = sum(by_key[k].size_bytes for k in candidates)
        raise EvictionError(
            f"insufficient_capacity: need {bytes_needed}, max evictable {evictable}"
        )

    return tuple(chosen)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EvictionEngine:
    """Bookkeeping-only cache governed by an :class:`EvictionPolicy`.

    The engine tracks keys, sizes, per-key usage counts, and per-key
    last-touch timestamps. It does *not* store payloads — callers are
    expected to keep their own key→value mapping and mirror insert /
    evict decisions.
    """

    def __init__(
        self,
        policy: EvictionPolicy,
        *,
        capacity_bytes: int,
        sink_count: int = 0,
        clock_ns: Optional[Callable[[], int]] = None,
    ) -> None:
        if not isinstance(policy, EvictionPolicy):
            raise EvictionError(f"invalid_policy: {policy!r}")
        if capacity_bytes <= 0:
            raise EvictionError(
                f"invalid_capacity: capacity_bytes must be > 0, got {capacity_bytes}"
            )
        if sink_count < 0:
            raise EvictionError(
                f"invalid_sink_count: sink_count must be >= 0, got {sink_count}"
            )

        self._policy: EvictionPolicy = policy
        self._capacity: int = int(capacity_bytes)
        self._sink_count: int = int(sink_count)
        self._clock_ns: Callable[[], int] = clock_ns or time.monotonic_ns

        self._entries: Dict[str, CacheEntry] = {}
        self._order: List[str] = []
        self._uses: Dict[str, int] = {}
        self._last_touch_ns: Dict[str, int] = {}
        self._total_bytes: int = 0
        # Monotone counter guarantees strict ordering even if the
        # injected clock returns duplicate values.
        self._tick: int = 0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def policy(self) -> EvictionPolicy:
        return self._policy

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def sink_count(self) -> int:
        return self._sink_count

    # ------------------------------------------------------------------
    # Clock helper
    # ------------------------------------------------------------------
    def _now(self) -> int:
        ts = int(self._clock_ns())
        self._tick += 1
        # Strictly increasing synthetic timestamp: base on clock, but
        # always advance at least by `_tick` so equal clock readings
        # still produce distinct values that preserve call order.
        return ts * 1_000_000 + self._tick

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def admit(self, entry: CacheEntry) -> EvictionDecision:
        """Insert or refresh ``entry``, evicting others if necessary.

        Transactional: on any failure path the engine state is unchanged.
        """

        if not isinstance(entry, CacheEntry):
            raise EvictionError(f"invalid_entry: expected CacheEntry, got {type(entry).__name__}")
        if not entry.key:
            raise EvictionError("invalid_entry: key must be non-empty")
        if entry.size_bytes < 0:
            raise EvictionError(
                f"invalid_entry: size_bytes must be >= 0, got {entry.size_bytes}"
            )

        # Refresh path: replace metadata in place, preserving insertion
        # rank and usage history. Capacity is re-checked using the delta.
        if entry.key in self._entries:
            prev = self._entries[entry.key]
            delta = entry.size_bytes - prev.size_bytes
            projected = self._total_bytes + delta
            if projected > self._capacity:
                # Need to make room for the delta. Victim selection
                # excludes the key being refreshed so we never evict
                # the entry we're updating.
                needed = projected - self._capacity
                other_entries = tuple(
                    e for e in self._entries.values() if e.key != entry.key
                )
                other_order = tuple(k for k in self._order if k != entry.key)
                victims = select_victims(
                    self._policy,
                    other_entries,
                    {k: self._uses[k] for k in other_order},
                    {k: self._last_touch_ns[k] for k in other_order},
                    other_order,
                    self._sink_count if entry.key not in self._order[: self._sink_count]
                    else max(0, self._sink_count - 1),
                    needed,
                )
                freed = self._apply_evictions(victims)
                self._entries[entry.key] = entry
                self._total_bytes += delta
                self._last_touch_ns[entry.key] = self._now()
                return EvictionDecision(
                    evicted=victims,
                    freed_bytes=freed,
                    policy=self._policy,
                    reason="refresh_with_eviction",
                )

            self._entries[entry.key] = entry
            self._total_bytes += delta
            self._last_touch_ns[entry.key] = self._now()
            return EvictionDecision(
                evicted=(),
                freed_bytes=0,
                policy=self._policy,
                reason="refresh",
            )

        # Fresh admission — reject anything larger than the whole cache
        # up-front so we don't churn the selector for an impossible ask.
        if entry.size_bytes > self._capacity:
            raise EvictionError(
                f"insufficient_capacity: need {entry.size_bytes}, max evictable "
                f"{self._capacity}"
            )

        projected = self._total_bytes + entry.size_bytes
        victims: Tuple[str, ...] = ()
        freed = 0
        if projected > self._capacity:
            needed = projected - self._capacity
            victims = select_victims(
                self._policy,
                tuple(self._entries.values()),
                dict(self._uses),
                dict(self._last_touch_ns),
                tuple(self._order),
                self._sink_count,
                needed,
            )
            freed = self._apply_evictions(victims)

        self._entries[entry.key] = entry
        self._order.append(entry.key)
        self._uses[entry.key] = 0
        self._last_touch_ns[entry.key] = self._now()
        self._total_bytes += entry.size_bytes

        if victims:
            return EvictionDecision(
                evicted=victims,
                freed_bytes=freed,
                policy=self._policy,
                reason="evicted_for_admit",
            )
        return EvictionDecision(
            evicted=(),
            freed_bytes=0,
            policy=self._policy,
            reason="admit",
        )

    def touch(self, key: str) -> None:
        """Mark ``key`` as recently used and bump its usage count."""

        if key not in self._entries:
            raise EvictionError(f"unknown_key: {key!r}")
        self._last_touch_ns[key] = self._now()
        self._uses[key] = self._uses.get(key, 0) + 1

    def evict(self, key: str) -> CacheEntry:
        """Explicitly remove ``key`` and return the removed entry."""

        if key not in self._entries:
            raise EvictionError(f"unknown_key: {key!r}")
        entry = self._entries.pop(key)
        self._order.remove(key)
        self._uses.pop(key, None)
        self._last_touch_ns.pop(key, None)
        self._total_bytes -= entry.size_bytes
        return entry

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def total_bytes(self) -> int:
        return self._total_bytes

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> Tuple[str, ...]:
        """Return currently-resident keys in deterministic (sorted) order."""

        return tuple(sorted(self._entries.keys()))

    def snapshot(self) -> Dict[str, object]:
        """Return a JSON-serializable view of engine state."""

        entries = []
        for key in self._order:
            entry = self._entries[key]
            entries.append(
                {
                    "key": entry.key,
                    "size_bytes": entry.size_bytes,
                    "weight": entry.weight,
                    "is_sink": entry.is_sink,
                    "uses": self._uses.get(key, 0),
                    "last_touch_ns": self._last_touch_ns.get(key, 0),
                }
            )
        return {
            "policy": self._policy.value,
            "capacity_bytes": self._capacity,
            "sink_count": self._sink_count,
            "total_bytes": self._total_bytes,
            "entries": entries,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _apply_evictions(self, victims: Tuple[str, ...]) -> int:
        freed = 0
        for key in victims:
            entry = self._entries.pop(key)
            # O(n) but n is bounded by cache size; stays deterministic.
            self._order.remove(key)
            self._uses.pop(key, None)
            self._last_touch_ns.pop(key, None)
            self._total_bytes -= entry.size_bytes
            freed += entry.size_bytes
        return freed


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EVICTION_POLICY_REGISTRY: Dict[str, type] = {p.value: EvictionEngine for p in EvictionPolicy}


__all__ = [
    "CacheEntry",
    "EVICTION_POLICY_REGISTRY",
    "EvictionDecision",
    "EvictionEngine",
    "EvictionError",
    "EvictionPolicy",
    "select_victims",
]
