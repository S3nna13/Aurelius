"""Context Budget Controller — adaptive 256K token budget prioritization.

Inspired by Kimi K2.6's 256K context management strategy.

At 256K tokens, naive context management leads to important tokens being
truncated. This controller allocates token budget across named segments
(system, tools, history, thinking, user) by priority, then evicts /
truncates lower-priority segments when the total exceeds the budget.

Eviction order:
    BACKGROUND (4) → LOW (3) → MEDIUM (2) → HIGH (1) → CRITICAL (0)
Within the same priority level, the largest segment is evicted first.
CRITICAL segments are NEVER evicted (evictable=False).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class SegmentPriority(IntEnum):
    CRITICAL = 0  # system prompt, current user turn — never evicted
    HIGH = 1  # recent conversation turns, tool results
    MEDIUM = 2  # older turns, tool definitions
    LOW = 3  # thinking chains, background context
    BACKGROUND = 4  # auxiliary context, retrieved docs


@dataclass
class ContextSegment:
    """A named, token-counted slice of the context window.

    Attributes:
        name: Unique identifier for this segment.
        tokens: Token count consumed by this segment.
        priority: Eviction priority (lower value = evicted later).
        evictable: When False the segment is never removed (e.g. CRITICAL).
        metadata: Arbitrary caller-supplied key/value pairs.
    """

    name: str
    tokens: int
    priority: SegmentPriority
    evictable: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class ContextBudgetConfig:
    """Configuration for :class:`ContextBudgetController`.

    Attributes:
        max_tokens: Hard upper bound on total context (default 256K).
        reserve_tokens: Tokens set aside for generation output.
        trigger_ratio: Fraction of usable tokens at which eviction starts.
    """

    max_tokens: int = 262144  # 256K
    reserve_tokens: int = 4096
    trigger_ratio: float = 0.85


class ContextBudgetController:
    """Manages token budget across prioritised context segments.

    Usage::

        config = ContextBudgetConfig(max_tokens=262144)
        ctrl = ContextBudgetController(config)
        ctrl.add_segment(ContextSegment("system", 512, SegmentPriority.CRITICAL, evictable=False))
        ctrl.add_segment(ContextSegment("history", 80000, SegmentPriority.HIGH))
        evicted = ctrl.evict()
    """

    def __init__(self, config: ContextBudgetConfig | None = None) -> None:
        self._config = config or ContextBudgetConfig()
        # Ordered dict preserves insertion order; key = segment name
        self._segments: dict[str, ContextSegment] = {}

    # ------------------------------------------------------------------
    # Segment management
    # ------------------------------------------------------------------

    def add_segment(self, segment: ContextSegment) -> None:
        """Add a segment to the controller.

        Raises:
            ValueError: If a segment with the same name already exists.
        """
        if segment.name in self._segments:
            raise ValueError(
                f"Segment '{segment.name}' already exists. Remove it first or use a unique name."
            )
        self._segments[segment.name] = segment

    def remove_segment(self, name: str) -> bool:
        """Remove the segment identified by *name*.

        Returns:
            True if the segment was found and removed, False otherwise.
        """
        if name in self._segments:
            del self._segments[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Budget queries
    # ------------------------------------------------------------------

    def total_tokens(self) -> int:
        """Return the sum of tokens across all current segments."""
        return sum(seg.tokens for seg in self._segments.values())

    def available_tokens(self) -> int:
        """Return tokens still available for new content.

        ``available = max_tokens - reserve_tokens - total_tokens``
        """
        usable = self._config.max_tokens - self._config.reserve_tokens
        return usable - self.total_tokens()

    def needs_eviction(self) -> bool:
        """Return True when total tokens exceed the trigger threshold.

        Trigger: ``total > trigger_ratio * (max_tokens - reserve_tokens)``
        """
        threshold = self._config.trigger_ratio * (
            self._config.max_tokens - self._config.reserve_tokens
        )
        return self.total_tokens() > threshold

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict(self) -> list[str]:
        """Evict segments until the budget is no longer over-threshold.

        Eviction policy:
        - Segments with ``evictable=False`` are never removed.
        - Among evictable segments, highest priority value (BACKGROUND=4)
          is evicted first.
        - Within the same priority level, the segment with the most tokens
          is evicted first (largest first).

        Returns:
            Ordered list of names of evicted segments.
        """
        evicted: list[str] = []

        while self.needs_eviction():
            # Collect candidates: evictable segments only
            candidates = [seg for seg in self._segments.values() if seg.evictable]
            if not candidates:
                # Nothing left to evict — cannot reduce further
                break

            # Sort: highest priority value first (BACKGROUND=4 before LOW=3),
            # then within same priority, largest tokens first.
            candidates.sort(key=lambda s: (-s.priority, -s.tokens))
            victim = candidates[0]
            self._segments.pop(victim.name)
            evicted.append(victim.name)

        return evicted

    # ------------------------------------------------------------------
    # Batch allocation
    # ------------------------------------------------------------------

    def allocate(self, segments: list[ContextSegment]) -> list[ContextSegment]:
        """Add all *segments* then evict as needed.

        Args:
            segments: Proposed segments to insert.

        Returns:
            The subset of *segments* that survived eviction (not evicted).
            Pre-existing segments that survive are NOT included in the
            return value; only the newly added ones that remain.
        """
        # Track which names we are adding in this call
        new_names: set[str] = set()
        for seg in segments:
            self.add_segment(seg)
            new_names.add(seg.name)

        evicted_names = set(self.evict())

        # Return segments from the proposed list that were not evicted
        return [seg for seg in segments if seg.name not in evicted_names]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def budget_summary(self) -> dict:
        """Return a summary of current budget utilisation.

        Returns::

            {
                "total": int,
                "available": int,
                "by_priority": {
                    "CRITICAL": int,
                    "HIGH": int,
                    "MEDIUM": int,
                    "LOW": int,
                    "BACKGROUND": int,
                }
            }
        """
        by_priority: dict[str, int] = {p.name: 0 for p in SegmentPriority}
        for seg in self._segments.values():
            by_priority[seg.priority.name] += seg.tokens

        return {
            "total": self.total_tokens(),
            "available": self.available_tokens(),
            "by_priority": by_priority,
        }
