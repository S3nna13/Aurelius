"""Aurelius computer_use — screen_state_tracker.py

Tracks successive screen states and surfaces diffs between them.
All logic is pure Python; no OS accessibility API or image library required.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScreenRegion:
    x: int
    y: int
    width: int
    height: int
    label: str = ""
    # Extra metadata forwarded from raw state dicts (e.g. interactive, role …)
    metadata: dict = field(default_factory=dict)


@dataclass
class ScreenState:
    timestamp: float
    regions: list[ScreenRegion]
    focused_region: str | None
    change_mask: list[bool]  # True = region changed since previous state


# ---------------------------------------------------------------------------
# ScreenStateTracker
# ---------------------------------------------------------------------------


class ScreenStateTracker:
    """Parse raw state dicts into ScreenState objects and track history."""

    _MAX_HISTORY = 10

    def __init__(self) -> None:
        self._history: deque[ScreenState] = deque(maxlen=self._MAX_HISTORY)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, new_state: dict) -> ScreenState:
        """Parse *new_state* dict and append to history. Returns the new ScreenState."""
        regions = self._parse_regions(new_state.get("regions", []))
        focused = new_state.get("focused_region") or new_state.get("focused")
        ts = new_state.get("timestamp", time.time())

        # Build change mask vs. previous state
        prev = self._history[-1] if self._history else None
        change_mask = self._compute_change_mask(regions, prev)

        state = ScreenState(
            timestamp=float(ts),
            regions=regions,
            focused_region=str(focused) if focused is not None else None,
            change_mask=change_mask,
        )
        self._history.append(state)
        return state

    def diff(self, prev: ScreenState, curr: ScreenState) -> list[str]:
        """Return human-readable descriptions of what changed between two states."""
        changes: list[str] = []

        prev_labels = {r.label for r in prev.regions}
        curr_labels = {r.label for r in curr.regions}

        added = curr_labels - prev_labels
        removed = prev_labels - curr_labels

        for label in sorted(added):
            changes.append(f"Region added: '{label}'")
        for label in sorted(removed):
            changes.append(f"Region removed: '{label}'")

        # Check geometry / metadata changes for regions present in both
        prev_map = {r.label: r for r in prev.regions}
        curr_map = {r.label: r for r in curr.regions}
        for label in sorted(prev_labels & curr_labels):
            p, c = prev_map[label], curr_map[label]
            if (p.x, p.y, p.width, p.height) != (c.x, c.y, c.width, c.height):
                changes.append(
                    f"Region '{label}' geometry changed: "
                    f"({p.x},{p.y},{p.width},{p.height}) → ({c.x},{c.y},{c.width},{c.height})"
                )

        if prev.focused_region != curr.focused_region:
            changes.append(f"Focus changed: '{prev.focused_region}' → '{curr.focused_region}'")

        return changes

    def get_interactive_regions(self) -> list[ScreenRegion]:
        """Return regions from the latest state that have interactive=True in metadata."""
        if not self._history:
            return []
        latest = self._history[-1]
        return [r for r in latest.regions if r.metadata.get("interactive", False)]

    @property
    def history(self) -> list[ScreenState]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_regions(raw: list[dict]) -> list[ScreenRegion]:
        regions: list[ScreenRegion] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            meta = {
                k: v for k, v in item.items() if k not in ("x", "y", "width", "height", "label")
            }
            regions.append(
                ScreenRegion(
                    x=int(item.get("x", 0)),
                    y=int(item.get("y", 0)),
                    width=int(item.get("width", 0)),
                    height=int(item.get("height", 0)),
                    label=str(item.get("label", "")),
                    metadata=meta,
                )
            )
        return regions

    @staticmethod
    def _compute_change_mask(
        regions: list[ScreenRegion],
        prev: ScreenState | None,
    ) -> list[bool]:
        if prev is None:
            return [True] * len(regions)
        prev_map = {r.label: r for r in prev.regions}
        mask: list[bool] = []
        for r in regions:
            if r.label not in prev_map:
                mask.append(True)
            else:
                p = prev_map[r.label]
                changed = (p.x, p.y, p.width, p.height) != (r.x, r.y, r.width, r.height)
                mask.append(changed)
        return mask


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

COMPUTER_USE_REGISTRY: dict[str, Any] = {}
COMPUTER_USE_REGISTRY["screen_state_tracker"] = ScreenStateTracker
