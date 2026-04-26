"""Coordinate-to-UI-element mapping for the Aurelius computer_use surface.

Maps screen coordinates to accessibility tree elements with z-order
resolution and fuzzy near-miss matching.  Pure stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.computer_use.screen_parser import AccessibilityNode, ScreenSnapshot

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VisualGroundingError(Exception):
    """Raised when visual grounding fails or receives invalid input."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GroundedElement:
    """A UI element matched to a coordinate pair."""

    node: AccessibilityNode
    distance: float
    z_index: int
    fuzzy: bool = False


@dataclass
class GroundingResult:
    """The full result of grounding (x, y) against an accessibility tree."""

    exact: GroundedElement | None = None
    fuzzy: GroundedElement | None = None
    all_matches: list[GroundedElement] = field(default_factory=list)
    x: int = 0
    y: int = 0


# ---------------------------------------------------------------------------
# Main grounding interface
# ---------------------------------------------------------------------------


class VisualGrounding:
    """Maps (x, y) screen coordinates to UI elements from an accessibility tree.

    Resolution strategy:

    1. Exact containment — elements whose bounding box strictly contains (x, y).
    2. Z-order — among overlapping elements, children rank above parents
       and later siblings rank above earlier siblings (document order).
    3. Fuzzy fallback — when no exact match exists, find the closest element
       within a configurable pixel tolerance (default 10 px).
    """

    def __init__(self, fuzzy_tolerance: int = 10) -> None:
        if fuzzy_tolerance < 0:
            raise VisualGroundingError(
                f"fuzzy_tolerance must be non-negative, got {fuzzy_tolerance}"
            )
        self._fuzzy_tolerance: int = fuzzy_tolerance

    def ground(
        self,
        x: int,
        y: int,
        snapshot: ScreenSnapshot,
    ) -> GroundingResult:
        """Ground *x*, *y* to a UI element in the accessibility tree.

        Parameters
        ----------
        x:
            X coordinate in screen pixels.
        y:
            Y coordinate in screen pixels.
        snapshot:
            Parsed screen accessibility tree.

        Returns
        -------
        GroundingResult

        Raises
        ------
        VisualGroundingError
            If coordinates are out of bounds for the snapshot dimensions.
        """
        if x < 0 or y < 0 or x > snapshot.width or y > snapshot.height:
            raise VisualGroundingError(
                f"Coordinates ({x}, {y}) out of bounds for "
                f"({snapshot.width} x {snapshot.height}) screen"
            )

        flat = self._flatten(snapshot.root_node, z_base=0)
        if not flat:
            return GroundingResult(x=x, y=y)

        with_dist = [(node, self._distance(x, y, node.bbox), z) for node, z in flat]

        exact = [
            GroundedElement(node=node, distance=d, z_index=z)
            for node, d, z in with_dist
            if self._contains(x, y, node.bbox)
        ]
        exact.sort(key=lambda e: (-e.z_index, e.distance))

        exact_match: GroundedElement | None = exact[0] if exact else None
        fuzzy_match: GroundedElement | None = None

        if exact_match is None and self._fuzzy_tolerance > 0:
            candidates = [
                GroundedElement(node=node, distance=d, z_index=z, fuzzy=True)
                for node, d, z in with_dist
                if d <= self._fuzzy_tolerance
            ]
            if candidates:
                candidates.sort(key=lambda e: (e.distance, -e.z_index))
                fuzzy_match = candidates[0]

        return GroundingResult(
            exact=exact_match,
            fuzzy=fuzzy_match,
            all_matches=exact,
            x=x,
            y=y,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(
        node: AccessibilityNode,
        z_base: int = 0,
    ) -> list[tuple[AccessibilityNode, int]]:
        """Walk the tree depth-first; return (node, z_index) for nodes with bboxes.

        Z-order: children get a higher z than their parents.  Among siblings,
        later elements in document order get higher z.
        """
        result: list[tuple[AccessibilityNode, int]] = []
        if node.bbox is not None:
            result.append((node, z_base))
        for i, child in enumerate(node.children):
            child_z = z_base + (i + 1)
            result.extend(VisualGrounding._flatten(child, child_z))
        return result

    @staticmethod
    def _contains(
        x: int,
        y: int,
        bbox: tuple[int, int, int, int],
    ) -> bool:
        """Return True when (x, y) falls inside *bbox* (x, y, w, h)."""
        bx, by, bw, bh = bbox
        return bx <= x <= bx + bw and by <= y <= by + bh

    @staticmethod
    def _distance(
        x: int,
        y: int,
        bbox: tuple[int, int, int, int],
    ) -> float:
        """Euclidean distance from (x, y) to the closest point on *bbox*."""
        bx, by, bw, bh = bbox
        cx = max(bx, min(x, bx + bw))
        cy = max(by, min(y, by + bh))
        return math.sqrt(float((x - cx) ** 2 + (y - cy) ** 2))


# ---------------------------------------------------------------------------
# Compound strategy (multiple tolerance passes)
# ---------------------------------------------------------------------------


class CompoundVisualGrounding:
    """Runs multiple :class:`VisualGrounding` strategies in order.

    Each strategy is tried until an exact match is found.  If none are found
    the last fuzzy result is returned.
    """

    def __init__(
        self,
        strategies: list[VisualGrounding] | None = None,
    ) -> None:
        self._strategies: list[VisualGrounding] = strategies or [
            VisualGrounding(fuzzy_tolerance=5),
            VisualGrounding(fuzzy_tolerance=10),
            VisualGrounding(fuzzy_tolerance=20),
        ]

    def ground(
        self,
        x: int,
        y: int,
        snapshot: ScreenSnapshot,
    ) -> GroundingResult:
        """Try each strategy; return first exact match or last fuzzy result."""
        last = GroundingResult(x=x, y=y)
        for strategy in self._strategies:
            result = strategy.ground(x, y, snapshot)
            if result.exact is not None:
                return result
            if result.fuzzy is not None:
                last = result
        return last


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VISUAL_GROUNDING_REGISTRY: dict[str, type[VisualGrounding]] = {
    "default": VisualGrounding,
    "compound": CompoundVisualGrounding,
}


def register_visual_grounding(name: str, cls: type) -> None:
    """Register a VisualGrounding implementation under *name*."""
    VISUAL_GROUNDING_REGISTRY[name] = cls


def get_visual_grounding(name: str) -> type:
    """Retrieve a registered VisualGrounding class by *name*.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name not in VISUAL_GROUNDING_REGISTRY:
        raise KeyError(
            f"No VisualGrounding registered as {name!r}. "
            f"Available: {sorted(VISUAL_GROUNDING_REGISTRY)}"
        )
    return VISUAL_GROUNDING_REGISTRY[name]
