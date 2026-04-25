"""Dockable split-pane layout manager for the Aurelius terminal UI surface.

Inspired by kimi-cli (MIT, multi-tab), Claude Code (MIT, panel layouts),
clean-room Aurelius implementation with original branding.

Only rich, stdlib, and project-local imports are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text


class SplitPaneError(Exception):
    """Raised when a SplitPane operation fails."""


class SplitDirection(Enum):
    """Direction in which panes are split."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass
class PaneConfig:
    """Configuration for a single pane in a split layout.

    Attributes:
        pane_id: Unique identifier for this pane.
        title: Display title for the pane.
        weight: Proportional size weight (must be > 0).
        min_size: Minimum size in characters.
        visible: Whether the pane is currently visible.
    """

    pane_id: str
    title: str
    weight: float = 1.0
    min_size: int = 5
    visible: bool = True


class SplitPane:
    """Dockable split-pane layout manager.

    Manages a list of panes in either horizontal or vertical orientation.
    Panes can be shown, hidden, resized, and rendered via Rich.
    """

    def __init__(self, direction: SplitDirection = SplitDirection.HORIZONTAL) -> None:
        self.direction = direction
        self.panes: list[PaneConfig] = []

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_pane(self, config: PaneConfig) -> None:
        """Append *config* to the pane list.

        Args:
            config: The :class:`PaneConfig` to add.

        Raises:
            SplitPaneError: If a pane with the same ``pane_id`` already exists.
        """
        existing_ids = {p.pane_id for p in self.panes}
        if config.pane_id in existing_ids:
            raise SplitPaneError(
                f"pane {config.pane_id!r} already exists in this layout"
            )
        self.panes.append(config)

    def remove_pane(self, pane_id: str) -> None:
        """Remove the pane with *pane_id*.

        Args:
            pane_id: ID of the pane to remove.

        Raises:
            SplitPaneError: If *pane_id* is not found.
        """
        for i, pane in enumerate(self.panes):
            if pane.pane_id == pane_id:
                del self.panes[i]
                return
        raise SplitPaneError(
            f"pane {pane_id!r} not found; "
            f"available: {[p.pane_id for p in self.panes]}"
        )

    def show(self, pane_id: str) -> None:
        """Set the pane with *pane_id* to visible.

        Args:
            pane_id: ID of the pane to show.

        Raises:
            SplitPaneError: If *pane_id* is not found.
        """
        pane = self._get_pane(pane_id)
        pane.visible = True

    def hide(self, pane_id: str) -> None:
        """Set the pane with *pane_id* to not visible.

        Args:
            pane_id: ID of the pane to hide.

        Raises:
            SplitPaneError: If *pane_id* is not found.
        """
        pane = self._get_pane(pane_id)
        pane.visible = False

    def resize(self, pane_id: str, new_weight: float) -> None:
        """Update the weight of the pane with *pane_id*.

        Args:
            pane_id: ID of the pane to resize.
            new_weight: New weight value; must be > 0.

        Raises:
            SplitPaneError: If *pane_id* is not found or *new_weight* <= 0.
        """
        if new_weight <= 0:
            raise SplitPaneError(
                f"weight must be > 0, got {new_weight!r}"
            )
        pane = self._get_pane(pane_id)
        pane.weight = new_weight

    # ------------------------------------------------------------------
    # Render API
    # ------------------------------------------------------------------

    def render(
        self,
        console: Console,
        content_map: dict[str, str] | None = None,
    ) -> None:
        """Render visible panes to *console* as a Rich Table.

        Args:
            console: A :class:`~rich.console.Console` to print to.
            content_map: Optional mapping of ``pane_id`` → text content
                to display inside each pane.
        """
        if content_map is None:
            content_map = {}

        visible = [p for p in self.panes if p.visible]

        if not visible:
            console.print("[dim](no visible panes)[/dim]")
            return

        total_weight = sum(p.weight for p in visible) or 1.0

        table = Table(
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
            expand=True,
        )

        for pane in visible:
            ratio = int(round((pane.weight / total_weight) * 100))
            table.add_column(
                f"{pane.title} ({ratio}%)",
                no_wrap=False,
            )

        row: list[str] = []
        for pane in visible:
            content = content_map.get(pane.pane_id, "")
            row.append(content if content else f"[dim]({pane.pane_id})[/dim]")

        table.add_row(*row)
        console.print(table)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of this layout."""
        return {
            "direction": self.direction.value,
            "panes": [
                {
                    "pane_id": p.pane_id,
                    "title": p.title,
                    "weight": p.weight,
                    "min_size": p.min_size,
                    "visible": p.visible,
                }
                for p in self.panes
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pane(self, pane_id: str) -> PaneConfig:
        """Return the pane with *pane_id* or raise :class:`SplitPaneError`."""
        for pane in self.panes:
            if pane.pane_id == pane_id:
                return pane
        raise SplitPaneError(
            f"pane {pane_id!r} not found; "
            f"available: {[p.pane_id for p in self.panes]}"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Named pool of :class:`SplitPane` instances.
SPLIT_PANE_REGISTRY: dict[str, SplitPane] = {}

__all__ = [
    "SplitDirection",
    "PaneConfig",
    "SplitPane",
    "SplitPaneError",
    "SPLIT_PANE_REGISTRY",
]
