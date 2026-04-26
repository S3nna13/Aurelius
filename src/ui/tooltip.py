"""Tooltip system: hover-delay, position calculation, registry.

Only rich, stdlib, and project-local imports are used.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

_VALID_POSITIONS = {"above", "below", "left", "right"}


@dataclass
class TooltipConfig:
    """Configuration for a single tooltip.

    Attributes:
        text: The tooltip message to display.
        delay_ms: Hover delay in milliseconds before the tooltip appears.
        max_width: Maximum character width before text is wrapped.
        position: Preferred display position relative to the anchor element.
    """

    text: str
    delay_ms: int = 300
    max_width: int = 40
    position: str = "above"

    def __post_init__(self) -> None:
        if self.position not in _VALID_POSITIONS:
            raise ValueError(
                f"Invalid position {self.position!r}; must be one of {sorted(_VALID_POSITIONS)}"
            )


class TooltipRenderer:
    """Renders and tracks visibility state for a single tooltip.

    Usage::

        renderer = TooltipRenderer()
        output = renderer.show(config)   # returns Rich-formatted string
        renderer.hide()
    """

    def __init__(self) -> None:
        self._visible: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self, config: TooltipConfig) -> str:
        """Render *config* as a Rich-formatted tooltip string.

        The tooltip text is wrapped to *config.max_width* characters and
        enclosed in a bordered panel.  The position hint is included as a
        subtitle so callers can use it for layout decisions.

        Args:
            config: The :class:`TooltipConfig` to render.

        Returns:
            A Rich-formatted string ready for printing to the terminal.
        """
        self._visible = True
        wrapped = textwrap.fill(config.text, width=config.max_width)
        panel = Panel(
            Text(wrapped),
            subtitle=f"[dim]{config.position}[/dim]",
            border_style="dim cyan",
            padding=(0, 1),
            expand=False,
        )
        console = Console(highlight=False)
        with console.capture() as capture:
            console.print(panel)
        return capture.get()

    def hide(self) -> None:
        """Mark the tooltip as hidden."""
        self._visible = False

    def is_visible(self) -> bool:
        """Return ``True`` if the tooltip is currently visible."""
        return self._visible


class TooltipRegistry:
    """A named registry of :class:`TooltipConfig` objects.

    Allows tooltips to be registered once by name and retrieved later by
    any part of the UI.
    """

    def __init__(self) -> None:
        self._store: dict[str, TooltipConfig] = {}

    def register(self, name: str, config: TooltipConfig) -> None:
        """Register *config* under *name*, overwriting any existing entry.

        Args:
            name: Unique identifier for this tooltip.
            config: The :class:`TooltipConfig` to store.
        """
        self._store[name] = config

    def get(self, name: str) -> TooltipConfig:
        """Return the :class:`TooltipConfig` registered as *name*.

        Args:
            name: The identifier to look up.

        Raises:
            KeyError: If no tooltip with that name is registered.
        """
        return self._store[name]

    def list_tooltips(self) -> list[str]:
        """Return a sorted list of all registered tooltip names."""
        return sorted(self._store.keys())


# ---------------------------------------------------------------------------
# Module-level singleton with pre-registered default tooltips
# ---------------------------------------------------------------------------

TOOLTIP_REGISTRY = TooltipRegistry()
TOOLTIP_REGISTRY.register("submit", TooltipConfig(text="Submit input [Enter]"))
TOOLTIP_REGISTRY.register("clear", TooltipConfig(text="Clear session [Ctrl+L]"))
TOOLTIP_REGISTRY.register("model", TooltipConfig(text="Show model info [Ctrl+M]"))
