"""Floating hotkey cheatsheet overlay for the Aurelius terminal UI surface.

Inspired by kimi-cli (MIT, multi-tab keyboard help), Claude Code (MIT, panel layouts),
clean-room Aurelius implementation with original branding.

Only rich, stdlib, and project-local imports are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class HotkeyOverlayError(Exception):
    """Raised when a HotkeyOverlay operation fails."""


@dataclass
class HotkeyGroup:
    """A named group of hotkey bindings.

    Attributes:
        name: Display name of this binding group.
        bindings: List of (key, description) pairs.
    """

    name: str
    bindings: list[tuple[str, str]] = field(default_factory=list)


class HotkeyOverlay:
    """Floating hotkey cheatsheet overlay rendered via Rich.

    Manages named groups of hotkey bindings and renders them as either
    a compact 2-column table or full grouped sections.
    """

    def __init__(self) -> None:
        self._groups: dict[str, HotkeyGroup] = {}

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_group(self, group: HotkeyGroup) -> None:
        """Register *group* by name; overwrites an existing entry.

        Args:
            group: The :class:`HotkeyGroup` to register.
        """
        self._groups[group.name] = group

    def remove_group(self, name: str) -> None:
        """Remove the group identified by *name*.

        Args:
            name: The group name to remove.

        Raises:
            HotkeyOverlayError: If *name* is not found.
        """
        if name not in self._groups:
            raise HotkeyOverlayError(
                f"hotkey group {name!r} not found; available: {list(self._groups)}"
            )
        del self._groups[name]

    # ------------------------------------------------------------------
    # Render API
    # ------------------------------------------------------------------

    def render(self, console: Console, compact: bool = False) -> None:
        """Render the hotkey overlay to *console*.

        Args:
            console: A :class:`~rich.console.Console` to print to.
            compact: When ``True`` renders a flat 2-column table of all
                bindings. When ``False`` renders grouped sections.
        """
        if not self._groups:
            console.print(Panel(Text("(no hotkeys registered)", style="dim"), title="Hotkeys"))
            return

        if compact:
            table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
            table.add_column("Key", style="bold yellow", no_wrap=True)
            table.add_column("Description")
            for group in self._groups.values():
                for key, desc in group.bindings:
                    table.add_row(key, desc)
            console.print(Panel(table, title="Hotkeys"))
        else:
            for group in self._groups.values():
                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_column("Key", style="bold yellow", no_wrap=True)
                table.add_column("Description", style="dim")
                for key, desc in group.bindings:
                    table.add_row(key, desc)
                section_title = Text(group.name, style="bold cyan")
                console.print(Panel(table, title=section_title))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of all groups."""
        return {
            name: {
                "name": group.name,
                "bindings": list(group.bindings),
            }
            for name, group in self._groups.items()
        }


# ---------------------------------------------------------------------------
# Registry and default singleton
# ---------------------------------------------------------------------------

#: Named pool of :class:`HotkeyOverlay` instances.
HOTKEY_OVERLAY_REGISTRY: dict[str, HotkeyOverlay] = {}

# Pre-populated default overlay
DEFAULT_HOTKEY_OVERLAY: HotkeyOverlay = HotkeyOverlay()
DEFAULT_HOTKEY_OVERLAY.add_group(
    HotkeyGroup(
        name="Navigation",
        bindings=[
            ("↑", "Move focus up"),
            ("↓", "Move focus down"),
            ("←", "Move focus left"),
            ("→", "Move focus right"),
        ],
    )
)
DEFAULT_HOTKEY_OVERLAY.add_group(
    HotkeyGroup(
        name="Actions",
        bindings=[
            ("Enter", "Confirm selection"),
            ("Esc", "Cancel / close overlay"),
        ],
    )
)
DEFAULT_HOTKEY_OVERLAY.add_group(
    HotkeyGroup(
        name="Palette",
        bindings=[
            ("/", "Open command palette"),
            ("?", "Show help"),
        ],
    )
)

__all__ = [
    "HotkeyGroup",
    "HotkeyOverlay",
    "HotkeyOverlayError",
    "HOTKEY_OVERLAY_REGISTRY",
    "DEFAULT_HOTKEY_OVERLAY",
]
