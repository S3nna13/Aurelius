"""Right-click/context menu: items, sections, keyboard-driven selection.

Only rich, stdlib, and project-local imports are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console
from rich.rule import Rule
from rich.text import Text


class MenuItemKind(str, Enum):
    """Discriminator for menu item types."""

    ACTION = "action"
    SEPARATOR = "separator"
    SUBMENU = "submenu"


@dataclass
class MenuItem:
    """A single entry in a :class:`ContextMenu`.

    Attributes:
        label: Display label shown in the menu.
        kind: The type of this menu item.
        shortcut: Optional keyboard shortcut displayed on the right.
        enabled: Whether the item can be selected.
        action_id: Opaque identifier for the action to invoke.
    """

    label: str
    kind: MenuItemKind
    shortcut: str = ""
    enabled: bool = True
    action_id: str = ""


class ContextMenu:
    """A keyboard-driven context menu rendered via Rich.

    Items are appended in insertion order.  Separators are plain
    :class:`MenuItem` objects with ``kind=MenuItemKind.SEPARATOR``.
    """

    def __init__(self, title: str = "") -> None:
        self._title = title
        self._items: list[MenuItem] = []

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_item(self, item: MenuItem) -> None:
        """Append *item* to the menu.

        Args:
            item: The :class:`MenuItem` to add.
        """
        self._items.append(item)

    def add_separator(self) -> None:
        """Append a visual separator to the menu."""
        self._items.append(MenuItem(label="", kind=MenuItemKind.SEPARATOR))

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def select(self, index: int) -> MenuItem | None:
        """Return the item at *index* if it is an enabled ACTION, else ``None``.

        Args:
            index: Zero-based position in the item list.

        Returns:
            The :class:`MenuItem` if selectable, ``None`` otherwise.
        """
        if index < 0 or index >= len(self._items):
            return None
        item = self._items[index]
        if item.kind == MenuItemKind.ACTION and item.enabled:
            return item
        return None

    def enabled_items(self) -> list[tuple[int, MenuItem]]:
        """Return ``(original_index, item)`` pairs for enabled ACTION items.

        Returns:
            A list of tuples preserving original insertion order.
        """
        return [
            (i, item)
            for i, item in enumerate(self._items)
            if item.kind == MenuItemKind.ACTION and item.enabled
        ]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Return a Rich-formatted string representing the menu.

        The output includes:
        - An optional title header (if set).
        - Each ACTION item with its label and right-aligned shortcut.
        - Disabled items rendered in dim style.
        - SEPARATOR items rendered as horizontal rules.

        Returns:
            A string suitable for printing to the terminal.
        """
        console = Console(highlight=False)
        lines: list[object] = []

        if self._title:
            lines.append(Text(f" {self._title} ", style="bold"))
            lines.append(Rule(style="dim"))

        for item in self._items:
            if item.kind == MenuItemKind.SEPARATOR:
                lines.append(Rule(style="dim"))
            elif item.kind in (MenuItemKind.ACTION, MenuItemKind.SUBMENU):
                row = Text()
                label_style = "dim" if not item.enabled else ""
                row.append(item.label, style=label_style)
                if item.shortcut:
                    # Right-align the shortcut by padding
                    row.append(f"  {item.shortcut}", style="dim")
                if item.kind == MenuItemKind.SUBMENU:
                    row.append(" ▸", style="dim")
                lines.append(row)

        with console.capture() as capture:
            for line in lines:
                console.print(line)
        return capture.get()


class ContextMenuRegistry:
    """A named registry of :class:`ContextMenu` objects."""

    def __init__(self) -> None:
        self._store: dict[str, ContextMenu] = {}

    def register(self, name: str, menu: ContextMenu) -> None:
        """Register *menu* under *name*, overwriting any existing entry.

        Args:
            name: Unique identifier for this menu.
            menu: The :class:`ContextMenu` to store.
        """
        self._store[name] = menu

    def get(self, name: str) -> ContextMenu:
        """Return the :class:`ContextMenu` registered as *name*.

        Args:
            name: The identifier to look up.

        Raises:
            KeyError: If no menu with that name is registered.
        """
        return self._store[name]

    def list_menus(self) -> list[str]:
        """Return a sorted list of all registered menu names."""
        return sorted(self._store.keys())


# ---------------------------------------------------------------------------
# Module-level singleton with pre-registered default menus
# ---------------------------------------------------------------------------

CONTEXT_MENU_REGISTRY = ContextMenuRegistry()

_editor_menu = ContextMenu(title="Editor")
_editor_menu.add_item(MenuItem(label="Copy", kind=MenuItemKind.ACTION, shortcut="Ctrl+C", action_id="copy"))
_editor_menu.add_item(MenuItem(label="Paste", kind=MenuItemKind.ACTION, shortcut="Ctrl+V", action_id="paste"))
_editor_menu.add_separator()
_editor_menu.add_item(MenuItem(label="Select All", kind=MenuItemKind.ACTION, shortcut="Ctrl+A", action_id="select_all"))
CONTEXT_MENU_REGISTRY.register("editor", _editor_menu)

_session_menu = ContextMenu(title="Session")
_session_menu.add_item(MenuItem(label="New Session", kind=MenuItemKind.ACTION, shortcut="Ctrl+N", action_id="new_session"))
_session_menu.add_item(MenuItem(label="Close Session", kind=MenuItemKind.ACTION, shortcut="Ctrl+W", action_id="close_session"))
CONTEXT_MENU_REGISTRY.register("session", _session_menu)
