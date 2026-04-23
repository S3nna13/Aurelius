"""Command palette for the Aurelius terminal UI surface.

Inspired by MoonshotAI/kimi-cli (MIT, terminal session lifecycle),
Anthropic Claude Code (MIT, command palette UX), clean-room
reimplementation with original Aurelius branding.

Provides a keyboard-first, reduced-motion aware command palette backed
by a class-level registry.  Only rich, stdlib, and project-local imports
are used.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable

from rich.console import Console
from rich.table import Table


class CommandPaletteError(Exception):
    """Raised when a command palette operation fails."""


class CommandPaletteState(enum.Enum):
    """Life-cycle states for the command palette overlay."""

    CLOSED = "closed"
    OPEN = "open"
    SEARCHING = "searching"
    EXECUTING = "executing"


@dataclass
class CommandEntry:
    """A single entry in the command palette registry.

    Attributes:
        name: Unique command identifier (e.g. ``"clear"``).
        description: Short human-readable description shown in the
            palette table.
        shortcut: Optional keyboard shortcut string (e.g. ``"ctrl-k"``).
        category: Grouping label (e.g. ``"session"``, ``"navigation"``).
        handler: Zero-argument callable invoked by :meth:`CommandPalette.execute`.
            ``None`` means the command is registered but not yet
            implemented; execution is silently skipped.
    """

    name: str
    description: str
    shortcut: str | None = None
    category: str = "general"
    handler: Callable[[], None] | None = None


def _score(entry: CommandEntry, query: str) -> int:
    """Return a non-negative match score; higher is better.

    Uses a simple substring-frequency heuristic over lowercased tokens
    so that no external fuzzy-match library is required.
    """
    q = query.lower()
    tokens = q.split()
    text = " ".join(
        [entry.name.lower(), entry.description.lower(), entry.category.lower()]
    )
    if not tokens:
        return 1  # empty query matches everything
    score = 0
    for tok in tokens:
        if tok in entry.name.lower():
            score += 3  # name match weighted more
        if tok in entry.description.lower():
            score += 2
        if tok in entry.category.lower():
            score += 1
    # Subsequence bonus: every character of query appears in name in order
    name_lower = entry.name.lower()
    qi = 0
    for ch in name_lower:
        if qi < len(q) and ch == q[qi]:
            qi += 1
    if qi == len(q) and q:
        score += 2
    return score


class CommandPalette:
    """Registry and renderer for Aurelius command palette entries.

    The palette is class-level so that commands registered at module
    import time are available everywhere without passing instances
    around.

    Usage::

        CommandPalette.register(CommandEntry("my-cmd", "Do something"))
        results = CommandPalette.search("my")
        CommandPalette.execute("my-cmd")
    """

    COMMAND_PALETTE_REGISTRY: dict[str, CommandEntry] = {}

    @classmethod
    def register(cls, entry: CommandEntry) -> None:
        """Register *entry* in the palette.

        Args:
            entry: The :class:`CommandEntry` to register.

        Raises:
            CommandPaletteError: If *entry* is not a
                :class:`CommandEntry` or if its name is already present.
        """
        if not isinstance(entry, CommandEntry):
            raise CommandPaletteError("register() requires a CommandEntry instance")
        if entry.name in cls.COMMAND_PALETTE_REGISTRY:
            raise CommandPaletteError(
                f"command {entry.name!r} is already registered in the palette"
            )
        cls.COMMAND_PALETTE_REGISTRY[entry.name] = entry

    @classmethod
    def search(cls, query: str) -> list[CommandEntry]:
        """Return entries whose name, description, or category match *query*.

        Matching uses a pure-stdlib substring/subsequence heuristic.
        Results are sorted by descending match score.  A blank *query*
        returns all entries sorted by name.

        Args:
            query: The search string.  May be empty.

        Returns:
            A list of matching :class:`CommandEntry` objects.  Empty
            list if no entries match.
        """
        if not isinstance(query, str):
            query = ""
        if not query.strip():
            return sorted(
                cls.COMMAND_PALETTE_REGISTRY.values(), key=lambda e: e.name
            )
        scored = [
            (entry, _score(entry, query))
            for entry in cls.COMMAND_PALETTE_REGISTRY.values()
        ]
        matched = [(e, s) for e, s in scored if s > 0]
        matched.sort(key=lambda t: -t[1])
        return [e for e, _ in matched]

    @classmethod
    def execute(cls, name: str) -> None:
        """Execute the command registered under *name*.

        If the command's ``handler`` is ``None`` the call is a no-op
        (the command is intentionally stub-registered).

        Args:
            name: The command name to execute.

        Raises:
            CommandPaletteError: If *name* is not in the registry.
        """
        if name not in cls.COMMAND_PALETTE_REGISTRY:
            raise CommandPaletteError(
                f"no command {name!r} registered in the palette"
            )
        entry = cls.COMMAND_PALETTE_REGISTRY[name]
        if entry.handler is not None:
            entry.handler()

    @classmethod
    def render(cls, console: Console, query: str = "") -> None:
        """Render the palette as a Rich Table to *console*.

        The table shows name, description, shortcut, and category
        columns.  If *query* is provided only matching entries are
        shown; otherwise all entries are listed.

        Args:
            console: A :class:`rich.console.Console` instance.
            query: Optional search filter string.
        """
        entries = cls.search(query)
        table = Table(
            title="Aurelius Command Palette",
            show_header=True,
            header_style="bold cyan",
            show_lines=False,
            expand=False,
        )
        table.add_column("Name", style="bold", no_wrap=True)
        table.add_column("Description")
        table.add_column("Shortcut", style="dim", no_wrap=True)
        table.add_column("Category", style="italic", no_wrap=True)

        for entry in entries:
            table.add_row(
                entry.name,
                entry.description,
                entry.shortcut or "",
                entry.category,
            )

        console.print(table)


# ---------------------------------------------------------------------------
# Built-in command registrations
# ---------------------------------------------------------------------------

_BUILTIN_COMMANDS: list[CommandEntry] = [
    CommandEntry(
        name="clear",
        description="Clear the terminal screen",
        shortcut="ctrl-l",
        category="session",
        handler=None,
    ),
    CommandEntry(
        name="help",
        description="Show help and available commands",
        shortcut="?",
        category="navigation",
        handler=None,
    ),
    CommandEntry(
        name="quit",
        description="Quit Aurelius",
        shortcut="ctrl-q",
        category="session",
        handler=None,
    ),
    CommandEntry(
        name="toggle-motion",
        description="Toggle reduced-motion mode for accessibility",
        shortcut=None,
        category="accessibility",
        handler=None,
    ),
    CommandEntry(
        name="show-branding",
        description="Display Aurelius branding and version info",
        shortcut=None,
        category="navigation",
        handler=None,
    ),
]

for _entry in _BUILTIN_COMMANDS:
    CommandPalette.register(_entry)

# Module-level alias required by the integration spec.
COMMAND_PALETTE_REGISTRY = CommandPalette.COMMAND_PALETTE_REGISTRY

__all__ = [
    "CommandEntry",
    "CommandPaletteError",
    "CommandPaletteState",
    "CommandPalette",
    "COMMAND_PALETTE_REGISTRY",
]
