"""Keyboard navigation controller for the Aurelius terminal UI surface.

Inspired by MoonshotAI/kimi-cli (MIT, terminal session lifecycle),
Anthropic Claude Code (MIT, command palette UX), clean-room
reimplementation with original Aurelius branding.

Provides a class-level binding registry and dispatcher so that all
keyboard-first interactions can be declared in one place and rendered
as a help table via Rich.  Only rich, stdlib, and project-local imports
are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table


class KeyBindingError(Exception):
    """Raised when a keyboard-navigation operation fails."""


@dataclass
class KeyBinding:
    """A single keyboard binding.

    Attributes:
        key: The primary key name (e.g. ``"Enter"``, ``"up"``, ``"/"``,
            ``"Escape"``).
        description: Short human-readable description of what the action
            does.
        action: Snake_case action name returned by
            :meth:`KeyboardNav.dispatch` on match.
        modifiers: Zero or more modifier names (e.g. ``["ctrl"]``).
            Modifiers are matched case-insensitively.
    """

    key: str
    description: str
    action: str
    modifiers: list[str] = field(default_factory=list)


class KeyboardNav:
    """Registry and dispatcher for Aurelius keyboard navigation bindings.

    The binding table is class-level so that navigation contexts declared
    at module-import time are available without passing instances around.

    Usage::

        KeyboardNav.register(KeyBinding("ctrl-z", "Undo", "undo"))
        action = KeyboardNav.dispatch("ctrl-z")
    """

    BINDINGS: dict[str, KeyBinding] = {}

    @classmethod
    def register(cls, binding: KeyBinding) -> None:
        """Register *binding* in the navigation table.

        The registry key is a normalised ``"[modifiers+]key"`` string
        (lower-case, sorted modifiers joined with ``+``).

        Args:
            binding: The :class:`KeyBinding` to register.

        Raises:
            KeyBindingError: If *binding* is not a :class:`KeyBinding`
                or if the normalised key combo is already registered.
        """
        if not isinstance(binding, KeyBinding):
            raise KeyBindingError("register() requires a KeyBinding instance")
        registry_key = cls._normalise(binding.key, binding.modifiers)
        if registry_key in cls.BINDINGS:
            raise KeyBindingError(f"key combo {registry_key!r} is already registered")
        cls.BINDINGS[registry_key] = binding

    @classmethod
    def dispatch(cls, key: str, modifiers: list[str] | None = None) -> str | None:
        """Return the action name for *key* + *modifiers*, or ``None``.

        Args:
            key: The pressed key name.
            modifiers: Optional list of active modifiers.

        Returns:
            The ``action`` string of the matching :class:`KeyBinding`,
            or ``None`` if no binding is registered for the combo.
        """
        if modifiers is None:
            modifiers = []
        registry_key = cls._normalise(key, modifiers)
        binding = cls.BINDINGS.get(registry_key)
        if binding is None:
            return None
        return binding.action

    @classmethod
    def render_help(cls, console: Console) -> None:
        """Render a help table of all registered bindings to *console*."""
        table = Table(
            title="Keyboard Navigation",
            show_header=True,
            header_style="bold cyan",
            show_lines=False,
            expand=False,
        )
        table.add_column("Key", style="bold", no_wrap=True)
        table.add_column("Modifiers", style="dim", no_wrap=True)
        table.add_column("Action", style="italic", no_wrap=True)
        table.add_column("Description")

        for binding in sorted(cls.BINDINGS.values(), key=lambda b: b.key):
            table.add_row(
                binding.key,
                "+".join(sorted(m.lower() for m in binding.modifiers)),
                binding.action,
                binding.description,
            )

        console.print(table)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(key: str, modifiers: list[str]) -> str:
        """Return a canonical registry key string for *key* + *modifiers*."""
        norm_mods = sorted(m.lower() for m in modifiers)
        if norm_mods:
            return "+".join(norm_mods) + "+" + key.lower()
        return key.lower()


KEYBOARD_NAV_REGISTRY: dict[str, KeyboardNav] = {}

# ---------------------------------------------------------------------------
# Pre-registered built-in bindings
# ---------------------------------------------------------------------------

_BUILTIN_BINDINGS: list[KeyBinding] = [
    KeyBinding(key="up", description="Move focus up", action="nav_up"),
    KeyBinding(key="down", description="Move focus down", action="nav_down"),
    KeyBinding(key="left", description="Move focus left", action="nav_left"),
    KeyBinding(key="right", description="Move focus right", action="nav_right"),
    KeyBinding(key="Enter", description="Confirm selection", action="confirm"),
    KeyBinding(key="Escape", description="Cancel / close overlay", action="cancel"),
    KeyBinding(key="/", description="Open command palette", action="command_palette"),
    KeyBinding(key="?", description="Show keyboard help", action="help"),
]

for _binding in _BUILTIN_BINDINGS:
    KeyboardNav.register(_binding)

__all__ = [
    "KeyBinding",
    "KeyBindingError",
    "KeyboardNav",
    "KEYBOARD_NAV_REGISTRY",
]
