"""UI-surface descriptor and registry for the Aurelius terminal UI.

A :class:`UISurface` is a lightweight, immutable description of one
keyboard-first terminal screen: its id, its title, the named panel
regions it draws, which :mod:`~src.ui.panel_layout` layout it defaults
to, which optional :mod:`~src.ui.motion` entry drives its intro, and
its keyboard-shortcut map from key-spec to snake_case action name.

The surface spec deliberately stays data-only: it does not execute
rendering. That keeps surfaces trivially serialisable (e.g. for a
future config loader) and trivially testable.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

from src.ui.errors import UIError

_SURFACE_ID_RE = re.compile(r"^[a-z0-9_\-]+$")
_ACTION_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_MODIFIER_KEY_RE = re.compile(r"^(?:ctrl|alt|shift|meta)-[a-z0-9]$")
_NAMED_KEYS = frozenset(
    {
        "esc",
        "enter",
        "tab",
        "space",
        "backspace",
        "delete",
        "up",
        "down",
        "left",
        "right",
        "home",
        "end",
        "pageup",
        "pagedown",
        "?",
    }
)


def _is_valid_key(k: str) -> bool:
    if not isinstance(k, str) or k == "":
        return False
    if len(k) == 1:
        return k in string.printable and k not in ("\n", "\r", "\t", "\x0b", "\x0c")
    if k in _NAMED_KEYS:
        return True
    if _MODIFIER_KEY_RE.match(k):
        return True
    return False


@dataclass(frozen=True)
class UISurface:
    """Immutable descriptor for one Aurelius UI surface.

    Attributes:
        surface_id: Stable registry key; must match ``[a-z0-9_\\-]+``.
        title: Human-readable surface title. Must be non-empty.
        panels: Non-empty tuple of region names rendered by the
            surface. Each must be a non-empty str.
        default_layout: ``layout_id`` of the default
            :class:`~src.ui.panel_layout.PanelLayout`. Must be a
            non-empty str; existence in the layout registry is not
            enforced at construction time (surfaces may be declared
            before their layouts register).
        motion: Optional motion-registry name for the surface's intro
            animation. ``None`` means no intro motion.
        keyboard_map: Mapping of key spec → action name. Keys are
            printable single characters (e.g. ``"?"``), named keys
            (``"esc"``, ``"enter"``, ``"up"``, …), or modifier combos
            (``"ctrl-k"``, ``"alt-p"``). Values are snake_case action
            names matching ``[a-z][a-z0-9_]*``.
    """

    surface_id: str
    title: str
    panels: tuple[str, ...]
    default_layout: str
    motion: str | None = None
    keyboard_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.surface_id, str) or not self.surface_id:
            raise UIError("surface_id must be a non-empty str")
        if not _SURFACE_ID_RE.match(self.surface_id):
            raise UIError(f"surface_id={self.surface_id!r} must match [a-z0-9_-]+")

        if not isinstance(self.title, str) or self.title == "":
            raise UIError("title must be a non-empty str")

        if not isinstance(self.panels, tuple) or len(self.panels) == 0:
            raise UIError("panels must be a non-empty tuple of str")
        for idx, p in enumerate(self.panels):
            if not isinstance(p, str) or p == "":
                raise UIError(f"panels[{idx}] must be a non-empty str, got {p!r}")

        if not isinstance(self.default_layout, str) or self.default_layout == "":
            raise UIError("default_layout must be a non-empty str")

        if self.motion is not None and (not isinstance(self.motion, str) or self.motion == ""):
            raise UIError("motion must be None or a non-empty str")

        if not isinstance(self.keyboard_map, dict):
            raise UIError("keyboard_map must be a dict[str, str]")
        for k, v in self.keyboard_map.items():
            if not _is_valid_key(k):
                raise UIError(f"keyboard_map key {k!r} is not a recognised key spec")
            if not isinstance(v, str) or v == "":
                raise UIError(f"keyboard_map[{k!r}] must be a non-empty str, got {v!r}")
            if not _ACTION_RE.match(v):
                raise UIError(f"keyboard_map[{k!r}]={v!r} is not a snake_case action name")


UI_SURFACE_REGISTRY: dict[str, UISurface] = {}


def register_ui_surface(surface: UISurface) -> None:
    """Register ``surface`` in :data:`UI_SURFACE_REGISTRY`.

    Raises:
        UIError: If ``surface`` is not a :class:`UISurface` or if its
            ``surface_id`` is already present in the registry.
    """
    if not isinstance(surface, UISurface):
        raise UIError("register_ui_surface requires a UISurface instance")
    if surface.surface_id in UI_SURFACE_REGISTRY:
        raise UIError(f"UI surface {surface.surface_id!r} is already registered")
    UI_SURFACE_REGISTRY[surface.surface_id] = surface


def get_ui_surface(surface_id: str) -> UISurface:
    """Return the registered :class:`UISurface` for ``surface_id``.

    Raises:
        UIError: If ``surface_id`` is not a str or is not registered.
    """
    if not isinstance(surface_id, str):
        raise UIError("surface_id must be a str")
    try:
        return UI_SURFACE_REGISTRY[surface_id]
    except KeyError:
        raise UIError(f"no UI surface registered under {surface_id!r}") from None


def list_ui_surfaces() -> list[str]:
    """Return a sorted list of registered surface ids."""
    return sorted(UI_SURFACE_REGISTRY)


__all__ = [
    "UISurface",
    "UI_SURFACE_REGISTRY",
    "register_ui_surface",
    "get_ui_surface",
    "list_ui_surfaces",
]
