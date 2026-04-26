"""Panel-layout primitive for the Aurelius terminal UI surface.

A :class:`PanelLayout` describes the named regions of a multi-pane
terminal view (e.g. ``header``, ``transcript``, ``status``, ``footer``),
the minimum terminal size the layout needs, and a keyboard-focus order
for those regions. The :func:`compose_layout` helper takes a layout, a
concrete terminal size in columns/rows, and a mapping from region name
to pre-rendered panel text, and returns a plain-ASCII composition using
only the characters ``+``, ``-`` and ``|`` for borders.

The module is pure stdlib and side-effect free at import apart from
registering two original built-in layouts (``stoic-3pane`` and
``stoic-focus``) in :data:`PANEL_LAYOUT_REGISTRY`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ui.errors import UIError

_ID_RE = re.compile(r"^[a-z0-9_\-]+$")

_MIN_COLS_HARD_FLOOR = 20
_MIN_ROWS_HARD_FLOOR = 6


@dataclass(frozen=True)
class PanelLayout:
    """Named-region layout spec for a terminal pane arrangement.

    Attributes:
        layout_id: Stable registry key; must match ``[a-z0-9_\\-]+``.
        regions: Ordered tuple of unique region names rendered top-to
            bottom. Must be non-empty.
        min_cols: Minimum columns the layout can render at without
            truncation. Must be >= 20.
        min_rows: Minimum rows the layout can render at without
            truncation. Must be >= 6.
        focus_order: A permutation of ``regions`` declaring keyboard
            focus order. Must contain each region exactly once.
    """

    layout_id: str
    regions: tuple[str, ...]
    min_cols: int
    min_rows: int
    focus_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise UIError("layout_id must be a non-empty str")
        if not _ID_RE.match(self.layout_id):
            raise UIError(f"layout_id={self.layout_id!r} must match [a-z0-9_-]+")

        if not isinstance(self.regions, tuple) or len(self.regions) == 0:
            raise UIError("regions must be a non-empty tuple of str")
        for idx, r in enumerate(self.regions):
            if not isinstance(r, str) or not r:
                raise UIError(f"regions[{idx}] must be a non-empty str, got {r!r}")
        if len(set(self.regions)) != len(self.regions):
            raise UIError(f"regions must be unique, got duplicates in {self.regions!r}")

        if not isinstance(self.min_cols, int) or isinstance(self.min_cols, bool):
            raise UIError("min_cols must be an int")
        if not isinstance(self.min_rows, int) or isinstance(self.min_rows, bool):
            raise UIError("min_rows must be an int")
        if self.min_cols < _MIN_COLS_HARD_FLOOR:
            raise UIError(f"min_cols={self.min_cols} below hard floor {_MIN_COLS_HARD_FLOOR}")
        if self.min_rows < _MIN_ROWS_HARD_FLOOR:
            raise UIError(f"min_rows={self.min_rows} below hard floor {_MIN_ROWS_HARD_FLOOR}")

        if not isinstance(self.focus_order, tuple):
            raise UIError("focus_order must be a tuple of str")
        if sorted(self.focus_order) != sorted(self.regions):
            raise UIError(
                f"focus_order={self.focus_order!r} is not a permutation of regions={self.regions!r}"
            )


def _sanitize_line(raw: str, width: int) -> str:
    """Strip control characters and truncate to ``width`` columns."""
    cleaned_chars = []
    for ch in raw:
        o = ord(ch)
        if ch == "\t":
            cleaned_chars.append("    ")
        elif o < 0x20 or o == 0x7F:
            continue
        else:
            cleaned_chars.append(ch)
    cleaned = "".join(cleaned_chars)
    if len(cleaned) > width:
        if width <= 1:
            return cleaned[:width]
        return cleaned[: width - 1] + "\u2026"
    return cleaned


def _region_body(content: str, inner_cols: int, inner_rows: int) -> list[str]:
    """Render panel ``content`` into exactly ``inner_rows`` lines of
    ``inner_cols`` columns each. Overlong lines are truncated; short
    inputs are padded with blank lines.
    """
    if inner_cols <= 0 or inner_rows <= 0:
        return []
    raw_lines = content.splitlines() if content else []
    body: list[str] = []
    for raw in raw_lines[:inner_rows]:
        body.append(_sanitize_line(raw, inner_cols).ljust(inner_cols))
    while len(body) < inner_rows:
        body.append(" " * inner_cols)
    return body


def compose_layout(
    layout: PanelLayout,
    cols: int,
    rows: int,
    *,
    panels: dict[str, str],
) -> str:
    """Compose ``layout`` into a plain-text string sized ``cols`` × ``rows``.

    Each region is stacked vertically in declaration order, boxed with
    ASCII borders, and receives a proportional share of the available
    rows. A region missing from ``panels`` renders as an empty box.

    Args:
        layout: The :class:`PanelLayout` to render.
        cols: Actual terminal width in columns.
        rows: Actual terminal height in rows.
        panels: Mapping of region name → pre-rendered panel text. Keys
            not in ``layout.regions`` are ignored. Values are treated
            as untrusted text: control characters are stripped and
            overlong lines truncated.

    Returns:
        A newline-joined string with exactly ``rows`` lines, each
        containing exactly ``cols`` characters.

    Raises:
        UIError: If ``cols < layout.min_cols`` or ``rows < layout.min_rows``,
            or if any argument has the wrong type.
    """
    if not isinstance(layout, PanelLayout):
        raise UIError("layout must be a PanelLayout instance")
    if not isinstance(cols, int) or isinstance(cols, bool):
        raise UIError("cols must be an int")
    if not isinstance(rows, int) or isinstance(rows, bool):
        raise UIError("rows must be an int")
    if not isinstance(panels, dict):
        raise UIError("panels must be a dict[str, str]")

    if cols < layout.min_cols:
        raise UIError(
            f"terminal width cols={cols} is below layout min_cols="
            f"{layout.min_cols} for layout {layout.layout_id!r}"
        )
    if rows < layout.min_rows:
        raise UIError(
            f"terminal height rows={rows} is below layout min_rows="
            f"{layout.min_rows} for layout {layout.layout_id!r}"
        )

    n_regions = len(layout.regions)

    total_border_rows = n_regions + 1
    body_budget = rows - total_border_rows
    if body_budget < n_regions:
        raise UIError(
            f"rows={rows} leaves no room for {n_regions} region bodies in "
            f"layout {layout.layout_id!r}"
        )

    base_share = body_budget // n_regions
    extra = body_budget - base_share * n_regions
    per_region_rows = [base_share + (1 if i < extra else 0) for i in range(n_regions)]

    inner_cols = cols - 2
    if inner_cols <= 0:
        raise UIError(f"cols={cols} too small to render borders for layout {layout.layout_id!r}")

    border = "+" + ("-" * inner_cols) + "+"

    lines: list[str] = []
    for idx, region in enumerate(layout.regions):
        content = panels.get(region, "")
        if not isinstance(content, str):
            raise UIError(f"panels[{region!r}] must be str, got {type(content).__name__}")
        header_label = _sanitize_line(region, inner_cols)
        header_label = header_label.ljust(inner_cols)
        lines.append(border)
        body_rows = per_region_rows[idx]
        if body_rows >= 1:
            lines.append("|" + header_label + "|")
            remaining = body_rows - 1
        else:
            remaining = 0
        if remaining > 0:
            for body_line in _region_body(content, inner_cols, remaining):
                lines.append("|" + body_line + "|")
    lines.append(border)

    while len(lines) < rows:
        lines.append(" " * cols)
    if len(lines) > rows:
        lines = lines[:rows]

    padded = [ln.ljust(cols)[:cols] for ln in lines]
    return "\n".join(padded)


PANEL_LAYOUT_REGISTRY: dict[str, PanelLayout] = {}


def register_panel_layout(layout: PanelLayout) -> None:
    """Register ``layout`` in :data:`PANEL_LAYOUT_REGISTRY`.

    Raises:
        UIError: If ``layout`` is not a :class:`PanelLayout` or if its
            ``layout_id`` is already present in the registry.
    """
    if not isinstance(layout, PanelLayout):
        raise UIError("register_panel_layout requires a PanelLayout instance")
    if layout.layout_id in PANEL_LAYOUT_REGISTRY:
        raise UIError(f"panel layout {layout.layout_id!r} is already registered")
    PANEL_LAYOUT_REGISTRY[layout.layout_id] = layout


def get_panel_layout(layout_id: str) -> PanelLayout:
    """Return the registered :class:`PanelLayout` for ``layout_id``.

    Raises:
        UIError: If ``layout_id`` is not a str or is not registered.
    """
    if not isinstance(layout_id, str):
        raise UIError("layout_id must be a str")
    try:
        return PANEL_LAYOUT_REGISTRY[layout_id]
    except KeyError:
        raise UIError(f"no panel layout registered under {layout_id!r}") from None


def list_panel_layouts() -> list[str]:
    """Return a sorted list of registered panel-layout ids."""
    return sorted(PANEL_LAYOUT_REGISTRY)


_STOIC_3PANE = PanelLayout(
    layout_id="stoic-3pane",
    regions=("header", "transcript", "status", "footer"),
    min_cols=60,
    min_rows=20,
    focus_order=("transcript", "status", "header", "footer"),
)

_STOIC_FOCUS = PanelLayout(
    layout_id="stoic-focus",
    regions=("header", "transcript", "footer"),
    min_cols=40,
    min_rows=12,
    focus_order=("transcript", "header", "footer"),
)

register_panel_layout(_STOIC_3PANE)
register_panel_layout(_STOIC_FOCUS)


__all__ = [
    "PanelLayout",
    "PANEL_LAYOUT_REGISTRY",
    "compose_layout",
    "register_panel_layout",
    "get_panel_layout",
    "list_panel_layouts",
]
