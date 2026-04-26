"""Welcome panel for the Aurelius terminal UI surface.

Renders an original, library-free, keyboard-first welcome banner: a
bordered box containing the wordmark flanked by the Aurelius primary
glyph, an optional owl-shaped ASCII mascot (Marcus), the tagline, an
optional one-shot intro motion placeholder, and an onboarding tip.
Sub-minimum terminals (``cols < 40`` or ``rows < 8``) degrade to a
single-line compact banner rather than truncating the box frame.

All branding strings come from :class:`~src.ui.branding.AureliusBranding`
and are treated as untrusted: control characters are stripped, overly
long lines are truncated, and an all-empty branding object degrades to
the compact banner instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ui.branding import MASCOT_ASCII, AureliusBranding
from src.ui.errors import UIError
from src.ui.motion import MotionSpec, get_motion, play
from src.ui.ui_surface import UISurface, register_ui_surface

_COMPACT_COLS_THRESHOLD = 40
_COMPACT_ROWS_THRESHOLD = 8

_STATIC_FALLBACK_GLYPH = "\u25c8"


@dataclass(frozen=True)
class WelcomePanel:
    """Configuration for :func:`render_welcome`.

    Attributes:
        branding: The :class:`AureliusBranding` token-set to render.
        show_mascot: If True, draw the Aurelius mascot ASCII to the
            left of the tagline.
        show_tagline: If True, render the tagline line.
        show_tip: If True, render the onboarding tip at the bottom.
        reduced_motion: If True, the intro motion returns its reduced
            frame on every render.
        motion: Optional intro :class:`MotionSpec`; ``None`` falls back
            to the primary glyph.
        tip: Onboarding tip string. May be any UTF-8 text; control
            characters are stripped on render.
    """

    branding: AureliusBranding
    show_mascot: bool = True
    show_tagline: bool = True
    show_tip: bool = True
    reduced_motion: bool = False
    motion: MotionSpec | None = None
    tip: str = "press ? for help, ctrl-c to cancel, enter to send"


def _sanitize(raw: str) -> str:
    """Strip control characters (including newlines) from ``raw``."""
    if not isinstance(raw, str):
        return ""
    out_chars = []
    for ch in raw:
        o = ord(ch)
        if o < 0x20 or o == 0x7F:
            continue
        out_chars.append(ch)
    return "".join(out_chars)


def _truncate(s: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(s) <= width:
        return s
    if width == 1:
        return s[:1]
    return s[: width - 1] + "\u2026"


def _compact_banner(branding: AureliusBranding) -> str:
    """Return the short-form banner used on tiny terminals."""
    wm = _sanitize(branding.wordmark) or "AURELIUS"
    glyph = _sanitize(branding.primary_glyph) or _STATIC_FALLBACK_GLYPH
    tagline = _sanitize(branding.tagline) or "the stoic coder's companion"
    return f"{wm} {glyph}  {tagline}"


def _centered(text: str, width: int) -> str:
    """Centre ``text`` in ``width`` columns (truncating if wider)."""
    text = _truncate(text, width)
    if len(text) >= width:
        return text
    pad = width - len(text)
    left = pad // 2
    right = pad - left
    return (" " * left) + text + (" " * right)


def render_welcome(panel: WelcomePanel, *, cols: int = 80, rows: int = 24) -> str:
    """Render ``panel`` into a plain-text multi-line string.

    Args:
        panel: The :class:`WelcomePanel` to render.
        cols: Terminal width in columns. ``cols < 40`` triggers the
            compact single-line banner.
        rows: Terminal height in rows. ``rows < 8`` also triggers the
            compact banner.

    Returns:
        A non-empty string. When ``cols`` and ``rows`` are both at or
        above the thresholds the return value is a multi-line bordered
        box; otherwise it is the compact banner.

    Raises:
        UIError: If ``panel`` is not a :class:`WelcomePanel`, or if
            ``cols`` / ``rows`` are not ints.
    """
    if not isinstance(panel, WelcomePanel):
        raise UIError("panel must be a WelcomePanel instance")
    if not isinstance(cols, int) or isinstance(cols, bool):
        raise UIError("cols must be an int")
    if not isinstance(rows, int) or isinstance(rows, bool):
        raise UIError("rows must be an int")
    if not isinstance(panel.branding, AureliusBranding):
        raise UIError("panel.branding must be an AureliusBranding instance")

    branding = panel.branding

    if cols < _COMPACT_COLS_THRESHOLD or rows < _COMPACT_ROWS_THRESHOLD:
        return _compact_banner(branding)

    wordmark = _sanitize(branding.wordmark)
    tagline = _sanitize(branding.tagline)
    primary_glyph = _sanitize(branding.primary_glyph)

    if not wordmark and not tagline and not primary_glyph:
        return _compact_banner(branding)

    if not primary_glyph:
        primary_glyph = _STATIC_FALLBACK_GLYPH
    if not wordmark:
        wordmark = "AURELIUS"
    if not tagline:
        tagline = ""

    inner_width = max(cols - 2, 1)
    border = "+" + ("-" * inner_width) + "+"

    body_lines: list[str] = []

    wordmark_line = f"{primary_glyph} {wordmark} {primary_glyph}"
    body_lines.append(_centered(wordmark_line, inner_width))

    if panel.show_mascot:
        mascot_rows = [_sanitize(line) for line in MASCOT_ASCII.splitlines() if line is not None]
        tagline_text = tagline if panel.show_tagline else ""
        pair_count = max(len(mascot_rows), 1 if tagline_text else 0)
        pair_count = max(pair_count, 1)
        tag_row_index = pair_count // 2
        for i in range(pair_count):
            m = mascot_rows[i] if i < len(mascot_rows) else ""
            m_col = 10
            m_fixed = _truncate(m, m_col).ljust(m_col)
            if i == tag_row_index and tagline_text:
                right = _truncate(tagline_text, max(inner_width - m_col - 1, 1))
            else:
                right = ""
            full = (m_fixed + " " + right).rstrip()
            body_lines.append(_truncate(full, inner_width).ljust(inner_width))
    else:
        if panel.show_tagline and tagline:
            body_lines.append(_centered(tagline, inner_width))

    if panel.motion is not None:
        if not isinstance(panel.motion, MotionSpec):
            raise UIError("panel.motion must be a MotionSpec or None")
        try:
            motion_frame = play(
                panel.motion,
                reduced_motion=panel.reduced_motion,
                t_ms=0,
            )
        except UIError:
            motion_frame = primary_glyph
    else:
        motion_frame = primary_glyph

    motion_line = _sanitize(motion_frame) or primary_glyph
    body_lines.append(_centered(motion_line, inner_width))

    if panel.show_tip:
        tip_text = _sanitize(panel.tip)
        if tip_text:
            body_lines.append(_centered(tip_text, inner_width))

    available_rows = rows - 2
    if len(body_lines) > available_rows:
        body_lines = body_lines[:available_rows]
    while len(body_lines) < available_rows:
        body_lines.append(" " * inner_width)

    rendered: list[str] = [border]
    for ln in body_lines:
        rendered.append("|" + ln.ljust(inner_width)[:inner_width] + "|")
    rendered.append(border)

    return "\n".join(rendered)


_WELCOME_SURFACE = UISurface(
    surface_id="welcome",
    title="Aurelius Welcome",
    panels=("banner", "mascot", "tagline", "tip"),
    default_layout="stoic-focus",
    motion="welcome-fade",
    keyboard_map={
        "enter": "continue",
        "esc": "quit",
        "?": "help",
    },
)

register_ui_surface(_WELCOME_SURFACE)


def _default_welcome_motion() -> MotionSpec | None:
    try:
        return get_motion("welcome-fade")
    except UIError:
        return None


__all__ = [
    "WelcomePanel",
    "render_welcome",
]
