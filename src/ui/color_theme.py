"""Color theme system for the Aurelius terminal UI.

Provides named color palettes, ANSI 256-color and truecolor helpers,
theme registry, and a reduced-color mode for accessibility.
Inspired by kimi-cli terminal theming (MoonshotAI/kimi-cli, MIT);
Aurelius-native implementation. License: MIT.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class ColorDepth(Enum):
    NONE = "none"  # no colors (dumb terminal or env override)
    ANSI16 = "ansi16"  # basic 16 ANSI colors
    ANSI256 = "ansi256"  # 256-color xterm
    TRUECOLOR = "truecolor"  # 24-bit RGB


@dataclass(frozen=True)
class ColorSpec:
    """A single color with fallbacks for different color depths."""

    truecolor: tuple[int, int, int]  # RGB (0-255 each)
    ansi256: int  # 0-255
    ansi16: int  # 0-15
    name: str = ""

    def __post_init__(self) -> None:
        r, g, b = self.truecolor
        for v in (r, g, b):
            if not (0 <= v <= 255):
                raise ValueError(f"RGB component {v} out of range [0, 255]")
        if not (0 <= self.ansi256 <= 255):
            raise ValueError(f"ansi256 {self.ansi256} out of range")
        if not (0 <= self.ansi16 <= 15):
            raise ValueError(f"ansi16 {self.ansi16} out of range [0, 15]")


@dataclass
class ColorTheme:
    """A named color palette for the Aurelius UI."""

    name: str
    primary: ColorSpec
    secondary: ColorSpec
    accent: ColorSpec
    success: ColorSpec
    warning: ColorSpec
    error: ColorSpec
    muted: ColorSpec
    background: ColorSpec
    foreground: ColorSpec
    # Reduced-motion / accessibility: flat colors without gradients
    reduced_color: bool = False

    def get(self, role: str) -> ColorSpec:
        """Get a ColorSpec by role name. Raises KeyError on unknown role."""
        roles = {
            "primary": self.primary,
            "secondary": self.secondary,
            "accent": self.accent,
            "success": self.success,
            "warning": self.warning,
            "error": self.error,
            "muted": self.muted,
            "background": self.background,
            "foreground": self.foreground,
        }
        if role not in roles:
            raise KeyError(f"unknown color role: {role!r}")
        return roles[role]

    def role_names(self) -> list[str]:
        return [
            "primary",
            "secondary",
            "accent",
            "success",
            "warning",
            "error",
            "muted",
            "background",
            "foreground",
        ]


class ThemeRenderer:
    """Render text with ANSI color codes given a ColorTheme and detected ColorDepth."""

    def __init__(self, theme: ColorTheme, depth: ColorDepth | None = None) -> None:
        self.theme = theme
        self.depth = depth if depth is not None else self._detect_depth()

    @staticmethod
    def _detect_depth() -> ColorDepth:
        """Detect terminal color support from environment."""
        if os.environ.get("NO_COLOR"):
            return ColorDepth.NONE
        colorterm = os.environ.get("COLORTERM", "").lower()
        if colorterm in ("truecolor", "24bit"):
            return ColorDepth.TRUECOLOR
        term = os.environ.get("TERM", "")
        if "256color" in term:
            return ColorDepth.ANSI256
        if os.environ.get("TERM_PROGRAM") in ("iTerm.app", "WezTerm", "Hyper"):
            return ColorDepth.TRUECOLOR
        if term.startswith("xterm") or term.startswith("screen"):
            return ColorDepth.ANSI16
        return ColorDepth.NONE

    def fg(self, role: str) -> str:
        """Return ANSI escape for foreground color by role."""
        color = self.theme.get(role)
        if self.depth == ColorDepth.TRUECOLOR:
            r, g, b = color.truecolor
            return f"\x1b[38;2;{r};{g};{b}m"
        elif self.depth == ColorDepth.ANSI256:
            return f"\x1b[38;5;{color.ansi256}m"
        elif self.depth == ColorDepth.ANSI16:
            code = 30 + (color.ansi16 % 8) + (60 if color.ansi16 >= 8 else 0)
            return f"\x1b[{code}m"
        return ""

    def reset(self) -> str:
        return "\x1b[0m" if self.depth != ColorDepth.NONE else ""

    def colorize(self, text: str, role: str) -> str:
        """Wrap text with foreground color for the given role."""
        prefix = self.fg(role)
        suffix = self.reset()
        return f"{prefix}{text}{suffix}" if prefix else text

    def strip_ansi(self, text: str) -> str:
        """Remove all ANSI escape sequences from text."""
        import re

        return re.sub(
            r"\x1b\[[0-9;]*[mKHJFABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz]", "", text
        )


# ---------------------------------------------------------------------------
# Built-in themes
# ---------------------------------------------------------------------------

_AURELIUS_DARK = ColorTheme(
    name="aurelius-dark",
    primary=ColorSpec((100, 180, 255), 75, 12, "primary"),
    secondary=ColorSpec((160, 120, 255), 135, 13, "secondary"),
    accent=ColorSpec((255, 200, 80), 220, 11, "accent"),
    success=ColorSpec((80, 220, 120), 77, 10, "success"),
    warning=ColorSpec((255, 160, 40), 214, 3, "warning"),
    error=ColorSpec((255, 80, 80), 196, 9, "error"),
    muted=ColorSpec((120, 120, 140), 245, 8, "muted"),
    background=ColorSpec((20, 22, 30), 235, 0, "background"),
    foreground=ColorSpec((220, 225, 235), 253, 7, "foreground"),
)

_AURELIUS_LIGHT = ColorTheme(
    name="aurelius-light",
    primary=ColorSpec((30, 100, 200), 26, 4, "primary"),
    secondary=ColorSpec((100, 50, 200), 99, 5, "secondary"),
    accent=ColorSpec((180, 100, 0), 130, 3, "accent"),
    success=ColorSpec((0, 140, 60), 28, 2, "success"),
    warning=ColorSpec((160, 80, 0), 130, 3, "warning"),
    error=ColorSpec((180, 0, 0), 160, 1, "error"),
    muted=ColorSpec((100, 100, 120), 102, 8, "muted"),
    background=ColorSpec((248, 248, 252), 255, 7, "background"),
    foreground=ColorSpec((20, 20, 30), 232, 0, "foreground"),
)

_AURELIUS_MONO = ColorTheme(
    name="aurelius-mono",
    primary=ColorSpec((200, 200, 200), 250, 7, "primary"),
    secondary=ColorSpec((160, 160, 160), 247, 7, "secondary"),
    accent=ColorSpec((240, 240, 240), 253, 15, "accent"),
    success=ColorSpec((200, 200, 200), 250, 7, "success"),
    warning=ColorSpec((160, 160, 160), 247, 7, "warning"),
    error=ColorSpec((100, 100, 100), 240, 8, "error"),
    muted=ColorSpec((80, 80, 80), 238, 8, "muted"),
    background=ColorSpec((0, 0, 0), 232, 0, "background"),
    foreground=ColorSpec((220, 220, 220), 252, 7, "foreground"),
    reduced_color=True,
)

COLOR_THEME_REGISTRY: dict[str, ColorTheme] = {
    "aurelius-dark": _AURELIUS_DARK,
    "aurelius-light": _AURELIUS_LIGHT,
    "aurelius-mono": _AURELIUS_MONO,
}

DEFAULT_THEME = _AURELIUS_DARK


def get_theme(name: str) -> ColorTheme:
    """Get a theme by name. Raises KeyError if not found."""
    if name not in COLOR_THEME_REGISTRY:
        raise KeyError(f"unknown theme: {name!r}")
    return COLOR_THEME_REGISTRY[name]


def register_theme(theme: ColorTheme) -> None:
    """Register a custom theme. Raises ValueError if name already registered."""
    if theme.name in COLOR_THEME_REGISTRY:
        raise ValueError(f"theme {theme.name!r} already registered")
    COLOR_THEME_REGISTRY[theme.name] = theme


__all__ = [
    "ColorDepth",
    "ColorSpec",
    "ColorTheme",
    "ThemeRenderer",
    "COLOR_THEME_REGISTRY",
    "DEFAULT_THEME",
    "get_theme",
    "register_theme",
]
