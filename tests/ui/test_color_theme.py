"""Unit tests for :mod:`src.ui.color_theme`."""

from __future__ import annotations

import pytest

from src.ui.color_theme import (
    _AURELIUS_DARK,
    _AURELIUS_MONO,
    COLOR_THEME_REGISTRY,
    DEFAULT_THEME,
    ColorDepth,
    ColorSpec,
    ColorTheme,
    ThemeRenderer,
    get_theme,
    register_theme,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(r: int = 100, g: int = 150, b: int = 200, a256: int = 74, a16: int = 4) -> ColorSpec:
    return ColorSpec((r, g, b), a256, a16, "test")


def _make_theme(name: str = "test-theme") -> ColorTheme:
    s = _make_spec()
    return ColorTheme(
        name=name,
        primary=s,
        secondary=s,
        accent=s,
        success=s,
        warning=s,
        error=s,
        muted=s,
        background=s,
        foreground=s,
    )


# ---------------------------------------------------------------------------
# ColorSpec validation
# ---------------------------------------------------------------------------


def test_colorspec_valid_construction():
    spec = ColorSpec((10, 20, 30), 42, 6, "mycolor")
    assert spec.truecolor == (10, 20, 30)
    assert spec.ansi256 == 42
    assert spec.ansi16 == 6
    assert spec.name == "mycolor"


def test_colorspec_rgb_r_out_of_range_raises():
    with pytest.raises(ValueError, match=r"RGB component"):
        ColorSpec((256, 0, 0), 0, 0)


def test_colorspec_rgb_g_out_of_range_raises():
    with pytest.raises(ValueError, match=r"RGB component"):
        ColorSpec((0, -1, 0), 0, 0)


def test_colorspec_rgb_b_out_of_range_raises():
    with pytest.raises(ValueError, match=r"RGB component"):
        ColorSpec((0, 0, 300), 0, 0)


def test_colorspec_ansi256_too_high_raises():
    with pytest.raises(ValueError, match=r"ansi256"):
        ColorSpec((0, 0, 0), 256, 0)


def test_colorspec_ansi256_negative_raises():
    with pytest.raises(ValueError, match=r"ansi256"):
        ColorSpec((0, 0, 0), -1, 0)


def test_colorspec_ansi16_too_high_raises():
    with pytest.raises(ValueError, match=r"ansi16"):
        ColorSpec((0, 0, 0), 0, 16)


def test_colorspec_ansi16_negative_raises():
    with pytest.raises(ValueError, match=r"ansi16"):
        ColorSpec((0, 0, 0), 0, -1)


def test_colorspec_boundary_values_valid():
    # Boundary values should not raise
    spec = ColorSpec((0, 255, 128), 255, 15, "boundary")
    assert spec.ansi256 == 255
    assert spec.ansi16 == 15


# ---------------------------------------------------------------------------
# ColorTheme.get
# ---------------------------------------------------------------------------


def test_theme_get_known_role_returns_correct_spec():
    theme = _make_theme()
    spec = _make_spec()
    assert theme.get("primary") == spec


def test_theme_get_all_roles_accessible():
    theme = _make_theme()
    for role in theme.role_names():
        result = theme.get(role)
        assert isinstance(result, ColorSpec)


def test_theme_get_unknown_role_raises_keyerror():
    theme = _make_theme()
    with pytest.raises(KeyError, match=r"unknown color role"):
        theme.get("nonexistent_role")


def test_theme_role_names_returns_all_9():
    theme = _make_theme()
    roles = theme.role_names()
    assert len(roles) == 9
    expected = {
        "primary",
        "secondary",
        "accent",
        "success",
        "warning",
        "error",
        "muted",
        "background",
        "foreground",
    }
    assert set(roles) == expected


# ---------------------------------------------------------------------------
# ThemeRenderer._detect_depth
# ---------------------------------------------------------------------------


def test_detect_depth_no_color_env_returns_none(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    depth = ThemeRenderer._detect_depth()
    assert depth == ColorDepth.NONE


def test_detect_depth_colorterm_truecolor(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.delenv("TERM", raising=False)
    depth = ThemeRenderer._detect_depth()
    assert depth == ColorDepth.TRUECOLOR


def test_detect_depth_colorterm_24bit(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("COLORTERM", "24bit")
    monkeypatch.delenv("TERM", raising=False)
    depth = ThemeRenderer._detect_depth()
    assert depth == ColorDepth.TRUECOLOR


def test_detect_depth_term_256color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    depth = ThemeRenderer._detect_depth()
    assert depth == ColorDepth.ANSI256


def test_detect_depth_term_program_iterm(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.setenv("TERM", "")
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    depth = ThemeRenderer._detect_depth()
    assert depth == ColorDepth.TRUECOLOR


# ---------------------------------------------------------------------------
# ThemeRenderer.fg
# ---------------------------------------------------------------------------


def test_fg_truecolor_escape_sequence():
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.TRUECOLOR)
    # primary spec is (100, 150, 200)
    result = renderer.fg("primary")
    assert result == "\x1b[38;2;100;150;200m"


def test_fg_ansi256_escape_sequence():
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.ANSI256)
    result = renderer.fg("primary")
    assert result == "\x1b[38;5;74m"


def test_fg_ansi16_escape_sequence():
    # ansi16=4 → code = 30 + 4 = 34
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.ANSI16)
    result = renderer.fg("primary")
    assert result == "\x1b[34m"


def test_fg_ansi16_high_color_bright():
    # ansi16=12 → code = 30 + (12 % 8) + 60 = 30 + 4 + 60 = 94
    spec = ColorSpec((100, 180, 255), 75, 12, "bright")
    s = _make_spec()
    theme = ColorTheme(
        name="t",
        primary=spec,
        secondary=s,
        accent=s,
        success=s,
        warning=s,
        error=s,
        muted=s,
        background=s,
        foreground=s,
    )
    renderer = ThemeRenderer(theme, depth=ColorDepth.ANSI16)
    result = renderer.fg("primary")
    assert result == "\x1b[94m"


def test_fg_none_depth_returns_empty_string():
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.NONE)
    result = renderer.fg("primary")
    assert result == ""


# ---------------------------------------------------------------------------
# ThemeRenderer.reset
# ---------------------------------------------------------------------------


def test_reset_none_depth_returns_empty_string():
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.NONE)
    assert renderer.reset() == ""


def test_reset_truecolor_returns_reset_escape():
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.TRUECOLOR)
    assert renderer.reset() == "\x1b[0m"


def test_reset_ansi256_returns_reset_escape():
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.ANSI256)
    assert renderer.reset() == "\x1b[0m"


def test_reset_ansi16_returns_reset_escape():
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.ANSI16)
    assert renderer.reset() == "\x1b[0m"


# ---------------------------------------------------------------------------
# ThemeRenderer.colorize
# ---------------------------------------------------------------------------


def test_colorize_truecolor_wraps_text():
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.TRUECOLOR)
    result = renderer.colorize("hello", "primary")
    assert result.startswith("\x1b[38;2;")
    assert "hello" in result
    assert result.endswith("\x1b[0m")


def test_colorize_none_depth_returns_unmodified_text():
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.NONE)
    result = renderer.colorize("plain text", "primary")
    assert result == "plain text"


def test_colorize_empty_string_colored_depth():
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.TRUECOLOR)
    result = renderer.colorize("", "primary")
    # Should be prefix + "" + suffix
    assert result == renderer.fg("primary") + renderer.reset()


def test_colorize_empty_string_none_depth():
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.NONE)
    result = renderer.colorize("", "primary")
    assert result == ""


# ---------------------------------------------------------------------------
# ThemeRenderer.strip_ansi
# ---------------------------------------------------------------------------


def test_strip_ansi_removes_escape_sequences():
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.TRUECOLOR)
    colored = renderer.colorize("hello world", "primary")
    stripped = renderer.strip_ansi(colored)
    assert stripped == "hello world"


def test_strip_ansi_noop_on_plain_text():
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.NONE)
    plain = "no escape codes here"
    assert renderer.strip_ansi(plain) == plain


# ---------------------------------------------------------------------------
# Registry and built-in themes
# ---------------------------------------------------------------------------


def test_color_theme_registry_has_all_3_builtin_themes():
    for name in ("aurelius-dark", "aurelius-light", "aurelius-mono"):
        assert name in COLOR_THEME_REGISTRY


def test_get_theme_known_name_returns_correct_theme():
    theme = get_theme("aurelius-dark")
    assert theme.name == "aurelius-dark"


def test_get_theme_light_returns_light_theme():
    theme = get_theme("aurelius-light")
    assert theme.name == "aurelius-light"


def test_get_theme_unknown_raises_keyerror():
    with pytest.raises(KeyError, match=r"unknown theme"):
        get_theme("does-not-exist")


def test_register_theme_new_theme_succeeds():
    theme = _make_theme("test-register-new")
    register_theme(theme)
    assert "test-register-new" in COLOR_THEME_REGISTRY
    # Clean up
    del COLOR_THEME_REGISTRY["test-register-new"]


def test_register_theme_duplicate_raises_valueerror():
    with pytest.raises(ValueError, match=r"already registered"):
        register_theme(_AURELIUS_DARK)


def test_default_theme_is_aurelius_dark():
    assert DEFAULT_THEME is _AURELIUS_DARK
    assert DEFAULT_THEME.name == "aurelius-dark"


# ---------------------------------------------------------------------------
# Accessibility / reduced color
# ---------------------------------------------------------------------------


def test_aurelius_mono_has_reduced_color_true():
    assert _AURELIUS_MONO.reduced_color is True


def test_aurelius_dark_not_reduced_color():
    assert _AURELIUS_DARK.reduced_color is False


def test_none_depth_never_outputs_escape_codes():
    """NONE depth must never emit any ANSI escape sequences (accessibility)."""
    theme = _make_theme()
    renderer = ThemeRenderer(theme, depth=ColorDepth.NONE)
    for role in theme.role_names():
        assert renderer.fg(role) == ""
        assert renderer.colorize("text", role) == "text"
    assert renderer.reset() == ""
