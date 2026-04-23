"""Unit tests for :mod:`src.ui.welcome`."""

from __future__ import annotations

import pytest

from src.ui import UI_SURFACE_REGISTRY
from src.ui.branding import DEFAULT_BRANDING, AureliusBranding
from src.ui.errors import UIError
from src.ui.motion import get_motion
from src.ui.welcome import WelcomePanel, render_welcome


_VENDOR_STRINGS = (
    "Claude",
    "Anthropic",
    "Cursor",
    "Codex",
    "OpenAI",
    "GPT",
    "Llama",
    "Mistral",
    "DeepSeek",
    "Qwen",
)


def _default_panel(**overrides) -> WelcomePanel:
    base: dict = {
        "branding": DEFAULT_BRANDING,
        "motion": get_motion("welcome-fade"),
    }
    base.update(overrides)
    return WelcomePanel(**base)


def test_default_render_contains_wordmark():
    out = render_welcome(_default_panel())
    assert isinstance(out, str)
    assert out != ""
    assert "AURELIUS" in out


def test_default_render_contains_tagline():
    out = render_welcome(_default_panel())
    assert "the stoic coder's companion" in out


def test_compact_banner_on_narrow_cols():
    out = render_welcome(_default_panel(), cols=30, rows=24)
    assert "\n" not in out
    assert "AURELIUS" in out
    assert "the stoic coder's companion" in out


def test_compact_banner_on_short_rows():
    out = render_welcome(_default_panel(), rows=4, cols=80)
    assert "\n" not in out
    assert "AURELIUS" in out


def test_show_mascot_false_hides_mascot_ascii():
    with_mascot = render_welcome(_default_panel(show_mascot=True))
    without_mascot = render_welcome(_default_panel(show_mascot=False))
    assert "(o,o)" in with_mascot
    assert "(o,o)" not in without_mascot


def test_show_tip_false_hides_tip():
    with_tip = render_welcome(_default_panel(show_tip=True))
    without_tip = render_welcome(_default_panel(show_tip=False))
    assert "press ? for help" in with_tip
    assert "press ? for help" not in without_tip


def test_reduced_motion_uses_reduced_frame():
    spec = get_motion("welcome-fade")
    assert spec.reduced_motion_frame
    panel = _default_panel(motion=spec, reduced_motion=True)
    out = render_welcome(panel)
    assert spec.reduced_motion_frame in out


def test_motion_none_does_not_crash():
    panel = _default_panel(motion=None)
    out = render_welcome(panel)
    assert isinstance(out, str) and out != ""
    assert "AURELIUS" in out


@pytest.mark.parametrize("vendor", _VENDOR_STRINGS)
def test_no_third_party_vendor_strings_appear(vendor):
    for panel in (
        _default_panel(),
        _default_panel(show_mascot=False),
        _default_panel(show_tip=False),
        _default_panel(reduced_motion=True),
        _default_panel(motion=None),
    ):
        out = render_welcome(panel)
        assert vendor not in out, (
            f"vendor string {vendor!r} leaked into welcome output"
        )
        compact = render_welcome(panel, cols=30, rows=4)
        assert vendor not in compact


def test_empty_branding_does_not_crash():
    empty = AureliusBranding(
        product_name="",
        wordmark="",
        tagline="",
        mascot_name="",
        primary_glyph="",
        secondary_glyph="",
    )
    panel = WelcomePanel(branding=empty, motion=None)
    out = render_welcome(panel)
    assert isinstance(out, str) and out != ""


def test_returned_string_has_content():
    out = render_welcome(_default_panel())
    assert len(out) > 0
    assert "\n" in out


def test_welcome_surface_keyboard_map_has_enter_continue():
    surf = UI_SURFACE_REGISTRY["welcome"]
    assert surf.keyboard_map["enter"] == "continue"
    assert surf.keyboard_map["esc"] == "quit"
    assert surf.keyboard_map["?"] == "help"


def test_welcome_registered_in_ui_surface_registry():
    assert "welcome" in UI_SURFACE_REGISTRY


def test_render_welcome_rejects_non_panel():
    with pytest.raises(UIError):
        render_welcome("not a panel")  # type: ignore[arg-type]


def test_render_welcome_rejects_bad_cols_type():
    with pytest.raises(UIError):
        render_welcome(_default_panel(), cols="80")  # type: ignore[arg-type]


def test_render_welcome_sanitizes_control_chars_in_tip():
    panel = _default_panel(tip="hi\x00there\x07bell")
    out = render_welcome(panel)
    assert "\x00" not in out
    assert "\x07" not in out
    assert "hithere" in out or "hitherebell" in out


def test_render_welcome_does_not_contain_vendor_case_insensitive_for_claude():
    out = render_welcome(_default_panel())
    assert "claude" not in out.lower()
    assert "anthropic" not in out.lower()
    assert "cursor" not in out.lower()
    assert "codex" not in out.lower()
