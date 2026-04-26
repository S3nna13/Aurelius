"""Integration tests for the Aurelius UI surface.

Verifies that the three UI registries are populated, the welcome
surface is wired up end-to-end, and that ``render_welcome`` with the
default branding plus the ``welcome-fade`` motion returns original
Aurelius copy containing no third-party vendor references.
"""

from __future__ import annotations


def test_ui_package_exports_three_registries():
    from src.ui import (
        MOTION_REGISTRY,
        PANEL_LAYOUT_REGISTRY,
        UI_SURFACE_REGISTRY,
    )

    assert isinstance(MOTION_REGISTRY, dict)
    assert isinstance(PANEL_LAYOUT_REGISTRY, dict)
    assert isinstance(UI_SURFACE_REGISTRY, dict)


def test_all_three_registries_non_empty():
    from src.ui import (
        MOTION_REGISTRY,
        PANEL_LAYOUT_REGISTRY,
        UI_SURFACE_REGISTRY,
    )

    assert len(MOTION_REGISTRY) >= 1
    assert len(PANEL_LAYOUT_REGISTRY) >= 1
    assert len(UI_SURFACE_REGISTRY) >= 1


def test_render_welcome_end_to_end_original_branding():
    from src.ui import (
        DEFAULT_BRANDING,
        WelcomePanel,
        get_motion,
        render_welcome,
    )

    panel = WelcomePanel(
        branding=DEFAULT_BRANDING,
        motion=get_motion("welcome-fade"),
    )
    out = render_welcome(panel)
    assert isinstance(out, str) and out != ""
    assert "AURELIUS" in out
    for vendor in (
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
    ):
        assert vendor not in out, f"vendor {vendor!r} leaked into UI copy"


def test_welcome_is_a_registered_ui_surface():
    from src.ui import UI_SURFACE_REGISTRY

    assert "welcome" in UI_SURFACE_REGISTRY
    surf = UI_SURFACE_REGISTRY["welcome"]
    assert surf.default_layout == "stoic-focus"
    assert surf.motion == "welcome-fade"


def test_pre_registered_panel_layouts_present():
    from src.ui import PANEL_LAYOUT_REGISTRY

    assert "stoic-3pane" in PANEL_LAYOUT_REGISTRY
    assert "stoic-focus" in PANEL_LAYOUT_REGISTRY
    three = PANEL_LAYOUT_REGISTRY["stoic-3pane"]
    focus = PANEL_LAYOUT_REGISTRY["stoic-focus"]
    assert three.regions == ("header", "transcript", "status", "footer")
    assert focus.regions == ("header", "transcript", "footer")


def test_pre_registered_motions_present():
    from src.ui import MOTION_REGISTRY

    for name in ("stoic-cursor", "thinking-dots", "welcome-fade"):
        assert name in MOTION_REGISTRY, f"missing motion {name!r}"


def test_ui_surface_imports_compose_and_play_available():
    from src.ui import compose_layout, get_motion, get_panel_layout, play

    layout = get_panel_layout("stoic-focus")
    out = compose_layout(
        layout,
        cols=layout.min_cols,
        rows=layout.min_rows,
        panels={"header": "AURELIUS", "transcript": "hello", "footer": "ready"},
    )
    assert isinstance(out, str) and out != ""

    motion = get_motion("stoic-cursor")
    frame = play(motion, t_ms=0)
    assert isinstance(frame, str) and frame != ""
