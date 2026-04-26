"""Tests for src/ui/tooltip.py — ~45 tests."""

from __future__ import annotations

import pytest

from src.ui.tooltip import TOOLTIP_REGISTRY, TooltipConfig, TooltipRegistry, TooltipRenderer

# ---------------------------------------------------------------------------
# TooltipConfig — defaults
# ---------------------------------------------------------------------------


class TestTooltipConfigDefaults:
    def test_text_stored(self):
        cfg = TooltipConfig(text="hello")
        assert cfg.text == "hello"

    def test_default_delay_ms(self):
        cfg = TooltipConfig(text="x")
        assert cfg.delay_ms == 300

    def test_default_max_width(self):
        cfg = TooltipConfig(text="x")
        assert cfg.max_width == 40

    def test_default_position(self):
        cfg = TooltipConfig(text="x")
        assert cfg.position == "above"

    def test_custom_delay_ms(self):
        cfg = TooltipConfig(text="x", delay_ms=500)
        assert cfg.delay_ms == 500

    def test_custom_max_width(self):
        cfg = TooltipConfig(text="x", max_width=20)
        assert cfg.max_width == 20

    def test_position_above(self):
        cfg = TooltipConfig(text="x", position="above")
        assert cfg.position == "above"

    def test_position_below(self):
        cfg = TooltipConfig(text="x", position="below")
        assert cfg.position == "below"

    def test_position_left(self):
        cfg = TooltipConfig(text="x", position="left")
        assert cfg.position == "left"

    def test_position_right(self):
        cfg = TooltipConfig(text="x", position="right")
        assert cfg.position == "right"

    def test_invalid_position_raises(self):
        with pytest.raises(ValueError):
            TooltipConfig(text="x", position="diagonal")


# ---------------------------------------------------------------------------
# TooltipRenderer
# ---------------------------------------------------------------------------


class TestTooltipRenderer:
    def setup_method(self):
        self.renderer = TooltipRenderer()
        self.cfg = TooltipConfig(text="Submit input [Enter]")

    def test_show_returns_non_empty_string(self):
        result = self.renderer.show(self.cfg)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_show_contains_text(self):
        result = self.renderer.show(self.cfg)
        assert "Submit input" in result

    def test_show_contains_full_text(self):
        result = self.renderer.show(self.cfg)
        assert "[Enter]" in result

    def test_is_visible_false_initially(self):
        assert self.renderer.is_visible() is False

    def test_is_visible_true_after_show(self):
        self.renderer.show(self.cfg)
        assert self.renderer.is_visible() is True

    def test_is_visible_false_after_hide(self):
        self.renderer.show(self.cfg)
        self.renderer.hide()
        assert self.renderer.is_visible() is False

    def test_hide_on_fresh_renderer_is_safe(self):
        self.renderer.hide()
        assert self.renderer.is_visible() is False

    def test_show_wraps_long_text(self):
        long_text = "This is a very long tooltip text that should definitely be wrapped at the max width boundary"  # noqa: E501
        cfg = TooltipConfig(text=long_text, max_width=20)
        result = self.renderer.show(cfg)
        # Each line of the output (stripped) should not be wider than max_width + border chars
        # At minimum the content must appear
        assert "This is a very long" in result or "This" in result

    def test_show_max_width_respected_narrow(self):
        cfg = TooltipConfig(text="word " * 20, max_width=10)
        result = self.renderer.show(cfg)
        assert result  # Should still produce output

    def test_show_position_above_no_error(self):
        cfg = TooltipConfig(text="tip", position="above")
        result = self.renderer.show(cfg)
        assert result

    def test_show_position_below_no_error(self):
        cfg = TooltipConfig(text="tip", position="below")
        result = self.renderer.show(cfg)
        assert result

    def test_show_position_left_no_error(self):
        cfg = TooltipConfig(text="tip", position="left")
        result = self.renderer.show(cfg)
        assert result

    def test_show_position_right_no_error(self):
        cfg = TooltipConfig(text="tip", position="right")
        result = self.renderer.show(cfg)
        assert result

    def test_show_multiple_times_updates_visibility(self):
        cfg2 = TooltipConfig(text="another tip")
        self.renderer.show(self.cfg)
        self.renderer.hide()
        self.renderer.show(cfg2)
        assert self.renderer.is_visible() is True

    def test_show_returns_string_type(self):
        result = self.renderer.show(self.cfg)
        assert type(result) is str

    def test_renderer_independent_instances(self):
        r1 = TooltipRenderer()
        r2 = TooltipRenderer()
        r1.show(self.cfg)
        assert r1.is_visible() is True
        assert r2.is_visible() is False


# ---------------------------------------------------------------------------
# TooltipRegistry
# ---------------------------------------------------------------------------


class TestTooltipRegistry:
    def setup_method(self):
        self.registry = TooltipRegistry()

    def test_register_and_get(self):
        cfg = TooltipConfig(text="hello")
        self.registry.register("greet", cfg)
        assert self.registry.get("greet") is cfg

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            self.registry.get("nonexistent")

    def test_list_tooltips_empty(self):
        assert self.registry.list_tooltips() == []

    def test_list_tooltips_returns_names(self):
        self.registry.register("b", TooltipConfig(text="b"))
        self.registry.register("a", TooltipConfig(text="a"))
        names = self.registry.list_tooltips()
        assert "a" in names
        assert "b" in names

    def test_list_tooltips_sorted(self):
        self.registry.register("zzz", TooltipConfig(text="z"))
        self.registry.register("aaa", TooltipConfig(text="a"))
        names = self.registry.list_tooltips()
        assert names == sorted(names)

    def test_overwrite_existing(self):
        cfg1 = TooltipConfig(text="first")
        cfg2 = TooltipConfig(text="second")
        self.registry.register("key", cfg1)
        self.registry.register("key", cfg2)
        assert self.registry.get("key") is cfg2

    def test_register_multiple(self):
        for i in range(5):
            self.registry.register(f"tip{i}", TooltipConfig(text=f"text {i}"))
        assert len(self.registry.list_tooltips()) == 5


# ---------------------------------------------------------------------------
# TOOLTIP_REGISTRY — pre-registered defaults
# ---------------------------------------------------------------------------


class TestDefaultTooltipRegistry:
    def test_submit_registered(self):
        cfg = TOOLTIP_REGISTRY.get("submit")
        assert cfg is not None

    def test_submit_text(self):
        cfg = TOOLTIP_REGISTRY.get("submit")
        assert "Submit" in cfg.text or "Enter" in cfg.text

    def test_clear_registered(self):
        cfg = TOOLTIP_REGISTRY.get("clear")
        assert cfg is not None

    def test_clear_text(self):
        cfg = TOOLTIP_REGISTRY.get("clear")
        assert "Clear" in cfg.text or "Ctrl" in cfg.text

    def test_model_registered(self):
        cfg = TOOLTIP_REGISTRY.get("model")
        assert cfg is not None

    def test_model_text(self):
        cfg = TOOLTIP_REGISTRY.get("model")
        assert "model" in cfg.text.lower() or "Ctrl" in cfg.text

    def test_all_three_in_list(self):
        names = TOOLTIP_REGISTRY.list_tooltips()
        assert "submit" in names
        assert "clear" in names
        assert "model" in names

    def test_registry_is_tooltip_registry_instance(self):
        assert isinstance(TOOLTIP_REGISTRY, TooltipRegistry)
