"""Tests for src/ui/debug_panel.py — DebugPanel, DebugSection, DebugMetric."""

from __future__ import annotations

import pytest
from rich.console import Console

from src.ui.debug_panel import (
    DEBUG_PANEL_REGISTRY,
    DebugMetric,
    DebugPanel,
    DebugPanelError,
    DebugSection,
)

# ---------------------------------------------------------------------------
# DebugMetric defaults
# ---------------------------------------------------------------------------


def test_debug_metric_defaults() -> None:
    m = DebugMetric(name="loss", value=0.5)
    assert m.unit == ""
    assert m.fmt == "{:.4f}"
    assert m.value == 0.5


def test_debug_metric_none_value() -> None:
    m = DebugMetric(name="loss", value=None)
    assert m.value is None


# ---------------------------------------------------------------------------
# DebugSection defaults
# ---------------------------------------------------------------------------


def test_debug_section_defaults() -> None:
    s = DebugSection(title="Test")
    assert s.collapsible is True
    assert s.metrics == []


# ---------------------------------------------------------------------------
# Default DebugPanel sections
# ---------------------------------------------------------------------------


def test_default_panel_has_three_sections() -> None:
    panel = DebugPanel()
    assert len(panel._sections) == 3


def test_default_panel_has_model_section() -> None:
    panel = DebugPanel()
    assert "Model" in panel._sections


def test_default_panel_has_memory_section() -> None:
    panel = DebugPanel()
    assert "Memory" in panel._sections


def test_default_panel_has_attention_section() -> None:
    panel = DebugPanel()
    assert "Attention" in panel._sections


def test_default_panel_model_metrics() -> None:
    panel = DebugPanel()
    names = [m.name for m in panel._sections["Model"].metrics]
    assert "loss" in names
    assert "perplexity" in names
    assert "tokens_per_sec" in names


def test_default_panel_memory_metrics() -> None:
    panel = DebugPanel()
    names = [m.name for m in panel._sections["Memory"].metrics]
    assert "gpu_allocated_gb" in names
    assert "gpu_reserved_gb" in names


def test_default_panel_attention_metrics() -> None:
    panel = DebugPanel()
    names = [m.name for m in panel._sections["Attention"].metrics]
    assert "avg_attn_entropy" in names
    assert "kv_cache_size_mb" in names


# ---------------------------------------------------------------------------
# update_metric
# ---------------------------------------------------------------------------


def test_update_metric_changes_value() -> None:
    panel = DebugPanel()
    panel.update_metric("Model", "loss", 1.2345)
    model_metrics = {m.name: m.value for m in panel._sections["Model"].metrics}
    assert model_metrics["loss"] == pytest.approx(1.2345)


def test_update_metric_to_none() -> None:
    panel = DebugPanel()
    panel.update_metric("Model", "perplexity", 3.14)
    panel.update_metric("Model", "perplexity", None)
    model_metrics = {m.name: m.value for m in panel._sections["Model"].metrics}
    assert model_metrics["perplexity"] is None


def test_update_metric_unknown_section_raises() -> None:
    panel = DebugPanel()
    with pytest.raises(DebugPanelError, match="section"):
        panel.update_metric("Nonexistent", "loss", 0.0)


def test_update_metric_unknown_metric_raises() -> None:
    panel = DebugPanel()
    with pytest.raises(DebugPanelError, match="metric"):
        panel.update_metric("Model", "nonexistent_metric", 0.0)


# ---------------------------------------------------------------------------
# add_section and add_metric
# ---------------------------------------------------------------------------


def test_add_section_registers_new_section() -> None:
    panel = DebugPanel()
    s = DebugSection(title="Custom", metrics=[DebugMetric(name="foo", value=42)])
    panel.add_section(s)
    assert "Custom" in panel._sections


def test_add_section_overwrites_existing() -> None:
    panel = DebugPanel()
    s = DebugSection(title="Model", metrics=[DebugMetric(name="only_loss", value=9.9)])
    panel.add_section(s)
    names = [m.name for m in panel._sections["Model"].metrics]
    assert names == ["only_loss"]


def test_add_metric_to_existing_section() -> None:
    panel = DebugPanel()
    new_metric = DebugMetric(name="grad_norm", value=0.01, unit="", fmt="{:.6f}")
    panel.add_metric("Model", new_metric)
    names = [m.name for m in panel._sections["Model"].metrics]
    assert "grad_norm" in names


def test_add_metric_unknown_section_raises() -> None:
    panel = DebugPanel()
    with pytest.raises(DebugPanelError, match="section"):
        panel.add_metric("Ghost", DebugMetric(name="x", value=0))


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_does_not_crash() -> None:
    panel = DebugPanel()
    console = Console(record=True)
    panel.render(console)
    output = console.export_text()
    assert len(output) > 0


def test_render_collapsed_does_not_crash() -> None:
    panel = DebugPanel()
    console = Console(record=True)
    panel.render(console, collapsed=True)
    output = console.export_text()
    assert len(output) > 0


def test_render_empty_panel_does_not_crash() -> None:
    panel = DebugPanel()
    panel._sections.clear()
    console = Console(record=True)
    panel.render(console)
    output = console.export_text()
    assert "no debug sections" in output


def test_render_shows_metric_name() -> None:
    panel = DebugPanel()
    panel.update_metric("Model", "loss", 0.1234)
    console = Console(record=True)
    panel.render(console)
    output = console.export_text()
    assert "loss" in output


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_returns_dict() -> None:
    panel = DebugPanel()
    result = panel.to_dict()
    assert isinstance(result, dict)


def test_to_dict_contains_sections() -> None:
    panel = DebugPanel()
    result = panel.to_dict()
    assert "Model" in result
    assert "Memory" in result
    assert "Attention" in result


def test_to_dict_contains_metrics() -> None:
    panel = DebugPanel()
    panel.update_metric("Model", "loss", 0.99)
    result = panel.to_dict()
    metrics = result["Model"]["metrics"]
    assert any(m["name"] == "loss" and m["value"] == pytest.approx(0.99) for m in metrics)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_debug_panel_registry_is_dict() -> None:
    assert isinstance(DEBUG_PANEL_REGISTRY, dict)


def test_debug_panel_registry_accepts_entries() -> None:
    DEBUG_PANEL_REGISTRY["test-panel"] = DebugPanel()
    assert "test-panel" in DEBUG_PANEL_REGISTRY
    del DEBUG_PANEL_REGISTRY["test-panel"]
