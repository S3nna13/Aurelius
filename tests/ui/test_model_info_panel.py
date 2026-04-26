"""Tests for src.ui.model_info_panel — ModelInfoPanel, ModelInfoEntry, ModelInfoError."""

from __future__ import annotations

import pytest
from rich.console import Console

from src.ui.model_info_panel import (
    DEFAULT_MODEL_INFO_PANEL,
    MODEL_INFO_PANEL_REGISTRY,
    ModelInfoEntry,
    ModelInfoError,
    ModelInfoPanel,
)

# ---------------------------------------------------------------------------
# DEFAULT_MODEL_INFO_PANEL pre-population
# ---------------------------------------------------------------------------


def test_default_panel_has_manifest_entries() -> None:
    manifest = DEFAULT_MODEL_INFO_PANEL.entries_by_category("manifest")
    assert len(manifest) == 7


def test_default_panel_has_inference_entries() -> None:
    inference = DEFAULT_MODEL_INFO_PANEL.entries_by_category("inference")
    assert len(inference) == 3


def test_default_panel_has_family_name_entry() -> None:
    entry = DEFAULT_MODEL_INFO_PANEL.get("family_name")
    assert entry.key == "family_name"
    assert entry.category == "manifest"


def test_default_panel_has_tokens_per_sec_entry() -> None:
    entry = DEFAULT_MODEL_INFO_PANEL.get("tokens_per_sec")
    assert entry.category == "inference"


def test_default_panel_initial_values_are_none() -> None:
    panel = ModelInfoPanel()
    entry = panel.get("vocab_size")
    assert entry.value is None


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


def test_update_changes_value() -> None:
    panel = ModelInfoPanel()
    panel.update("vocab_size", 32000)
    assert panel.get("vocab_size").value == 32000


def test_update_unknown_key_raises() -> None:
    panel = ModelInfoPanel()
    with pytest.raises(ModelInfoError):
        panel.update("nonexistent_key", 42)


def test_update_to_none() -> None:
    panel = ModelInfoPanel()
    panel.update("family_name", "aurelius-1b")
    panel.update("family_name", None)
    assert panel.get("family_name").value is None


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


def test_get_retrieves_entry() -> None:
    panel = ModelInfoPanel()
    entry = panel.get("backend_name")
    assert isinstance(entry, ModelInfoEntry)
    assert entry.key == "backend_name"


def test_get_unknown_key_raises() -> None:
    panel = ModelInfoPanel()
    with pytest.raises(ModelInfoError):
        panel.get("does_not_exist")


# ---------------------------------------------------------------------------
# entries_by_category
# ---------------------------------------------------------------------------


def test_entries_by_category_manifest() -> None:
    panel = ModelInfoPanel()
    manifest = panel.entries_by_category("manifest")
    keys = {e.key for e in manifest}
    assert "family_name" in keys
    assert "vocab_size" in keys
    assert "n_parameters" in keys


def test_entries_by_category_nonexistent_returns_empty() -> None:
    panel = ModelInfoPanel()
    result = panel.entries_by_category("nonexistent_category")
    assert result == []


def test_entries_by_category_inference() -> None:
    panel = ModelInfoPanel()
    inference = panel.entries_by_category("inference")
    keys = {e.key for e in inference}
    assert "tokens_per_sec" in keys
    assert "latency_p50_ms" in keys
    assert "context_utilization" in keys


# ---------------------------------------------------------------------------
# add_entry
# ---------------------------------------------------------------------------


def test_add_entry_registers_new() -> None:
    panel = ModelInfoPanel()
    panel.add_entry(ModelInfoEntry(key="custom_key", value="hello", category="custom"))
    entry = panel.get("custom_key")
    assert entry.value == "hello"
    assert entry.category == "custom"


def test_add_entry_overwrites_existing() -> None:
    panel = ModelInfoPanel()
    panel.add_entry(ModelInfoEntry(key="family_name", value="overridden", category="manifest"))
    assert panel.get("family_name").value == "overridden"


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_does_not_crash() -> None:
    panel = ModelInfoPanel()
    console = Console(record=True)
    panel.render(console)
    output = console.export_text()
    assert len(output) > 0


def test_render_shows_categories() -> None:
    panel = ModelInfoPanel()
    console = Console(record=True)
    panel.render(console)
    output = console.export_text()
    assert "MANIFEST" in output or "manifest" in output.lower()


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_returns_dict() -> None:
    panel = ModelInfoPanel()
    result = panel.to_dict()
    assert isinstance(result, dict)


def test_to_dict_contains_all_keys() -> None:
    panel = ModelInfoPanel()
    d = panel.to_dict()
    assert "family_name" in d
    assert "tokens_per_sec" in d
    assert d["vocab_size"]["category"] == "manifest"


# ---------------------------------------------------------------------------
# MODEL_INFO_PANEL_REGISTRY
# ---------------------------------------------------------------------------


def test_model_info_panel_registry_is_dict() -> None:
    assert isinstance(MODEL_INFO_PANEL_REGISTRY, dict)


def test_model_info_panel_registry_can_store_panel() -> None:
    panel = ModelInfoPanel()
    MODEL_INFO_PANEL_REGISTRY["test-panel"] = panel
    assert "test-panel" in MODEL_INFO_PANEL_REGISTRY
    del MODEL_INFO_PANEL_REGISTRY["test-panel"]
