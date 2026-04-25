from __future__ import annotations

import pytest

from src.backends.model_presets import (
    MODEL_PRESET_REGISTRY,
    ModelPreset,
    ModelPresetError,
    ModelPresetRegistry,
)


@pytest.fixture()
def registry() -> ModelPresetRegistry:
    reg = ModelPresetRegistry()
    return reg


def _make_preset(model_id: str = "test-model", **kwargs) -> ModelPreset:
    defaults = dict(temperature=0.7, max_tokens=512)
    defaults.update(kwargs)
    return ModelPreset(model_id=model_id, **defaults)


# ---------------------------------------------------------------------------
# ModelPreset validation
# ---------------------------------------------------------------------------


def test_preset_valid_construction() -> None:
    p = _make_preset("my-model", temperature=0.5, max_tokens=1024, top_p=0.9)
    assert p.model_id == "my-model"
    assert p.temperature == 0.5
    assert p.top_p == 0.9


def test_preset_empty_model_id_raises() -> None:
    with pytest.raises(ModelPresetError, match="model_id"):
        ModelPreset(model_id="", temperature=0.5, max_tokens=256)


def test_preset_temperature_out_of_range_raises() -> None:
    with pytest.raises(ModelPresetError, match="temperature"):
        ModelPreset(model_id="x", temperature=3.0, max_tokens=256)


def test_preset_max_tokens_zero_raises() -> None:
    with pytest.raises(ModelPresetError, match="max_tokens"):
        ModelPreset(model_id="x", temperature=0.5, max_tokens=0)


def test_preset_top_p_zero_raises() -> None:
    with pytest.raises(ModelPresetError, match="top_p"):
        ModelPreset(model_id="x", temperature=0.5, max_tokens=256, top_p=0.0)


# ---------------------------------------------------------------------------
# Registry – register / get
# ---------------------------------------------------------------------------


def test_register_and_get_roundtrip(registry: ModelPresetRegistry) -> None:
    preset = _make_preset("alpha")
    registry.register(preset)
    result = registry.get("alpha")
    assert result is preset


def test_get_missing_returns_none(registry: ModelPresetRegistry) -> None:
    assert registry.get("nonexistent") is None


def test_register_non_preset_raises(registry: ModelPresetRegistry) -> None:
    with pytest.raises(ModelPresetError, match="ModelPreset"):
        registry.register({"model_id": "bad"})  # type: ignore[arg-type]


def test_register_overwrites_same_model_id(registry: ModelPresetRegistry) -> None:
    p1 = _make_preset("dup", temperature=0.5)
    p2 = _make_preset("dup", temperature=0.9)
    registry.register(p1)
    registry.register(p2)
    assert registry.get("dup").temperature == 0.9


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


def test_list_models_sorted(registry: ModelPresetRegistry) -> None:
    registry.register(_make_preset("zeta"))
    registry.register(_make_preset("alpha"))
    models = registry.list_models()
    assert models == sorted(models)
    assert "alpha" in models and "zeta" in models


def test_list_models_empty(registry: ModelPresetRegistry) -> None:
    assert registry.list_models() == []


# ---------------------------------------------------------------------------
# apply_preset
# ---------------------------------------------------------------------------


def test_apply_preset_no_overrides(registry: ModelPresetRegistry) -> None:
    registry.register(_make_preset("base", temperature=0.6, max_tokens=128))
    result = registry.apply_preset("base", {})
    assert result["temperature"] == 0.6
    assert result["max_tokens"] == 128


def test_apply_preset_with_overrides(registry: ModelPresetRegistry) -> None:
    registry.register(_make_preset("base", temperature=0.6, max_tokens=128))
    result = registry.apply_preset("base", {"temperature": 0.2, "max_tokens": 64})
    assert result["temperature"] == 0.2
    assert result["max_tokens"] == 64


def test_apply_preset_missing_model_raises(registry: ModelPresetRegistry) -> None:
    with pytest.raises(ModelPresetError, match="No preset registered"):
        registry.apply_preset("no-such-model", {})


# ---------------------------------------------------------------------------
# Module-level MODEL_PRESET_REGISTRY (built-ins)
# ---------------------------------------------------------------------------


def test_builtin_registry_has_five_models() -> None:
    models = MODEL_PRESET_REGISTRY.list_models()
    assert len(models) == 5


@pytest.mark.parametrize(
    "model_id",
    ["gpt-4", "claude-3-5-sonnet", "aurelius-1b", "llama-3-8b", "mistral-7b"],
)
def test_builtin_presets_present(model_id: str) -> None:
    preset = MODEL_PRESET_REGISTRY.get(model_id)
    assert preset is not None
    assert preset.model_id == model_id
    assert preset.max_tokens >= 1
    assert 0.0 <= preset.temperature <= 2.0
