from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ModelPreset",
    "ModelPresetRegistry",
    "ModelPresetError",
    "MODEL_PRESET_REGISTRY",
]


class ModelPresetError(Exception):
    """Raised for invalid preset operations."""


@dataclass
class ModelPreset:
    model_id: str
    temperature: float
    max_tokens: int
    top_p: float = 1.0
    presence_penalty: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ModelPresetError("model_id must be a non-empty string")
        if not (0.0 <= self.temperature <= 2.0):
            raise ModelPresetError(f"temperature must be in [0.0, 2.0], got {self.temperature}")
        if self.max_tokens < 1:
            raise ModelPresetError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if not (0.0 < self.top_p <= 1.0):
            raise ModelPresetError(f"top_p must be in (0.0, 1.0], got {self.top_p}")


class ModelPresetRegistry:
    """Per-model temperature/max_tokens presets registry."""

    def __init__(self) -> None:
        self._presets: dict[str, ModelPreset] = {}

    def register(self, preset: ModelPreset) -> None:
        if not isinstance(preset, ModelPreset):
            raise ModelPresetError(
                f"preset must be a ModelPreset instance, got {type(preset).__name__}"
            )
        self._presets[preset.model_id] = preset

    def get(self, model_id: str) -> ModelPreset | None:
        return self._presets.get(model_id)

    def list_models(self) -> list[str]:
        return sorted(self._presets)

    def apply_preset(self, model_id: str, overrides: dict) -> dict:
        preset = self._presets.get(model_id)
        if preset is None:
            raise ModelPresetError(f"No preset registered for model_id {model_id!r}")
        base = {
            "model_id": preset.model_id,
            "temperature": preset.temperature,
            "max_tokens": preset.max_tokens,
            "top_p": preset.top_p,
            "presence_penalty": preset.presence_penalty,
            "description": preset.description,
        }
        base.update(overrides)
        return base


_BUILTIN_PRESETS = [
    ModelPreset(
        model_id="gpt-4",
        temperature=0.7,
        max_tokens=4096,
        top_p=1.0,
        presence_penalty=0.0,
        description="OpenAI GPT-4 balanced preset",
    ),
    ModelPreset(
        model_id="claude-3-5-sonnet",
        temperature=0.5,
        max_tokens=8192,
        top_p=1.0,
        presence_penalty=0.0,
        description="Anthropic Claude 3.5 Sonnet balanced preset",
    ),
    ModelPreset(
        model_id="aurelius-1b",
        temperature=0.8,
        max_tokens=2048,
        top_p=0.95,
        presence_penalty=0.0,
        description="Aurelius 1.395B research model preset",
    ),
    ModelPreset(
        model_id="llama-3-8b",
        temperature=0.7,
        max_tokens=4096,
        top_p=0.9,
        presence_penalty=0.0,
        description="Meta LLaMA 3 8B balanced preset",
    ),
    ModelPreset(
        model_id="mistral-7b",
        temperature=0.75,
        max_tokens=4096,
        top_p=0.95,
        presence_penalty=0.0,
        description="Mistral 7B balanced preset",
    ),
]

MODEL_PRESET_REGISTRY = ModelPresetRegistry()
for _preset in _BUILTIN_PRESETS:
    MODEL_PRESET_REGISTRY.register(_preset)
