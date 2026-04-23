"""Multimodal surface registry — CRUD layer for vision, audio, projector, and tokenizer modules.

Inspired by MoonshotAI/Kimi-K2 MoonViT (2602.02276), Llama 4 vision encoder,
Meta AI Apache-2.0, clean-room reimplementation.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Registry error
# ---------------------------------------------------------------------------

class MultimodalRegistryError(Exception):
    """Raised when a requested registry key does not exist (no silent fallbacks)."""


# ---------------------------------------------------------------------------
# Internal registry stores (shared singletons mutated by register_* functions)
# ---------------------------------------------------------------------------

VISION_ENCODER_REGISTRY: dict[str, type] = {}
AUDIO_ENCODER_REGISTRY: dict[str, type] = {}
MODALITY_PROJECTOR_REGISTRY: dict[str, type] = {}
MODALITY_TOKENIZER_REGISTRY: dict[str, type] = {}


# ---------------------------------------------------------------------------
# Vision encoder registry
# ---------------------------------------------------------------------------

def register_vision_encoder(name: str, cls: type) -> None:
    """Register a vision encoder class under *name*.

    Args:
        name: Registry key (e.g. ``"VisionTransformer"``).
        cls:  The class to register.
    """
    VISION_ENCODER_REGISTRY[name] = cls


def get_vision_encoder(name: str) -> type:
    """Retrieve a vision encoder class by name.

    Args:
        name: Registry key previously registered with :func:`register_vision_encoder`.

    Returns:
        The registered class.

    Raises:
        MultimodalRegistryError: If *name* is not in the registry.
    """
    if name not in VISION_ENCODER_REGISTRY:
        raise MultimodalRegistryError(
            f"Vision encoder {name!r} not found. "
            f"Available: {list(VISION_ENCODER_REGISTRY.keys())}"
        )
    return VISION_ENCODER_REGISTRY[name]


def list_vision_encoders() -> list[str]:
    """Return a sorted list of all registered vision encoder names."""
    return sorted(VISION_ENCODER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Audio encoder registry
# ---------------------------------------------------------------------------

def register_audio_encoder(name: str, cls: type) -> None:
    """Register an audio encoder class under *name*."""
    AUDIO_ENCODER_REGISTRY[name] = cls


def get_audio_encoder(name: str) -> type:
    """Retrieve an audio encoder class by name.

    Raises:
        MultimodalRegistryError: If *name* is not in the registry.
    """
    if name not in AUDIO_ENCODER_REGISTRY:
        raise MultimodalRegistryError(
            f"Audio encoder {name!r} not found. "
            f"Available: {list(AUDIO_ENCODER_REGISTRY.keys())}"
        )
    return AUDIO_ENCODER_REGISTRY[name]


def list_audio_encoders() -> list[str]:
    """Return a sorted list of all registered audio encoder names."""
    return sorted(AUDIO_ENCODER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Modality projector registry
# ---------------------------------------------------------------------------

def register_modality_projector(name: str, cls: type) -> None:
    """Register a modality projector class under *name*."""
    MODALITY_PROJECTOR_REGISTRY[name] = cls


def get_modality_projector(name: str) -> type:
    """Retrieve a modality projector class by name.

    Raises:
        MultimodalRegistryError: If *name* is not in the registry.
    """
    if name not in MODALITY_PROJECTOR_REGISTRY:
        raise MultimodalRegistryError(
            f"Modality projector {name!r} not found. "
            f"Available: {list(MODALITY_PROJECTOR_REGISTRY.keys())}"
        )
    return MODALITY_PROJECTOR_REGISTRY[name]


def list_modality_projectors() -> list[str]:
    """Return a sorted list of all registered modality projector names."""
    return sorted(MODALITY_PROJECTOR_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Modality tokenizer registry
# ---------------------------------------------------------------------------

def register_modality_tokenizer(name: str, cls: type) -> None:
    """Register a modality tokenizer class under *name*."""
    MODALITY_TOKENIZER_REGISTRY[name] = cls


def get_modality_tokenizer(name: str) -> type:
    """Retrieve a modality tokenizer class by name.

    Raises:
        MultimodalRegistryError: If *name* is not in the registry.
    """
    if name not in MODALITY_TOKENIZER_REGISTRY:
        raise MultimodalRegistryError(
            f"Modality tokenizer {name!r} not found. "
            f"Available: {list(MODALITY_TOKENIZER_REGISTRY.keys())}"
        )
    return MODALITY_TOKENIZER_REGISTRY[name]


def list_modality_tokenizers() -> list[str]:
    """Return a sorted list of all registered modality tokenizer names."""
    return sorted(MODALITY_TOKENIZER_REGISTRY.keys())


__all__ = [
    "MultimodalRegistryError",
    "VISION_ENCODER_REGISTRY",
    "AUDIO_ENCODER_REGISTRY",
    "MODALITY_PROJECTOR_REGISTRY",
    "MODALITY_TOKENIZER_REGISTRY",
    "register_vision_encoder",
    "get_vision_encoder",
    "list_vision_encoders",
    "register_audio_encoder",
    "get_audio_encoder",
    "list_audio_encoders",
    "register_modality_projector",
    "get_modality_projector",
    "list_modality_projectors",
    "register_modality_tokenizer",
    "get_modality_tokenizer",
    "list_modality_tokenizers",
]
