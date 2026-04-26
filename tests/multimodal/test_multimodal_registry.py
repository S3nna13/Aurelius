"""Tests for the multimodal surface registry (src.multimodal.multimodal_registry)."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _DummyEncoder:
    """Minimal stub used for round-trip registration tests."""

    pass


class _AnotherEncoder:
    pass


# ---------------------------------------------------------------------------
# Registry population (model-side auto-registration via __init__)
# ---------------------------------------------------------------------------


def test_vision_encoder_registry_non_empty():
    """VISION_ENCODER_REGISTRY must contain at least one entry after import."""
    import src.multimodal as mm

    assert len(mm.VISION_ENCODER_REGISTRY) >= 1


def test_audio_encoder_registry_non_empty():
    """AUDIO_ENCODER_REGISTRY must contain at least one entry after import."""
    import src.multimodal as mm

    assert len(mm.AUDIO_ENCODER_REGISTRY) >= 1


# ---------------------------------------------------------------------------
# VISION round-trip
# ---------------------------------------------------------------------------


def test_register_and_get_vision_encoder_round_trip():
    """register_vision_encoder → get_vision_encoder must return the same class."""
    from src.multimodal.multimodal_registry import (
        get_vision_encoder,
        register_vision_encoder,
    )

    register_vision_encoder("_DummyEncoder_vision", _DummyEncoder)
    retrieved = get_vision_encoder("_DummyEncoder_vision")
    assert retrieved is _DummyEncoder


def test_get_vision_encoder_unknown_raises():
    """get_vision_encoder with an unregistered name must raise MultimodalRegistryError."""
    from src.multimodal.multimodal_registry import (
        MultimodalRegistryError,
        get_vision_encoder,
    )

    with pytest.raises(MultimodalRegistryError):
        get_vision_encoder("__this_name_does_not_exist__")


def test_list_vision_encoders_returns_list_of_strings():
    """list_vision_encoders() must return a list whose elements are all strings."""
    from src.multimodal.multimodal_registry import list_vision_encoders

    names = list_vision_encoders()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# AUDIO round-trip
# ---------------------------------------------------------------------------


def test_register_and_get_audio_encoder_round_trip():
    """register_audio_encoder → get_audio_encoder must return the same class."""
    from src.multimodal.multimodal_registry import (
        get_audio_encoder,
        register_audio_encoder,
    )

    register_audio_encoder("_DummyEncoder_audio", _DummyEncoder)
    retrieved = get_audio_encoder("_DummyEncoder_audio")
    assert retrieved is _DummyEncoder


def test_get_audio_encoder_unknown_raises():
    """get_audio_encoder with an unregistered name must raise MultimodalRegistryError."""
    from src.multimodal.multimodal_registry import (
        MultimodalRegistryError,
        get_audio_encoder,
    )

    with pytest.raises(MultimodalRegistryError):
        get_audio_encoder("__this_name_does_not_exist__")


def test_list_audio_encoders_returns_list_of_strings():
    """list_audio_encoders() must return a list whose elements are all strings."""
    from src.multimodal.multimodal_registry import list_audio_encoders

    names = list_audio_encoders()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# PROJECTOR round-trip
# ---------------------------------------------------------------------------


def test_register_and_get_modality_projector_round_trip():
    from src.multimodal.multimodal_registry import (
        get_modality_projector,
        register_modality_projector,
    )

    register_modality_projector("_DummyProjector", _DummyEncoder)
    assert get_modality_projector("_DummyProjector") is _DummyEncoder


def test_get_modality_projector_unknown_raises():
    from src.multimodal.multimodal_registry import (
        MultimodalRegistryError,
        get_modality_projector,
    )

    with pytest.raises(MultimodalRegistryError):
        get_modality_projector("__no_such_projector__")


# ---------------------------------------------------------------------------
# TOKENIZER round-trip
# ---------------------------------------------------------------------------


def test_register_and_get_modality_tokenizer_round_trip():
    from src.multimodal.multimodal_registry import (
        get_modality_tokenizer,
        register_modality_tokenizer,
    )

    register_modality_tokenizer("_DummyTokenizer", _DummyEncoder)
    assert get_modality_tokenizer("_DummyTokenizer") is _DummyEncoder


def test_get_modality_tokenizer_unknown_raises():
    from src.multimodal.multimodal_registry import (
        MultimodalRegistryError,
        get_modality_tokenizer,
    )

    with pytest.raises(MultimodalRegistryError):
        get_modality_tokenizer("__no_such_tokenizer__")
