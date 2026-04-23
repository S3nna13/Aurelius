"""Integration tests for the Aurelius multimodal contract package and encoder registries."""

from __future__ import annotations

import json

import src.multimodal as multimodal_pkg
from src.multimodal import ModalityContract, load_modality_contract


def test_multimodal_package_exports_registry_surface():
    for name in (
        "ModalityContract",
        "ModalityContractError",
        "MODALITY_CONTRACT_REGISTRY",
        "VISION_MODALITY_CONTRACT",
        "AUDIO_MODALITY_CONTRACT",
        "DOCUMENT_MODALITY_CONTRACT",
        "load_modality_contract",
        "dump_modality_contract",
        "register_modality_contract",
        "get_modality_contract",
        "list_modality_contracts",
        "describe_modality_registry",
    ):
        assert hasattr(multimodal_pkg, name)
        assert name in multimodal_pkg.__all__


def test_multimodal_registry_round_trip_and_json_summary():
    registry_summary = multimodal_pkg.describe_modality_registry()
    vision = multimodal_pkg.get_modality_contract("vision")
    dumped = multimodal_pkg.dump_modality_contract(vision)
    round_trip = load_modality_contract(dumped)
    custom = ModalityContract(
        name="slide_deck",
        description="Slide deck and presentation inputs and outputs.",
        input_kinds=("pptx", "pdf", "image"),
        output_kinds=("outline", "speaker_notes", "slides"),
    )

    assert registry_summary["count"] == 3
    assert json.dumps(registry_summary)
    assert json.dumps(vision.summary())
    assert round_trip == vision
    assert custom.summary()["kind_summary"].startswith("slide_deck:")
    assert custom.input_kinds == ("pptx", "pdf", "image")


# ---------------------------------------------------------------------------
# New: encoder / projector / tokenizer registry tests
# ---------------------------------------------------------------------------


def test_all_four_encoder_registries_are_non_none():
    """All 4 required registries must be importable and non-None from src.multimodal."""
    assert multimodal_pkg.VISION_ENCODER_REGISTRY is not None
    assert multimodal_pkg.AUDIO_ENCODER_REGISTRY is not None
    assert multimodal_pkg.MODALITY_PROJECTOR_REGISTRY is not None
    assert multimodal_pkg.MODALITY_TOKENIZER_REGISTRY is not None


def test_all_four_encoder_registries_are_dicts():
    """All 4 required registries must be dict instances."""
    assert isinstance(multimodal_pkg.VISION_ENCODER_REGISTRY, dict)
    assert isinstance(multimodal_pkg.AUDIO_ENCODER_REGISTRY, dict)
    assert isinstance(multimodal_pkg.MODALITY_PROJECTOR_REGISTRY, dict)
    assert isinstance(multimodal_pkg.MODALITY_TOKENIZER_REGISTRY, dict)


def test_all_four_encoder_registries_non_empty():
    """All 4 registries must be pre-populated after import."""
    assert len(multimodal_pkg.VISION_ENCODER_REGISTRY) >= 1
    assert len(multimodal_pkg.AUDIO_ENCODER_REGISTRY) >= 1
    assert len(multimodal_pkg.MODALITY_PROJECTOR_REGISTRY) >= 1
    assert len(multimodal_pkg.MODALITY_TOKENIZER_REGISTRY) >= 1


def test_dummy_vision_encoder_round_trip():
    """register → get → instantiate round-trip for a dummy vision encoder class."""

    class _DummyVisionEncoderForIntegration:
        def __init__(self):
            self.tag = "dummy"

    multimodal_pkg.register_vision_encoder(
        "_DummyVisionEncoderForIntegration",
        _DummyVisionEncoderForIntegration,
    )
    retrieved = multimodal_pkg.get_vision_encoder("_DummyVisionEncoderForIntegration")
    assert retrieved is _DummyVisionEncoderForIntegration
    instance = retrieved()
    assert instance.tag == "dummy"


def test_model_side_vision_encoder_registered():
    """VisionTransformer from src.model must appear in VISION_ENCODER_REGISTRY."""
    assert "VisionTransformer" in multimodal_pkg.VISION_ENCODER_REGISTRY


def test_model_side_audio_encoder_registered():
    """WhisperStyleEncoder from src.model must appear in AUDIO_ENCODER_REGISTRY."""
    assert "WhisperStyleEncoder" in multimodal_pkg.AUDIO_ENCODER_REGISTRY


def test_model_side_projector_registered():
    """ModalityProjector and VisionProjector must appear in MODALITY_PROJECTOR_REGISTRY."""
    assert "ModalityProjector" in multimodal_pkg.MODALITY_PROJECTOR_REGISTRY
    assert "VisionProjector" in multimodal_pkg.MODALITY_PROJECTOR_REGISTRY


def test_passthrough_tokenizer_registered():
    """PassthroughModalityTokenizer must appear in MODALITY_TOKENIZER_REGISTRY."""
    assert "PassthroughModalityTokenizer" in multimodal_pkg.MODALITY_TOKENIZER_REGISTRY
