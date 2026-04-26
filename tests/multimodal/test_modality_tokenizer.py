"""Tests for src.multimodal.modality_tokenizer."""

from __future__ import annotations

import dataclasses

import pytest

from src.multimodal.modality_tokenizer import (
    ModalityToken,
    ModalityTokenizer,
    ModalityTokenizerError,
    ModalityType,
    PassthroughModalityTokenizer,
)

# ---------------------------------------------------------------------------
# ModalityToken dataclass
# ---------------------------------------------------------------------------


def test_modality_token_is_dataclass():
    """ModalityToken must be a proper dataclass."""
    assert dataclasses.is_dataclass(ModalityToken)


def test_modality_token_fields():
    """ModalityToken must have modality, token_ids, and metadata fields."""
    field_names = {f.name for f in dataclasses.fields(ModalityToken)}
    assert "modality" in field_names
    assert "token_ids" in field_names
    assert "metadata" in field_names


def test_modality_token_construction():
    """ModalityToken can be constructed with expected field values."""
    tok = ModalityToken(
        modality=ModalityType.TEXT,
        token_ids=[1, 2, 3],
        metadata={"source": "test"},
    )
    assert tok.modality is ModalityType.TEXT
    assert tok.token_ids == [1, 2, 3]
    assert tok.metadata == {"source": "test"}


def test_modality_token_default_metadata():
    """ModalityToken metadata defaults to an empty dict."""
    tok = ModalityToken(modality=ModalityType.IMAGE, token_ids=[])
    assert tok.metadata == {}


# ---------------------------------------------------------------------------
# ModalityType enum
# ---------------------------------------------------------------------------


def test_modality_type_values():
    """ModalityType must contain at least TEXT, IMAGE, AUDIO, VIDEO, DOCUMENT."""
    names = {m.name for m in ModalityType}
    assert names >= {"TEXT", "IMAGE", "AUDIO", "VIDEO", "DOCUMENT"}


# ---------------------------------------------------------------------------
# PassthroughModalityTokenizer — TEXT modality
# ---------------------------------------------------------------------------


def test_passthrough_text_returns_modality_token():
    """PassthroughModalityTokenizer.tokenize(TEXT) must return a ModalityToken."""
    tokenizer = PassthroughModalityTokenizer()
    result = tokenizer.tokenize([10, 20, 30], ModalityType.TEXT)
    assert isinstance(result, ModalityToken)


def test_passthrough_text_correct_token_ids():
    """PassthroughModalityTokenizer must wrap token_ids unchanged."""
    tokenizer = PassthroughModalityTokenizer()
    ids = [5, 99, 1024, 0]
    result = tokenizer.tokenize(ids, ModalityType.TEXT)
    assert result.token_ids == ids


def test_passthrough_text_correct_modality():
    """PassthroughModalityTokenizer result must have modality=TEXT."""
    tokenizer = PassthroughModalityTokenizer()
    result = tokenizer.tokenize([1], ModalityType.TEXT)
    assert result.modality is ModalityType.TEXT


def test_passthrough_text_empty_ids():
    """PassthroughModalityTokenizer accepts empty token_ids list."""
    tokenizer = PassthroughModalityTokenizer()
    result = tokenizer.tokenize([], ModalityType.TEXT)
    assert result.token_ids == []


# ---------------------------------------------------------------------------
# PassthroughModalityTokenizer — non-TEXT modalities raise NotImplementedError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "modality",
    [
        ModalityType.IMAGE,
        ModalityType.AUDIO,
        ModalityType.VIDEO,
        ModalityType.DOCUMENT,
    ],
)
def test_passthrough_non_text_raises_not_implemented(modality):
    """PassthroughModalityTokenizer must raise NotImplementedError for non-TEXT modalities."""
    tokenizer = PassthroughModalityTokenizer()
    with pytest.raises(NotImplementedError):
        tokenizer.tokenize(b"some_bytes", modality)


# ---------------------------------------------------------------------------
# ModalityTokenizer is abstract
# ---------------------------------------------------------------------------


def test_modality_tokenizer_is_abstract():
    """ModalityTokenizer cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ModalityTokenizer()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# MODALITY_TOKENIZER_REGISTRY
# ---------------------------------------------------------------------------


def test_modality_tokenizer_registry_is_dict():
    """MODALITY_TOKENIZER_REGISTRY must be accessible from src.multimodal and be a dict."""
    import src.multimodal as mm

    assert isinstance(mm.MODALITY_TOKENIZER_REGISTRY, dict)


def test_modality_tokenizer_registry_non_empty():
    """MODALITY_TOKENIZER_REGISTRY must have at least one entry after import."""
    import src.multimodal as mm

    assert len(mm.MODALITY_TOKENIZER_REGISTRY) >= 1


# ---------------------------------------------------------------------------
# ModalityTokenizerError
# ---------------------------------------------------------------------------


def test_modality_tokenizer_error_is_exception():
    """ModalityTokenizerError must be a subclass of Exception."""
    assert issubclass(ModalityTokenizerError, Exception)
