"""Modality tokenization layer — converts raw modality data into ModalityToken sequences.

Inspired by MoonshotAI/Kimi-K2 MoonViT (2602.02276), Llama 4 vision encoder,
Meta AI Apache-2.0, clean-room reimplementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# ModalityType
# ---------------------------------------------------------------------------


class ModalityType(Enum):
    """Supported modalities for the Aurelius multimodal surface."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


# ---------------------------------------------------------------------------
# ModalityToken
# ---------------------------------------------------------------------------


@dataclass
class ModalityToken:
    """A tokenised representation of a single modality input.

    Attributes:
        modality:  The :class:`ModalityType` this token belongs to.
        token_ids: Integer token ids produced by the tokeniser.
        metadata:  Arbitrary key-value pairs (e.g. shape, sample rate, …).
    """

    modality: ModalityType
    token_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ModalityTokenizerError(Exception):
    """Raised for tokenisation errors (unsupported modality, malformed input, …)."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ModalityTokenizer(ABC):
    """Abstract base class for modality tokenisers."""

    @abstractmethod
    def tokenize(self, data: Any, modality: ModalityType) -> ModalityToken:
        """Convert *data* of the given *modality* into a :class:`ModalityToken`.

        Args:
            data:     Raw modality input (type depends on the modality).
            modality: Which :class:`ModalityType` is being tokenised.

        Returns:
            :class:`ModalityToken` wrapping the tokenised representation.
        """


# ---------------------------------------------------------------------------
# PassthroughModalityTokenizer
# ---------------------------------------------------------------------------


class PassthroughModalityTokenizer(ModalityTokenizer):
    """Text-only tokeniser that passes a list of int token ids through unchanged.

    For :attr:`ModalityType.TEXT`, *data* must be a ``list[int]`` of pre-computed
    token ids.  The ids are wrapped in a :class:`ModalityToken` with no further
    transformation.

    For any other modality a :exc:`NotImplementedError` is raised — concrete
    subclasses are responsible for vision and audio tokenisation.
    """

    def tokenize(self, data: Any, modality: ModalityType) -> ModalityToken:
        """Tokenise *data* for *modality*.

        Args:
            data:     For TEXT — ``list[int]`` of token ids.
            modality: :attr:`ModalityType.TEXT` is supported; others raise
                      :exc:`NotImplementedError`.

        Returns:
            :class:`ModalityToken` with ``token_ids=data`` and empty metadata.

        Raises:
            NotImplementedError: If *modality* is not :attr:`ModalityType.TEXT`.
        """
        if modality is not ModalityType.TEXT:
            raise NotImplementedError(
                f"PassthroughModalityTokenizer only handles TEXT modality; "
                f"got {modality!r}. Use a concrete subclass for {modality.value} tokenisation."
            )
        return ModalityToken(modality=modality, token_ids=list(data), metadata={})


__all__ = [
    "ModalityType",
    "ModalityToken",
    "ModalityTokenizerError",
    "ModalityTokenizer",
    "PassthroughModalityTokenizer",
]
