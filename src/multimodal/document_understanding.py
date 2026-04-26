"""Document understanding — table/form/layout document processor.

Inspired by MoonshotAI/Kimi-K2 MoonViT 3D patch packer (2602.02276), Google Gemini 2.5
doc grounding (Tech Report 2025), Apache-2.0, clean-room reimplementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Region type enum
# ---------------------------------------------------------------------------


class DocumentRegionType(Enum):
    """Semantic category of a document region."""

    TEXT = auto()
    TABLE = auto()
    FIGURE = auto()
    FORM_FIELD = auto()
    HEADER = auto()
    FOOTER = auto()
    LIST = auto()


# ---------------------------------------------------------------------------
# Document data structures
# ---------------------------------------------------------------------------


@dataclass
class DocumentRegion:
    """A single semantic region extracted from a document page.

    Attributes:
        region_type: :class:`DocumentRegionType` category.
        text:        Extracted text content of the region.
        bbox:        Optional (x0, y0, x1, y1) bounding box in page units.
        confidence:  Parser confidence score in [0, 1].
        children:    Nested child regions (e.g. table cells inside a table).
    """

    region_type: DocumentRegionType
    text: str
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 1.0
    children: list[DocumentRegion] = field(default_factory=list)


@dataclass
class DocumentPage:
    """A parsed page from a document.

    Attributes:
        page_number: 1-based page index.
        width:       Page width in points.
        height:      Page height in points.
        regions:     Ordered list of :class:`DocumentRegion` objects.
    """

    page_number: int
    width: float
    height: float
    regions: list[DocumentRegion] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract parser
# ---------------------------------------------------------------------------


class DocumentParser(ABC):
    """Abstract base class for document parsers.

    Subclasses must implement :meth:`parse`.
    """

    @abstractmethod
    def parse(self, raw_data: dict) -> list[DocumentPage]:
        """Parse raw document data into a list of :class:`DocumentPage` objects.

        Args:
            raw_data: Parsed (i.e. already-deserialised) document dict.

        Returns:
            List of :class:`DocumentPage` instances (may be empty).
        """


# ---------------------------------------------------------------------------
# JSON layout parser
# ---------------------------------------------------------------------------

_REGION_TYPE_MAP: dict[str, DocumentRegionType] = {rt.name.lower(): rt for rt in DocumentRegionType}
# Also accept the raw enum names in mixed/upper case.
_REGION_TYPE_MAP.update({rt.name: rt for rt in DocumentRegionType})


def _parse_region(raw: Any) -> DocumentRegion:
    """Best-effort conversion of a raw dict to a :class:`DocumentRegion`.

    Missing or invalid fields produce safe defaults rather than raising.
    """
    if not isinstance(raw, dict):
        return DocumentRegion(region_type=DocumentRegionType.TEXT, text="")

    # region_type
    rt_raw = raw.get("region_type", "text")
    region_type = _REGION_TYPE_MAP.get(str(rt_raw).lower(), DocumentRegionType.TEXT)

    text = str(raw.get("text", ""))

    bbox_raw = raw.get("bbox")
    bbox: tuple[float, float, float, float] | None = None
    if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        try:
            bbox = (
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            )
        except (TypeError, ValueError):
            bbox = None

    try:
        confidence = float(raw.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0

    children_raw = raw.get("children", [])
    children: list[DocumentRegion] = []
    if isinstance(children_raw, list):
        for child_raw in children_raw:
            children.append(_parse_region(child_raw))

    return DocumentRegion(
        region_type=region_type,
        text=text,
        bbox=bbox,
        confidence=confidence,
        children=children,
    )


def _parse_page(raw: Any) -> DocumentPage:
    """Best-effort conversion of a raw dict to a :class:`DocumentPage`."""
    if not isinstance(raw, dict):
        return DocumentPage(page_number=1, width=0.0, height=0.0)

    try:
        page_number = int(raw.get("page_number", 1))
    except (TypeError, ValueError):
        page_number = 1

    try:
        width = float(raw.get("width", 0.0))
    except (TypeError, ValueError):
        width = 0.0

    try:
        height = float(raw.get("height", 0.0))
    except (TypeError, ValueError):
        height = 0.0

    regions_raw = raw.get("regions", [])
    regions: list[DocumentRegion] = []
    if isinstance(regions_raw, list):
        for r in regions_raw:
            regions.append(_parse_region(r))

    return DocumentPage(
        page_number=page_number,
        width=width,
        height=height,
        regions=regions,
    )


class JSONLayoutParser(DocumentParser):
    """Parse documents from a structured JSON layout dict (stdlib-only, no PDF libs).

    Expected input format::

        {
            "pages": [
                {
                    "page_number": 1,
                    "width": 612,
                    "height": 792,
                    "regions": [
                        {
                            "region_type": "text",
                            "text": "Hello world",
                            "bbox": [10, 20, 200, 40],
                            "confidence": 0.99
                        }
                    ]
                }
            ]
        }

    Invalid or missing fields produce empty :class:`DocumentPage` objects rather
    than raising exceptions.
    """

    def parse(self, raw_data: dict) -> list[DocumentPage]:
        """Parse a JSON layout dict into :class:`DocumentPage` objects.

        Args:
            raw_data: Already-deserialised dict (use ``json.loads`` upstream if
                      starting from a string).

        Returns:
            List of :class:`DocumentPage`; empty list if ``raw_data`` is
            not a dict or has no ``"pages"`` key.
        """
        if not isinstance(raw_data, dict):
            return []

        pages_raw = raw_data.get("pages")
        if pages_raw is None:
            return []

        if not isinstance(pages_raw, list):
            return []

        return [_parse_page(p) for p in pages_raw]


# ---------------------------------------------------------------------------
# Document embedder config
# ---------------------------------------------------------------------------


@dataclass
class DocumentEmbedderConfig:
    """Configuration for :class:`DocumentEmbedder`.

    Attributes:
        vocab_size:   Character vocabulary size (char id = ord(c) % vocab_size).
        d_model:      Embedding dimension.
        max_regions:  Maximum number of regions per page (clips at this limit).
        max_text_len: Maximum characters per region text (clips at this limit).
    """

    vocab_size: int = 256
    d_model: int = 128
    max_regions: int = 64
    max_text_len: int = 512


# ---------------------------------------------------------------------------
# Document embedder
# ---------------------------------------------------------------------------


class DocumentEmbedder(nn.Module):
    """Embed a :class:`DocumentPage` into a dense token matrix.

    For each region in the page:
    1. Tokenise the region text as char-level integer ids (``ord(c) % vocab_size``).
    2. Look up embeddings via :class:`nn.Embedding`.
    3. Mean-pool over the text sequence to yield one (d_model,) vector per region.

    The result is a (N_regions, d_model) tensor.

    Args:
        config: :class:`DocumentEmbedderConfig` instance.
    """

    def __init__(self, config: DocumentEmbedderConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenise_text(self, text: str) -> Tensor:
        """Convert text to a 1-D integer tensor of char ids.

        Args:
            text: Region text string.

        Returns:
            (L,) long tensor where L = min(len(text), max_text_len), L >= 1.
        """
        cfg = self.config
        # Clip to max_text_len
        text = text[: cfg.max_text_len] if len(text) > cfg.max_text_len else text
        if not text:
            # Pad with a single zero token to avoid empty mean-pool
            return torch.zeros(1, dtype=torch.long)
        ids = [ord(c) % cfg.vocab_size for c in text]
        return torch.tensor(ids, dtype=torch.long)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, page: DocumentPage) -> Tensor:
        """Embed all regions in *page*.

        Args:
            page: :class:`DocumentPage` to embed.

        Returns:
            (N, d_model) float tensor where N = min(len(page.regions), max_regions).
            Returns a zero-row tensor of shape (0, d_model) if page has no regions.
        """
        cfg = self.config
        regions = page.regions[: cfg.max_regions]

        if not regions:
            return torch.zeros(0, cfg.d_model)

        region_embeds: list[Tensor] = []
        for region in regions:
            ids = self._tokenise_text(region.text)  # (L,)
            emb = self.embed(ids)  # (L, d_model)
            pooled = emb.mean(dim=0)  # (d_model,)
            region_embeds.append(pooled)

        return torch.stack(region_embeds, dim=0)  # (N, d_model)


# ---------------------------------------------------------------------------
# Module-level registries
# ---------------------------------------------------------------------------

DOCUMENT_PARSER_REGISTRY: dict[str, type[DocumentParser]] = {
    "json_layout": JSONLayoutParser,
}

DOCUMENT_EMBEDDER_REGISTRY: dict[str, type[DocumentEmbedder]] = {}


__all__ = [
    "DocumentRegionType",
    "DocumentRegion",
    "DocumentPage",
    "DocumentParser",
    "JSONLayoutParser",
    "DocumentEmbedderConfig",
    "DocumentEmbedder",
    "DOCUMENT_PARSER_REGISTRY",
    "DOCUMENT_EMBEDDER_REGISTRY",
]
