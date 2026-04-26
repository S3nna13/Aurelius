"""Tests for src/multimodal/document_understanding.py --- JSONLayoutParser and DocumentEmbedder."""

from __future__ import annotations

import torch

from src.multimodal.document_understanding import (
    DOCUMENT_EMBEDDER_REGISTRY,
    DOCUMENT_PARSER_REGISTRY,
    DocumentEmbedder,
    DocumentEmbedderConfig,
    DocumentPage,
    DocumentParser,
    DocumentRegion,
    DocumentRegionType,
    JSONLayoutParser,
)

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

VALID_DOC = {
    "pages": [
        {
            "page_number": 1,
            "width": 612.0,
            "height": 792.0,
            "regions": [
                {
                    "region_type": "text",
                    "text": "Hello world",
                    "bbox": [10, 20, 200, 40],
                    "confidence": 0.99,
                },
                {
                    "region_type": "table",
                    "text": "Row1 Col1\tRow1 Col2",
                    "bbox": [10, 50, 400, 200],
                },
            ],
        },
        {
            "page_number": 2,
            "width": 612.0,
            "height": 792.0,
            "regions": [
                {"region_type": "header", "text": "Report Title"},
            ],
        },
    ]
}

TINY_EMB_CFG = DocumentEmbedderConfig(vocab_size=256, d_model=128, max_regions=64, max_text_len=512)

# ---------------------------------------------------------------------------
# DocumentRegionType enum tests
# ---------------------------------------------------------------------------


def test_region_type_has_text():
    assert DocumentRegionType.TEXT is not None


def test_region_type_has_table():
    assert DocumentRegionType.TABLE is not None


def test_region_type_has_figure():
    assert DocumentRegionType.FIGURE is not None


def test_region_type_has_form_field():
    assert DocumentRegionType.FORM_FIELD is not None


def test_region_type_has_header():
    assert DocumentRegionType.HEADER is not None


def test_region_type_has_footer():
    assert DocumentRegionType.FOOTER is not None


def test_region_type_has_list():
    assert DocumentRegionType.LIST is not None


# ---------------------------------------------------------------------------
# JSONLayoutParser --- valid input
# ---------------------------------------------------------------------------


def test_json_parser_returns_correct_page_count():
    parser = JSONLayoutParser()
    pages = parser.parse(VALID_DOC)
    assert len(pages) == 2


def test_json_parser_page_number():
    parser = JSONLayoutParser()
    pages = parser.parse(VALID_DOC)
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2


def test_json_parser_page_dimensions():
    parser = JSONLayoutParser()
    pages = parser.parse(VALID_DOC)
    assert pages[0].width == 612.0
    assert pages[0].height == 792.0


def test_json_parser_region_count():
    parser = JSONLayoutParser()
    pages = parser.parse(VALID_DOC)
    assert len(pages[0].regions) == 2
    assert len(pages[1].regions) == 1


def test_json_parser_region_types():
    parser = JSONLayoutParser()
    pages = parser.parse(VALID_DOC)
    assert pages[0].regions[0].region_type == DocumentRegionType.TEXT
    assert pages[0].regions[1].region_type == DocumentRegionType.TABLE
    assert pages[1].regions[0].region_type == DocumentRegionType.HEADER


def test_json_parser_region_text():
    parser = JSONLayoutParser()
    pages = parser.parse(VALID_DOC)
    assert pages[0].regions[0].text == "Hello world"


def test_json_parser_region_bbox():
    parser = JSONLayoutParser()
    pages = parser.parse(VALID_DOC)
    assert pages[0].regions[0].bbox == (10.0, 20.0, 200.0, 40.0)


def test_json_parser_region_confidence():
    parser = JSONLayoutParser()
    pages = parser.parse(VALID_DOC)
    assert abs(pages[0].regions[0].confidence - 0.99) < 1e-9


def test_json_parser_returns_document_page_instances():
    parser = JSONLayoutParser()
    pages = parser.parse(VALID_DOC)
    for page in pages:
        assert isinstance(page, DocumentPage)


def test_json_parser_regions_are_document_region_instances():
    parser = JSONLayoutParser()
    pages = parser.parse(VALID_DOC)
    for region in pages[0].regions:
        assert isinstance(region, DocumentRegion)


# ---------------------------------------------------------------------------
# JSONLayoutParser --- error / edge cases
# ---------------------------------------------------------------------------


def test_json_parser_empty_dict_returns_empty_list():
    parser = JSONLayoutParser()
    result = parser.parse({})
    assert result == []


def test_json_parser_missing_pages_key_returns_empty_list():
    parser = JSONLayoutParser()
    result = parser.parse({"documents": []})
    assert result == []


def test_json_parser_non_dict_input_returns_empty_list():
    parser = JSONLayoutParser()
    result = parser.parse([])  # type: ignore[arg-type]
    assert result == []


def test_json_parser_none_pages_returns_empty_list():
    parser = JSONLayoutParser()
    result = parser.parse({"pages": None})
    assert result == []


def test_json_parser_bad_region_type_defaults_to_text():
    doc = {
        "pages": [
            {
                "page_number": 1,
                "width": 100,
                "height": 200,
                "regions": [{"region_type": "unknown_type", "text": "hi"}],
            }
        ]
    }
    parser = JSONLayoutParser()
    pages = parser.parse(doc)
    assert pages[0].regions[0].region_type == DocumentRegionType.TEXT


# ---------------------------------------------------------------------------
# DocumentEmbedder tests
# ---------------------------------------------------------------------------


def test_embedder_output_shape_two_regions():
    page = DocumentPage(
        page_number=1,
        width=612.0,
        height=792.0,
        regions=[
            DocumentRegion(region_type=DocumentRegionType.TEXT, text="Hello"),
            DocumentRegion(region_type=DocumentRegionType.TABLE, text="A B C"),
        ],
    )
    embedder = DocumentEmbedder(TINY_EMB_CFG)
    out = embedder(page)
    assert out.shape == (2, 128)


def test_embedder_no_nan():
    page = DocumentPage(
        page_number=1,
        width=612.0,
        height=792.0,
        regions=[DocumentRegion(region_type=DocumentRegionType.TEXT, text="Test text")],
    )
    embedder = DocumentEmbedder(TINY_EMB_CFG)
    out = embedder(page)
    assert not torch.isnan(out).any()


def test_embedder_empty_page_returns_zero_rows():
    page = DocumentPage(page_number=1, width=612.0, height=792.0, regions=[])
    embedder = DocumentEmbedder(TINY_EMB_CFG)
    out = embedder(page)
    assert out.shape == (0, 128)


def test_embedder_empty_text_region_does_not_crash():
    page = DocumentPage(
        page_number=1,
        width=612.0,
        height=792.0,
        regions=[DocumentRegion(region_type=DocumentRegionType.TEXT, text="")],
    )
    embedder = DocumentEmbedder(TINY_EMB_CFG)
    out = embedder(page)
    assert out.shape == (1, 128)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_document_parser_registry_has_json_layout():
    assert "json_layout" in DOCUMENT_PARSER_REGISTRY


def test_document_parser_registry_json_layout_is_class():
    cls = DOCUMENT_PARSER_REGISTRY["json_layout"]
    assert issubclass(cls, DocumentParser)


def test_document_embedder_registry_is_dict():
    assert isinstance(DOCUMENT_EMBEDDER_REGISTRY, dict)
