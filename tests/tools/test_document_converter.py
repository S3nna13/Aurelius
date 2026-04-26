import tempfile
from pathlib import Path

import pytest

from src.tools.document_converter import (
    DOCUMENT_CONVERTER_REGISTRY,
    DEFAULT_DOCUMENT_CONVERTER,
    ConversionResult,
    DocumentConversionError,
    DocumentConverter,
)


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------
def test_supported_formats():
    converter = DocumentConverter()
    formats = converter.supported_formats()
    assert isinstance(formats, list)
    assert ".csv" in formats
    assert ".html" in formats
    assert ".htm" in formats
    assert ".json" in formats
    assert ".md" in formats
    assert ".txt" in formats
    assert ".xml" in formats


def test_document_conversion_error_exists():
    assert issubclass(DocumentConversionError, Exception)


def test_default_singleton():
    assert isinstance(DEFAULT_DOCUMENT_CONVERTER, DocumentConverter)
    assert "default" in DOCUMENT_CONVERTER_REGISTRY
    assert DOCUMENT_CONVERTER_REGISTRY["default"] is DEFAULT_DOCUMENT_CONVERTER


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def test_convert_html_headings():
    converter = DocumentConverter()
    html = "<h1>Title</h1><h2>Sub</h2><h3>Sub-sub</h3>"
    result = converter.convert_text(html, "html")
    assert result.success is True
    assert "# Title" in result.markdown
    assert "## Sub" in result.markdown
    assert "### Sub-sub" in result.markdown


def test_convert_html_paragraphs_and_links():
    converter = DocumentConverter()
    html = '<p>Hello <a href="https://example.com">world</a>!</p>'
    result = converter.convert_text(html, "html")
    assert result.success is True
    assert "[world](https://example.com)" in result.markdown


def test_convert_html_unordered_list():
    converter = DocumentConverter()
    html = "<ul><li>First</li><li>Second</li></ul>"
    result = converter.convert_text(html, "html")
    assert result.success is True
    assert "- First" in result.markdown
    assert "- Second" in result.markdown


def test_convert_html_ordered_list():
    converter = DocumentConverter()
    html = "<ol><li>One</li><li>Two</li></ol>"
    result = converter.convert_text(html, "html")
    assert result.success is True
    assert "1. One" in result.markdown
    assert "2. Two" in result.markdown


def test_convert_html_code_blocks():
    converter = DocumentConverter()
    html = "<pre><code>print('hello')</code></pre>"
    result = converter.convert_text(html, "html")
    assert result.success is True
    assert "```" in result.markdown
    assert "print('hello')" in result.markdown


def test_convert_html_inline_code():
    converter = DocumentConverter()
    html = "<p>Use <code>pip install</code> to get started.</p>"
    result = converter.convert_text(html, "html")
    assert result.success is True
    assert "`pip install`" in result.markdown


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def test_convert_json_pretty_print():
    converter = DocumentConverter()
    raw = '{"a":1,"b":2}'
    result = converter.convert_text(raw, "json")
    assert result.success is True
    assert result.markdown.startswith("```json")
    assert result.markdown.endswith("```")
    assert '"a": 1' in result.markdown


def test_convert_json_jq_path():
    converter = DocumentConverter()
    raw = '{"user": {"name": "Alice", "age": 30}}'
    result = converter.convert_text(raw, "json", jq_path="user.name")
    assert result.success is True
    assert '"Alice"' in result.markdown


def test_convert_json_jq_path_list_index():
    converter = DocumentConverter()
    raw = '{"items": [10, 20, 30]}'
    result = converter.convert_text(raw, "json", jq_path="items.1")
    assert result.success is True
    assert "20" in result.markdown


# ---------------------------------------------------------------------------
# XML
# ---------------------------------------------------------------------------
def test_convert_xml_strip_tags():
    converter = DocumentConverter()
    xml = "<root><item>Hello</item><item>World</item></root>"
    result = converter.convert_text(xml, "xml")
    assert result.success is True
    assert "Hello" in result.markdown
    assert "World" in result.markdown


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def test_convert_csv_table():
    converter = DocumentConverter()
    csv_text = "Name,Age\nAlice,30\nBob,25"
    result = converter.convert_text(csv_text, "csv")
    assert result.success is True
    lines = result.markdown.splitlines()
    assert "| Name | Age |" in lines[0]
    assert "| --- | --- |" in lines[1]
    assert "| Alice | 30 |" in lines[2]
    assert "| Bob | 25 |" in lines[3]


# ---------------------------------------------------------------------------
# Passthrough
# ---------------------------------------------------------------------------
def test_convert_txt_passthrough():
    converter = DocumentConverter()
    text = "Plain text content"
    result = converter.convert_text(text, "txt")
    assert result.success is True
    assert result.markdown == text


def test_convert_md_passthrough():
    converter = DocumentConverter()
    text = "# Already Markdown"
    result = converter.convert_text(text, "md")
    assert result.success is True
    assert result.markdown == text


# ---------------------------------------------------------------------------
# Unknown format
# ---------------------------------------------------------------------------
def test_convert_unknown_format():
    converter = DocumentConverter()
    text = "Some random data"
    result = converter.convert_text(text, "unknown")
    assert result.success is True
    assert "Warning: Unknown format" in result.markdown
    assert "Some random data" in result.markdown


# ---------------------------------------------------------------------------
# File-based conversion
# ---------------------------------------------------------------------------
def test_convert_html_file():
    converter = DocumentConverter()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.html"
        path.write_text("<h1>File Test</h1>", encoding="utf-8")
        result = converter.convert(path)
    assert result.success is True
    assert "# File Test" in result.markdown
    assert result.source_format == "html"


def test_convert_txt_file_no_extension():
    converter = DocumentConverter()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "README"
        path.write_text("No extension", encoding="utf-8")
        result = converter.convert(path)
    assert result.success is True
    assert result.markdown == "No extension"
    assert result.source_format == "txt"


# ---------------------------------------------------------------------------
# Encoding fallback
# ---------------------------------------------------------------------------
def test_encoding_utf8():
    converter = DocumentConverter()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "utf8.txt"
        path.write_text("UTF-8 content: café", encoding="utf-8")
        result = converter.convert(path)
    assert result.success is True
    assert "café" in result.markdown


def test_encoding_fallback_latin1():
    converter = DocumentConverter()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "latin1.txt"
        path.write_bytes(b"Latin-1 content: caf\xe9")
        result = converter.convert(path)
    assert result.success is True
    assert "café" in result.markdown


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
def test_file_not_found():
    converter = DocumentConverter()
    result = converter.convert("/nonexistent/path/file.txt")
    assert result.success is False
    assert result.error_message is not None
    assert (
        "No such file" in result.error_message
        or "not found" in result.error_message.lower()
    )


def test_malformed_json():
    converter = DocumentConverter()
    result = converter.convert_text("{bad json", "json")
    assert result.success is False
    assert result.error_message is not None


def test_malformed_xml():
    converter = DocumentConverter()
    result = converter.convert_text("<unclosed>", "xml")
    assert result.success is False
    assert result.error_message is not None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_contains_default():
    assert "default" in DOCUMENT_CONVERTER_REGISTRY
    registry_instance = DOCUMENT_CONVERTER_REGISTRY["default"]
    assert isinstance(registry_instance, DocumentConverter)


def test_conversion_result_dataclass():
    result = ConversionResult(
        markdown="# Hello",
        source_path="/tmp/test.md",
        source_format="md",
        success=True,
    )
    assert result.error_message is None
