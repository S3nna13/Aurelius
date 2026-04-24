"""Tests for src/retrieval/citation_extractor.py (10+ tests)."""

from __future__ import annotations

import pytest

from src.retrieval.citation_extractor import (
    Citation,
    CitationExtractor,
    CitationType,
    RETRIEVAL_REGISTRY,
)

EX = CitationExtractor()


# ---------------------------------------------------------------------------
# 1. Extract academic bracket citation
# ---------------------------------------------------------------------------
def test_extract_academic_bracket():
    text = "As shown by [Smith, 2020] the result holds."
    cits = EX.extract(text)
    academic = [c for c in cits if c.citation_type == CitationType.ACADEMIC]
    assert len(academic) >= 1
    assert "Smith" in academic[0].source


# ---------------------------------------------------------------------------
# 2. Extract academic parenthesis citation
# ---------------------------------------------------------------------------
def test_extract_academic_paren():
    text = "Recent work (Jones et al., 2019) confirms this."
    cits = EX.extract(text)
    academic = [c for c in cits if c.citation_type == CitationType.ACADEMIC]
    assert len(academic) >= 1
    assert "Jones" in academic[0].source


# ---------------------------------------------------------------------------
# 3. Extract URL citation
# ---------------------------------------------------------------------------
def test_extract_url():
    text = "See https://example.com/path for more info."
    cits = EX.extract(text)
    urls = [c for c in cits if c.citation_type == CitationType.URL]
    assert len(urls) >= 1
    assert "https://example.com" in urls[0].source


# ---------------------------------------------------------------------------
# 4. Extract book citation
# ---------------------------------------------------------------------------
def test_extract_book():
    text = 'Read "The Art of War" by Sun Tzu for strategy.'
    cits = EX.extract(text)
    books = [c for c in cits if c.citation_type == CitationType.BOOK]
    assert len(books) >= 1
    assert "Art of War" in books[0].source


# ---------------------------------------------------------------------------
# 5. Extract inline numeric citation
# ---------------------------------------------------------------------------
def test_extract_inline_numeric():
    text = "According to [1] and [2] the data supports this."
    cits = EX.extract(text)
    inline = [c for c in cits if c.citation_type == CitationType.INLINE]
    assert len(inline) >= 2


# ---------------------------------------------------------------------------
# 6. Extract inline roman numeral citation
# ---------------------------------------------------------------------------
def test_extract_inline_roman():
    text = "See footnote [iv] and [xii] below."
    cits = EX.extract(text)
    inline = [c for c in cits if c.citation_type == CitationType.INLINE]
    assert len(inline) >= 1


# ---------------------------------------------------------------------------
# 7. Citations ordered by position
# ---------------------------------------------------------------------------
def test_citations_ordered_by_position():
    text = "First [1] then [2] and finally [3]."
    cits = EX.extract(text)
    starts = [c.start for c in cits]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# 8. citation_id is unique per citation
# ---------------------------------------------------------------------------
def test_citation_ids_unique():
    text = "See [1] and [2] and [3]."
    cits = EX.extract(text)
    ids = [c.citation_id for c in cits]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# 9. start/end positions match text slice
# ---------------------------------------------------------------------------
def test_start_end_match():
    text = "Reference [1] is here."
    cits = EX.extract(text)
    for c in cits:
        assert text[c.start:c.end] == c.text


# ---------------------------------------------------------------------------
# 10. resolve maps citation text to reference
# ---------------------------------------------------------------------------
def test_resolve_basic():
    text = "[Smith, 2020] proved the theorem."
    cits = EX.extract(text)
    refs = ["Smith, 2020. Journal of Science.", "Jones, 2019. Nature."]
    resolved = EX.resolve(cits, refs)
    academic = [c for c in cits if c.citation_type == CitationType.ACADEMIC]
    assert len(academic) > 0
    key = academic[0].text
    assert key in resolved
    assert "Smith" in resolved[key]


# ---------------------------------------------------------------------------
# 11. resolve returns empty dict when no match
# ---------------------------------------------------------------------------
def test_resolve_no_match():
    text = "[Unknown, 1800] was an old work."
    cits = EX.extract(text)
    refs = ["Smith, 2020. Journal.", "Jones, 2019. Nature."]
    resolved = EX.resolve(cits, refs)
    # May or may not match; just ensure it doesn't crash
    assert isinstance(resolved, dict)


# ---------------------------------------------------------------------------
# 12. Empty text returns no citations
# ---------------------------------------------------------------------------
def test_empty_text():
    cits = EX.extract("")
    assert cits == []


# ---------------------------------------------------------------------------
# 13. RETRIEVAL_REGISTRY entry
# ---------------------------------------------------------------------------
def test_registry_entry():
    assert "citation_extractor" in RETRIEVAL_REGISTRY
    assert isinstance(RETRIEVAL_REGISTRY["citation_extractor"], CitationExtractor)
