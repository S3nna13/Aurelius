"""Tests for src/search/snippet_extractor.py  (>=28 tests)."""

from __future__ import annotations

import pytest

from src.search.snippet_extractor import (
    Snippet,
    SnippetConfig,
    SnippetExtractor,
    SNIPPET_EXTRACTOR_REGISTRY,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_exists(self):
        assert SNIPPET_EXTRACTOR_REGISTRY is not None

    def test_registry_has_default_key(self):
        assert "default" in SNIPPET_EXTRACTOR_REGISTRY

    def test_registry_default_is_class(self):
        assert SNIPPET_EXTRACTOR_REGISTRY["default"] is SnippetExtractor


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

class TestSnippetFrozen:
    def test_snippet_is_frozen(self):
        s = Snippet(text="hello", start_pos=0, end_pos=5, score=1.0, query_terms=["hello"])
        with pytest.raises((TypeError, AttributeError)):
            s.text = "changed"  # type: ignore[misc]

    def test_snippet_fields_accessible(self):
        s = Snippet(text="abc", start_pos=2, end_pos=5, score=2.0, query_terms=["a", "b"])
        assert s.text == "abc"
        assert s.start_pos == 2
        assert s.end_pos == 5
        assert s.score == 2.0
        assert s.query_terms == ["a", "b"]


class TestSnippetConfigFrozen:
    def test_config_is_frozen(self):
        cfg = SnippetConfig()
        with pytest.raises((TypeError, AttributeError)):
            cfg.window_size = 999  # type: ignore[misc]

    def test_config_defaults(self):
        cfg = SnippetConfig()
        assert cfg.window_size == 150
        assert cfg.max_snippets == 3
        assert cfg.merge_threshold == 50

    def test_config_custom_values(self):
        cfg = SnippetConfig(window_size=80, max_snippets=5, merge_threshold=20)
        assert cfg.window_size == 80
        assert cfg.max_snippets == 5
        assert cfg.merge_threshold == 20


# ---------------------------------------------------------------------------
# SnippetExtractor.extract
# ---------------------------------------------------------------------------

class TestExtractBasic:
    def test_extract_finds_term_in_doc(self):
        extractor = SnippetExtractor()
        doc = "The quick brown fox jumps over the lazy dog"
        snippets = extractor.extract(doc, ["fox"])
        assert len(snippets) >= 1
        assert "fox" in snippets[0].text

    def test_extract_empty_terms_returns_empty(self):
        extractor = SnippetExtractor()
        result = extractor.extract("some document text", [])
        assert result == []

    def test_extract_empty_document_returns_empty(self):
        extractor = SnippetExtractor()
        result = extractor.extract("", ["term"])
        assert result == []

    def test_extract_term_not_in_doc_returns_empty(self):
        extractor = SnippetExtractor()
        result = extractor.extract("hello world", ["python"])
        assert result == []

    def test_extract_case_insensitive(self):
        extractor = SnippetExtractor()
        doc = "The Quick Brown FOX jumps over the Lazy Dog"
        snippets = extractor.extract(doc, ["fox"])
        assert len(snippets) >= 1
        assert any("FOX" in s.text for s in snippets)

    def test_extract_case_insensitive_query_upper(self):
        extractor = SnippetExtractor()
        doc = "machine learning is powerful"
        snippets = extractor.extract(doc, ["LEARNING"])
        assert len(snippets) >= 1


class TestExtractWindowSize:
    def test_window_size_limits_snippet_length(self):
        cfg = SnippetConfig(window_size=30, max_snippets=1, merge_threshold=5)
        extractor = SnippetExtractor(cfg)
        doc = "a" * 200 + "TARGET" + "b" * 200
        snippets = extractor.extract(doc, ["target"])
        assert len(snippets) == 1
        assert len(snippets[0].text) <= 30 + len("target")  # slight slack for clamping

    def test_snippet_text_is_substring_of_document(self):
        extractor = SnippetExtractor()
        doc = "We study deep learning and neural networks."
        snippets = extractor.extract(doc, ["learning"])
        for s in snippets:
            assert doc[s.start_pos : s.end_pos] == s.text


class TestExtractMaxSnippets:
    def test_max_snippets_limits_output(self):
        cfg = SnippetConfig(window_size=10, max_snippets=2, merge_threshold=1)
        extractor = SnippetExtractor(cfg)
        # Build a doc where each hit is far apart so windows don't merge
        doc = " ".join(["alpha"] + ["x" * 50] * 10 + ["alpha"] + ["x" * 50] * 10 + ["alpha"])
        snippets = extractor.extract(doc, ["alpha"])
        assert len(snippets) <= 2

    def test_max_snippets_default_is_3(self):
        extractor = SnippetExtractor()
        # Create doc with many separated occurrences
        sep = "z" * 300
        doc = sep.join(["target"] * 10)
        snippets = extractor.extract(doc, ["target"])
        assert len(snippets) <= 3


class TestExtractScore:
    def test_score_counts_query_terms_covered(self):
        extractor = SnippetExtractor(SnippetConfig(window_size=200, max_snippets=5))
        doc = "alpha and beta together in one sentence"
        snippets = extractor.extract(doc, ["alpha", "beta"])
        # At least one snippet should cover both terms
        assert any(s.score >= 2.0 for s in snippets)

    def test_score_single_term_is_one(self):
        extractor = SnippetExtractor(SnippetConfig(window_size=50, max_snippets=5))
        doc = "alpha is here but beta is far " + "z" * 300 + " only alpha"
        snippets = extractor.extract(doc, ["alpha", "beta"])
        # The last occurrence only contains alpha, score should be 1
        single_term_snippets = [s for s in snippets if s.score == 1.0]
        assert len(single_term_snippets) >= 1

    def test_snippets_sorted_by_score_desc(self):
        extractor = SnippetExtractor(SnippetConfig(window_size=300, max_snippets=5))
        sep = "z" * 400
        doc = "alpha beta" + sep + "alpha"
        snippets = extractor.extract(doc, ["alpha", "beta"])
        scores = [s.score for s in snippets]
        assert scores == sorted(scores, reverse=True)


class TestExtractMerging:
    def test_overlapping_windows_are_merged(self):
        cfg = SnippetConfig(window_size=40, max_snippets=5, merge_threshold=20)
        extractor = SnippetExtractor(cfg)
        doc = "hello world hello"
        snippets = extractor.extract(doc, ["hello"])
        # Both 'hello' occurrences are close — should be merged into one snippet
        assert len(snippets) == 1

    def test_far_apart_windows_not_merged(self):
        cfg = SnippetConfig(window_size=10, max_snippets=5, merge_threshold=5)
        extractor = SnippetExtractor(cfg)
        sep = "x" * 200
        doc = "alpha" + sep + "alpha"
        snippets = extractor.extract(doc, ["alpha"])
        assert len(snippets) == 2


# ---------------------------------------------------------------------------
# SnippetExtractor.highlight
# ---------------------------------------------------------------------------

class TestHighlight:
    def test_highlight_wraps_term(self):
        extractor = SnippetExtractor()
        s = Snippet(text="hello world", start_pos=0, end_pos=11, score=1.0, query_terms=["world"])
        result = extractor.highlight(s)
        assert "**world**" in result

    def test_highlight_custom_marker(self):
        extractor = SnippetExtractor()
        s = Snippet(text="foo bar", start_pos=0, end_pos=7, score=1.0, query_terms=["foo"])
        result = extractor.highlight(s, marker="__")
        assert "__foo__" in result

    def test_highlight_case_insensitive(self):
        extractor = SnippetExtractor()
        s = Snippet(text="Hello World", start_pos=0, end_pos=11, score=1.0, query_terms=["hello"])
        result = extractor.highlight(s)
        assert "**Hello**" in result

    def test_highlight_multiple_terms(self):
        extractor = SnippetExtractor()
        s = Snippet(text="alpha and beta", start_pos=0, end_pos=14, score=2.0, query_terms=["alpha", "beta"])
        result = extractor.highlight(s)
        assert "**alpha**" in result
        assert "**beta**" in result

    def test_highlight_no_terms_unchanged(self):
        extractor = SnippetExtractor()
        s = Snippet(text="no matches here", start_pos=0, end_pos=15, score=0.0, query_terms=[])
        result = extractor.highlight(s)
        assert result == "no matches here"
