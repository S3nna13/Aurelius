"""Tests for src/search/web_search.py (~50 tests)."""

from __future__ import annotations

from src.search.web_search import (
    WEB_SEARCH,
    SearchQuery,
    SearchResult,
    WebSearchStub,
)

# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_auto_id_generated(self):
        r = SearchResult(title="T", url="http://x.com", snippet="s")
        assert r.id is not None
        assert isinstance(r.id, str)
        assert len(r.id) == 8

    def test_auto_id_unique(self):
        r1 = SearchResult(title="T", url="http://x.com", snippet="s")
        r2 = SearchResult(title="T", url="http://x.com", snippet="s")
        assert r1.id != r2.id

    def test_default_score(self):
        r = SearchResult(title="T", url="http://x.com", snippet="s")
        assert r.score == 1.0

    def test_default_source(self):
        r = SearchResult(title="T", url="http://x.com", snippet="s")
        assert r.source == "web"

    def test_custom_score(self):
        r = SearchResult(title="T", url="http://x.com", snippet="s", score=0.5)
        assert r.score == 0.5

    def test_custom_source(self):
        r = SearchResult(title="T", url="http://x.com", snippet="s", source="api")
        assert r.source == "api"

    def test_title_field(self):
        r = SearchResult(title="Hello", url="http://x.com", snippet="s")
        assert r.title == "Hello"

    def test_url_field(self):
        r = SearchResult(title="T", url="http://example.com", snippet="s")
        assert r.url == "http://example.com"

    def test_snippet_field(self):
        r = SearchResult(title="T", url="http://x.com", snippet="some snippet")
        assert r.snippet == "some snippet"

    def test_explicit_id(self):
        r = SearchResult(title="T", url="http://x.com", snippet="s", id="abc12345")
        assert r.id == "abc12345"


# ---------------------------------------------------------------------------
# SearchQuery
# ---------------------------------------------------------------------------


class TestSearchQuery:
    def test_query_field(self):
        q = SearchQuery(query="hello")
        assert q.query == "hello"

    def test_default_max_results(self):
        q = SearchQuery(query="hello")
        assert q.max_results == 10

    def test_default_language(self):
        q = SearchQuery(query="hello")
        assert q.language == "en"

    def test_default_safe_search(self):
        q = SearchQuery(query="hello")
        assert q.safe_search is True

    def test_custom_max_results(self):
        q = SearchQuery(query="hello", max_results=5)
        assert q.max_results == 5

    def test_custom_language(self):
        q = SearchQuery(query="hello", language="fr")
        assert q.language == "fr"

    def test_custom_safe_search(self):
        q = SearchQuery(query="hello", safe_search=False)
        assert q.safe_search is False


# ---------------------------------------------------------------------------
# WebSearchStub
# ---------------------------------------------------------------------------


class TestWebSearchStubInit:
    def test_default_has_five_results(self):
        stub = WebSearchStub()
        assert len(stub._seeds) == 5

    def test_custom_seed_results(self):
        seeds = [SearchResult(title="Custom", url="http://c.com", snippet="custom")]
        stub = WebSearchStub(seed_results=seeds)
        assert len(stub._seeds) == 1

    def test_none_uses_defaults(self):
        stub = WebSearchStub(seed_results=None)
        assert len(stub._seeds) == 5


class TestNormalizeQuery:
    def setup_method(self):
        self.stub = WebSearchStub()

    def test_lowercases(self):
        assert self.stub.normalize_query("HELLO") == "hello"

    def test_strips_leading_whitespace(self):
        assert self.stub.normalize_query("  hello") == "hello"

    def test_strips_trailing_whitespace(self):
        assert self.stub.normalize_query("hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        assert self.stub.normalize_query("hello   world") == "hello world"

    def test_combined(self):
        assert self.stub.normalize_query("  HELLO   WORLD  ") == "hello world"

    def test_empty_string(self):
        assert self.stub.normalize_query("") == ""

    def test_single_word(self):
        assert self.stub.normalize_query("Python") == "python"


class TestSearch:
    def setup_method(self):
        self.stub = WebSearchStub()

    def test_returns_list(self):
        q = SearchQuery(query="python")
        result = self.stub.search(q)
        assert isinstance(result, list)

    def test_results_are_search_results(self):
        q = SearchQuery(query="python")
        results = self.stub.search(q)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_python_query_finds_python_result(self):
        q = SearchQuery(query="python")
        results = self.stub.search(q)
        titles = [r.title.lower() for r in results]
        assert any("python" in t for t in titles)

    def test_machine_learning_query_finds_ml_result(self):
        q = SearchQuery(query="machine learning")
        results = self.stub.search(q)
        combined = " ".join((r.title + r.snippet).lower() for r in results)
        assert "machine" in combined or "learning" in combined

    def test_transformers_query_finds_result(self):
        q = SearchQuery(query="transformers")
        results = self.stub.search(q)
        combined = " ".join((r.title + r.snippet).lower() for r in results)
        assert "transformer" in combined

    def test_max_results_limits_output(self):
        q = SearchQuery(query="python", max_results=1)
        results = self.stub.search(q)
        assert len(results) <= 1

    def test_max_results_zero_returns_empty(self):
        q = SearchQuery(query="python", max_results=0)
        results = self.stub.search(q)
        assert results == []

    def test_results_sorted_by_score_desc(self):
        q = SearchQuery(query="python")
        results = self.stub.search(q)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_match_returns_all(self):
        q = SearchQuery(query="xyzzy_nonexistent_query_abc")
        results = self.stub.search(q)
        assert len(results) == 5

    def test_no_match_sorted_by_score(self):
        q = SearchQuery(query="xyzzy_nonexistent_query_abc")
        results = self.stub.search(q)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_case_insensitive_matching(self):
        q = SearchQuery(query="PYTHON")
        results = self.stub.search(q)
        combined = " ".join((r.title + r.snippet).lower() for r in results)
        assert "python" in combined

    def test_max_results_respected_with_no_match(self):
        q = SearchQuery(query="xyzzy_nonexistent", max_results=3)
        results = self.stub.search(q)
        assert len(results) <= 3


class TestTopK:
    def setup_method(self):
        self.stub = WebSearchStub()
        self.results = [
            SearchResult(title=f"R{i}", url=f"http://{i}.com", snippet="s", score=float(i))
            for i in range(5)
        ]

    def test_top_k_returns_first_k(self):
        out = self.stub.top_k(self.results, 3)
        assert len(out) == 3

    def test_top_k_correct_elements(self):
        out = self.stub.top_k(self.results, 2)
        assert out == self.results[:2]

    def test_top_k_zero_returns_empty(self):
        out = self.stub.top_k(self.results, 0)
        assert out == []

    def test_top_k_larger_than_list(self):
        out = self.stub.top_k(self.results, 100)
        assert out == self.results


class TestGlobalInstance:
    def test_web_search_exists(self):
        assert WEB_SEARCH is not None

    def test_web_search_is_stub(self):
        assert isinstance(WEB_SEARCH, WebSearchStub)

    def test_web_search_can_search(self):
        results = WEB_SEARCH.search(SearchQuery(query="python"))
        assert isinstance(results, list)
