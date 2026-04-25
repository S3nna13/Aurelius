"""Tests for src/search/query_parser.py  (≥28 tests)."""

import pytest

from src.search.query_parser import (
    QueryParser,
    QueryToken,
    TokenType,
    QUERY_PARSER_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(query: str):
    return QueryParser().tokenize(query)


def _parse(query: str):
    return QueryParser().parse(query)


def _terms(query: str):
    return QueryParser().extract_terms(query)


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_exists(self):
        assert QUERY_PARSER_REGISTRY is not None

    def test_registry_default_key(self):
        assert "default" in QUERY_PARSER_REGISTRY

    def test_registry_default_is_query_parser_class(self):
        assert QUERY_PARSER_REGISTRY["default"] is QueryParser


# ---------------------------------------------------------------------------
# QueryToken dataclass
# ---------------------------------------------------------------------------

class TestQueryToken:
    def test_query_token_frozen(self):
        tok = QueryToken(type=TokenType.TERM, value="hello")
        with pytest.raises((AttributeError, TypeError)):
            tok.value = "other"  # type: ignore[misc]

    def test_query_token_equality(self):
        assert QueryToken(TokenType.TERM, "a") == QueryToken(TokenType.TERM, "a")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_single_term(self):
        tokens = _tokenize("hello")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.TERM
        assert tokens[0].value == "hello"

    def test_and_keyword(self):
        tokens = _tokenize("foo AND bar")
        types = [t.type for t in tokens]
        assert TokenType.AND in types

    def test_or_keyword(self):
        tokens = _tokenize("foo OR bar")
        types = [t.type for t in tokens]
        assert TokenType.OR in types

    def test_not_keyword(self):
        tokens = _tokenize("NOT foo")
        assert tokens[0].type == TokenType.NOT

    def test_quoted_phrase(self):
        tokens = _tokenize('"hello world"')
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.PHRASE
        assert tokens[0].value == "hello world"

    def test_left_paren(self):
        tokens = _tokenize("(foo)")
        assert tokens[0].type == TokenType.LPAREN

    def test_right_paren(self):
        tokens = _tokenize("(foo)")
        assert tokens[-1].type == TokenType.RPAREN

    def test_multiple_terms(self):
        tokens = _tokenize("a b c")
        assert all(t.type == TokenType.TERM for t in tokens)
        assert [t.value for t in tokens] == ["a", "b", "c"]

    def test_mixed_tokens(self):
        tokens = _tokenize('cat AND "big dog" OR NOT fish')
        types = [t.type for t in tokens]
        assert TokenType.PHRASE in types
        assert TokenType.AND in types
        assert TokenType.OR in types
        assert TokenType.NOT in types

    def test_empty_string_tokenizes_to_empty_list(self):
        assert _tokenize("") == []

    def test_whitespace_only_tokenizes_to_empty_list(self):
        assert _tokenize("   ") == []


# ---------------------------------------------------------------------------
# Parser – AST structure
# ---------------------------------------------------------------------------

class TestParseSingleTerm:
    def test_single_term_ast(self):
        ast = _parse("foo")
        assert ast["op"] == "TERM"
        assert ast["value"] == "foo"


class TestParseAND:
    def test_explicit_and(self):
        ast = _parse("foo AND bar")
        assert ast["op"] == "AND"
        assert len(ast["operands"]) == 2
        assert ast["operands"][0]["value"] == "foo"
        assert ast["operands"][1]["value"] == "bar"

    def test_and_both_operands_are_terms(self):
        ast = _parse("x AND y")
        for operand in ast["operands"]:
            assert operand["op"] == "TERM"


class TestParseOR:
    def test_explicit_or(self):
        ast = _parse("foo OR bar")
        assert ast["op"] == "OR"
        assert len(ast["operands"]) == 2

    def test_or_operands_values(self):
        ast = _parse("alpha OR beta")
        vals = {op["value"] for op in ast["operands"]}
        assert vals == {"alpha", "beta"}


class TestParseNOT:
    def test_not_wraps_term(self):
        ast = _parse("NOT foo")
        assert ast["op"] == "NOT"
        assert ast["operand"]["op"] == "TERM"
        assert ast["operand"]["value"] == "foo"

    def test_double_not(self):
        ast = _parse("NOT NOT foo")
        assert ast["op"] == "NOT"
        assert ast["operand"]["op"] == "NOT"


class TestParseImplicitAND:
    def test_two_terms_implicit_and(self):
        ast = _parse("hello world")
        assert ast["op"] == "AND"
        assert ast["operands"][0]["value"] == "hello"
        assert ast["operands"][1]["value"] == "world"

    def test_three_terms_implicit_and(self):
        ast = _parse("a b c")
        # Should be left-associative: (a AND b) AND c
        assert ast["op"] == "AND"


class TestParsePhrase:
    def test_phrase_ast(self):
        ast = _parse('"exact phrase"')
        assert ast["op"] == "PHRASE"
        assert ast["value"] == "exact phrase"

    def test_phrase_in_and_query(self):
        ast = _parse('"machine learning" AND python')
        assert ast["op"] == "AND"
        phrase_node = ast["operands"][0]
        assert phrase_node["op"] == "PHRASE"
        assert phrase_node["value"] == "machine learning"


class TestParseParens:
    def test_parens_grouping(self):
        ast = _parse("(foo OR bar) AND baz")
        assert ast["op"] == "AND"
        left = ast["operands"][0]
        assert left["op"] == "OR"

    def test_nested_parens(self):
        ast = _parse("((foo))")
        assert ast["op"] == "TERM"
        assert ast["value"] == "foo"


class TestParseEmpty:
    def test_empty_query_no_crash(self):
        ast = _parse("")
        assert "op" in ast  # returns something sensible

    def test_empty_query_op_value(self):
        ast = _parse("")
        assert ast["op"] == "EMPTY"


# ---------------------------------------------------------------------------
# extract_terms
# ---------------------------------------------------------------------------

class TestExtractTerms:
    def test_single_term_extracted(self):
        assert _terms("Hello") == ["hello"]

    def test_and_query_extracts_both(self):
        result = _terms("foo AND bar")
        assert sorted(result) == ["bar", "foo"]

    def test_or_query_extracts_both(self):
        result = _terms("cat OR dog")
        assert sorted(result) == ["cat", "dog"]

    def test_not_query_extracts_term(self):
        result = _terms("NOT spam")
        assert result == ["spam"]

    def test_phrase_lowercased(self):
        result = _terms('"Hello World"')
        assert result == ["hello world"]

    def test_implicit_and_extracts_all(self):
        result = _terms("a b c")
        assert sorted(result) == ["a", "b", "c"]

    def test_extract_terms_empty_query(self):
        result = _terms("")
        assert result == []
