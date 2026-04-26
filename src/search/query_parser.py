"""Aurelius search – Boolean query parser (AND / OR / NOT / phrases / parens)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


class TokenType(Enum):
    TERM = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    LPAREN = auto()
    RPAREN = auto()
    PHRASE = auto()


@dataclass(frozen=True)
class QueryToken:
    type: TokenType
    value: str


_KEYWORD_MAP = {
    "AND": TokenType.AND,
    "OR": TokenType.OR,
    "NOT": TokenType.NOT,
}


# ---------------------------------------------------------------------------
# Parser AST helpers
# ---------------------------------------------------------------------------


def _make_term(value: str) -> dict[str, Any]:
    return {"op": "TERM", "value": value}


def _make_phrase(value: str) -> dict[str, Any]:
    return {"op": "PHRASE", "value": value}


def _make_and(left: dict, right: dict) -> dict[str, Any]:
    return {"op": "AND", "operands": [left, right]}


def _make_or(left: dict, right: dict) -> dict[str, Any]:
    return {"op": "OR", "operands": [left, right]}


def _make_not(child: dict) -> dict[str, Any]:
    return {"op": "NOT", "operand": child}


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------
# Grammar (after tokenisation):
#
#   expr     → or_expr
#   or_expr  → and_expr  ( OR and_expr )*
#   and_expr → not_expr  ( (AND | implicit) not_expr )*
#   not_expr → NOT not_expr | primary
#   primary  → LPAREN expr RPAREN | TERM | PHRASE
#
# "implicit AND" = two primaries in a row with no explicit operator.


class _Parser:
    def __init__(self, tokens: list[QueryToken]) -> None:
        self._tokens = tokens
        self._pos = 0

    # ---- internal helpers -------------------------------------------------

    def _peek(self) -> QueryToken | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self) -> QueryToken:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, ttype: TokenType) -> QueryToken:
        tok = self._peek()
        if tok is None or tok.type != ttype:
            raise SyntaxError(f"Expected {ttype} but got {tok!r} at position {self._pos}")
        return self._consume()

    # ---- grammar rules ----------------------------------------------------

    def parse(self) -> dict[str, Any] | None:
        if not self._tokens:
            return {"op": "EMPTY", "value": ""}
        node = self._or_expr()
        return node

    def _or_expr(self) -> dict[str, Any]:
        left = self._and_expr()
        while True:
            tok = self._peek()
            if tok is not None and tok.type == TokenType.OR:
                self._consume()  # eat OR
                right = self._and_expr()
                left = _make_or(left, right)
            else:
                break
        return left

    def _and_expr(self) -> dict[str, Any]:
        left = self._not_expr()
        while True:
            tok = self._peek()
            if tok is None:
                break
            if tok.type == TokenType.AND:
                self._consume()  # eat AND
                right = self._not_expr()
                left = _make_and(left, right)
            elif tok.type in (TokenType.TERM, TokenType.PHRASE, TokenType.NOT, TokenType.LPAREN):
                # implicit AND
                right = self._not_expr()
                left = _make_and(left, right)
            else:
                break
        return left

    def _not_expr(self) -> dict[str, Any]:
        tok = self._peek()
        if tok is not None and tok.type == TokenType.NOT:
            self._consume()
            child = self._not_expr()
            return _make_not(child)
        return self._primary()

    def _primary(self) -> dict[str, Any]:
        tok = self._peek()
        if tok is None:
            raise SyntaxError("Unexpected end of query")
        if tok.type == TokenType.LPAREN:
            self._consume()
            node = self._or_expr()
            # consume RPAREN if present; be lenient about missing closing paren
            tok = self._peek()
            if tok is not None and tok.type == TokenType.RPAREN:
                self._consume()
            return node
        if tok.type == TokenType.TERM:
            self._consume()
            return _make_term(tok.value)
        if tok.type == TokenType.PHRASE:
            self._consume()
            return _make_phrase(tok.value)
        raise SyntaxError(f"Unexpected token {tok!r} at position {self._pos}")


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class QueryParser:
    """Parse Boolean queries into AST dicts."""

    # ---- tokeniser --------------------------------------------------------

    def tokenize(self, query: str) -> list[QueryToken]:
        """Split *query* into :class:`QueryToken` objects."""
        tokens: list[QueryToken] = []
        i = 0
        query = query.strip()
        while i < len(query):
            ch = query[i]

            # Skip whitespace
            if ch.isspace():
                i += 1
                continue

            # Quoted phrase
            if ch == '"':
                end = query.find('"', i + 1)
                if end == -1:
                    # Unterminated quote – treat rest as phrase
                    phrase = query[i + 1 :]
                    i = len(query)
                else:
                    phrase = query[i + 1 : end]
                    i = end + 1
                tokens.append(QueryToken(type=TokenType.PHRASE, value=phrase))
                continue

            # Parentheses
            if ch == "(":
                tokens.append(QueryToken(type=TokenType.LPAREN, value="("))
                i += 1
                continue
            if ch == ")":
                tokens.append(QueryToken(type=TokenType.RPAREN, value=")"))
                i += 1
                continue

            # Word token (term or keyword)
            match = re.match(r'[^\s"()]+', query[i:])
            if match:
                word = match.group(0)
                i += len(word)
                ttype = _KEYWORD_MAP.get(word, TokenType.TERM)
                tokens.append(QueryToken(type=ttype, value=word))
                continue

            # Fallback: skip unknown character
            i += 1

        return tokens

    # ---- parser -----------------------------------------------------------

    def parse(self, query: str) -> dict[str, Any]:
        """Parse *query* into an AST dict.

        Returns ``{"op": "EMPTY", "value": ""}`` for an empty/blank query.
        """
        tokens = self.tokenize(query)
        parser = _Parser(tokens)
        result = parser.parse()
        if result is None:
            return {"op": "EMPTY", "value": ""}
        return result

    # ---- term extraction --------------------------------------------------

    def extract_terms(self, query: str) -> list[str]:
        """Return all leaf TERM/PHRASE values (lowercased) from *query*."""
        ast = self.parse(query)
        terms: list[str] = []
        self._collect_terms(ast, terms)
        return terms

    def _collect_terms(self, node: dict[str, Any], out: list[str]) -> None:
        op = node.get("op")
        if op in ("TERM", "PHRASE"):
            out.append(node["value"].lower())
        elif op in ("AND", "OR"):
            for child in node.get("operands", []):
                self._collect_terms(child, out)
        elif op == "NOT":
            child = node.get("operand")
            if child:
                self._collect_terms(child, out)
        # EMPTY and unknown ops produce nothing


QUERY_PARSER_REGISTRY: dict[str, type] = {"default": QueryParser}
