"""Query expansion for code and natural-language search.

Splits identifiers (camelCase/snake_case), generates n-gram variants,
and applies optional synonym substitution.
Inspired by code-aware tokenization in Qwen2.5-Coder (Apache-2.0) and
DeepSeek-Coder (MIT); Aurelius-native clean-room. License: MIT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MAX_QUERY_LEN = 2048
_MAX_SYNONYMS = 256
_MAX_EXPANSIONS = 64


@dataclass
class ExpandedQuery:
    original: str
    tokens: list[str]  # tokenized original
    variants: list[str]  # all query string variants
    synonyms_applied: int = 0


class QueryExpander:
    """Expands queries by identifier splitting and synonym substitution."""

    def __init__(
        self, synonyms: dict[str, list[str]] | None = None, max_expansions: int = 16
    ) -> None:
        if max_expansions > _MAX_EXPANSIONS:
            raise ValueError(f"max_expansions exceeds {_MAX_EXPANSIONS}")
        self.max_expansions = max_expansions
        self._synonyms: dict[str, list[str]] = {}
        if synonyms:
            self.add_synonyms(synonyms)

    def add_synonyms(self, synonyms: dict[str, list[str]]) -> None:
        """Register synonym mappings. Raises ValueError if total exceeds _MAX_SYNONYMS."""
        if len(self._synonyms) + len(synonyms) > _MAX_SYNONYMS:
            raise ValueError(f"synonym count would exceed {_MAX_SYNONYMS}")
        for k, vs in synonyms.items():
            self._synonyms[k.lower()] = [v.lower() for v in vs]

    def split_identifier(self, token: str) -> list[str]:
        """Split camelCase and snake_case identifiers into sub-tokens."""
        # snake_case split
        parts = token.split("_")
        result = []
        for part in parts:
            if not part:
                continue
            # camelCase split: insert space before uppercase preceded by lowercase
            sub = re.sub(r"([a-z])([A-Z])", r"\1 \2", part)
            sub = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", sub)
            result.extend(sub.lower().split())
        return [t for t in result if t]

    def tokenize(self, query: str) -> list[str]:
        """Tokenize query into words, splitting identifiers."""
        raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*|[0-9]+", query)
        result = []
        for tok in raw_tokens:
            if "_" in tok or re.search(r"[a-z][A-Z]", tok):
                result.extend(self.split_identifier(tok))
            else:
                result.append(tok.lower())
        return result

    def expand(self, query: str) -> ExpandedQuery:
        """Expand a query into variants."""
        if len(query) > _MAX_QUERY_LEN:
            raise ValueError(f"query exceeds {_MAX_QUERY_LEN} chars")
        tokens = self.tokenize(query)
        variants: list[str] = [query]
        synonyms_applied = 0

        # Add joined token string as a variant
        joined = " ".join(tokens)
        if joined != query and joined not in variants:
            variants.append(joined)

        # Synonym substitution
        for i, tok in enumerate(tokens):
            if tok in self._synonyms:
                for syn in self._synonyms[tok]:
                    new_tokens = tokens[:i] + [syn] + tokens[i + 1 :]
                    v = " ".join(new_tokens)
                    if v not in variants:
                        variants.append(v)
                        synonyms_applied += 1
                    if len(variants) >= self.max_expansions:
                        break
            if len(variants) >= self.max_expansions:
                break

        # Add individual tokens as standalone variants
        for tok in tokens:
            if tok not in variants and len(variants) < self.max_expansions:
                variants.append(tok)

        return ExpandedQuery(
            original=query,
            tokens=tokens,
            variants=variants[: self.max_expansions],
            synonyms_applied=synonyms_applied,
        )


QUERY_EXPANDER = QueryExpander()
