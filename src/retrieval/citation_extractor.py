"""Citation extraction for Aurelius retrieval pipeline.

Extracts inline citations of several types from free text and resolves them
against a reference list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

RETRIEVAL_REGISTRY: dict = {}


class CitationType(StrEnum):
    ACADEMIC = "academic"
    URL = "url"
    BOOK = "book"
    INLINE = "inline"


@dataclass
class Citation:
    citation_id: str
    text: str
    source: str
    citation_type: CitationType
    start: int
    end: int


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# ACADEMIC: "[Author, Year]" or "(Author et al., Year)"
_RE_ACADEMIC = re.compile(
    r"(?:"
    r"\[(?P<bk>[A-Z][^,\[\]]+?,\s*\d{4}[a-z]?)\]"
    r"|"
    r"\((?P<pk>[A-Z][^()]+?(?:et al\.)?[^()]*?,\s*\d{4}[a-z]?)\)"
    r")"
)

# URL: http(s)://...
_RE_URL = re.compile(r"https?://[^\s<>\"\']+")

# BOOK: "Title" by Author
_RE_BOOK = re.compile(r'"(?P<title>[^"]{2,}?)"\s+by\s+(?P<author>[A-Z][^.,\n]{1,60})')

# INLINE footnote: [1], [2], [iv], [i]
_RE_INLINE = re.compile(r"\[(?P<ref>\d+|[ivxlcdmIVXLCDM]+)\]")


class CitationExtractor:
    """Extracts and resolves citations from text."""

    def extract(self, text: str) -> list[Citation]:
        """Return all citations found in *text*, ordered by position."""
        citations: list[Citation] = []
        cid_counter = 0

        def _next_id() -> str:
            nonlocal cid_counter
            cid_counter += 1
            return f"cit_{cid_counter}"

        # -- ACADEMIC --
        for m in _RE_ACADEMIC.finditer(text):
            raw = m.group("bk") or m.group("pk")
            citations.append(
                Citation(
                    citation_id=_next_id(),
                    text=m.group(0),
                    source=raw.strip(),
                    citation_type=CitationType.ACADEMIC,
                    start=m.start(),
                    end=m.end(),
                )
            )

        # -- URL --
        for m in _RE_URL.finditer(text):
            citations.append(
                Citation(
                    citation_id=_next_id(),
                    text=m.group(0),
                    source=m.group(0),
                    citation_type=CitationType.URL,
                    start=m.start(),
                    end=m.end(),
                )
            )

        # -- BOOK --
        for m in _RE_BOOK.finditer(text):
            source = f'"{m.group("title")}" by {m.group("author").strip()}'
            citations.append(
                Citation(
                    citation_id=_next_id(),
                    text=m.group(0),
                    source=source,
                    citation_type=CitationType.BOOK,
                    start=m.start(),
                    end=m.end(),
                )
            )

        # -- INLINE -- skip those already matched as ACADEMIC (e.g. [1])
        academic_ranges = {
            (c.start, c.end) for c in citations if c.citation_type == CitationType.ACADEMIC
        }
        for m in _RE_INLINE.finditer(text):
            if (m.start(), m.end()) in academic_ranges:
                continue
            citations.append(
                Citation(
                    citation_id=_next_id(),
                    text=m.group(0),
                    source=m.group("ref"),
                    citation_type=CitationType.INLINE,
                    start=m.start(),
                    end=m.end(),
                )
            )

        citations.sort(key=lambda c: c.start)
        return citations

    def resolve(
        self,
        citations: list[Citation],
        reference_list: list[str],
    ) -> dict[str, str]:
        """Map citation.text -> matching reference string via substring match.

        Returns a dict keyed by citation text; value is the first reference that
        contains the citation source as a substring (case-insensitive).  If no
        reference matches, the key is omitted.
        """
        resolved: dict[str, str] = {}
        for cit in citations:
            needle = cit.source.lower()
            for ref in reference_list:
                if needle in ref.lower():
                    resolved[cit.text] = ref
                    break
        return resolved


RETRIEVAL_REGISTRY["citation_extractor"] = CitationExtractor()
