"""Aurelius search – snippet extraction around query-term hits in documents."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Snippet:
    """A single extracted text snippet with position and relevance metadata."""

    text: str
    start_pos: int
    end_pos: int
    score: float
    query_terms: list[str]


@dataclass(frozen=True)
class SnippetConfig:
    """Configuration for :class:`SnippetExtractor`."""

    window_size: int = 150  # characters on either side of the hit centre
    max_snippets: int = 3
    merge_threshold: int = 50  # merge windows within this many chars of each other


class SnippetExtractor:
    """Extracts and scores relevant text snippets from a document."""

    def __init__(self, config: SnippetConfig | None = None) -> None:
        self.config = config or SnippetConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, document: str, query_terms: list[str]) -> list[Snippet]:
        """Return up to *max_snippets* scored snippets covering query-term hits.

        Algorithm
        ---------
        1. For every unique, non-empty query term find all case-insensitive
           positions in *document*.
        2. Build a candidate window (start, end) of *window_size* chars
           centred on each hit mid-point, clamped to document bounds.
        3. Merge windows whose ranges overlap or are within *merge_threshold*
           chars of each other.
        4. Score each merged window by the number of distinct query terms
           that appear inside it.
        5. Return the top *max_snippets* windows sorted by score descending.
        """
        if not query_terms or not document:
            return []

        cfg = self.config
        doc_len = len(document)
        doc_lower = document.lower()

        # Step 1 & 2 – collect raw windows
        raw_windows: list[tuple[int, int]] = []
        unique_terms = [t for t in dict.fromkeys(t.lower() for t in query_terms if t)]
        if not unique_terms:
            return []

        for term in unique_terms:
            start = 0
            while True:
                pos = doc_lower.find(term, start)
                if pos == -1:
                    break
                hit_mid = pos + len(term) // 2
                win_start = max(0, hit_mid - cfg.window_size // 2)
                win_end = min(doc_len, win_start + cfg.window_size)
                # Re-clamp start after end adjustment
                win_start = max(0, win_end - cfg.window_size)
                raw_windows.append((win_start, win_end))
                start = pos + 1

        if not raw_windows:
            return []

        # Step 3 – merge overlapping / near-adjacent windows
        raw_windows.sort()
        merged: list[tuple[int, int]] = []
        cur_s, cur_e = raw_windows[0]
        for s, e in raw_windows[1:]:
            if s <= cur_e + cfg.merge_threshold:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        # Step 4 – score each merged window
        scored: list[Snippet] = []
        for ws, we in merged:
            chunk = document[ws:we]
            chunk_lower = chunk.lower()
            covered = [t for t in unique_terms if t in chunk_lower]
            score = float(len(covered))
            scored.append(
                Snippet(
                    text=chunk,
                    start_pos=ws,
                    end_pos=we,
                    score=score,
                    query_terms=covered,
                )
            )

        # Step 5 – top max_snippets by score desc
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[: cfg.max_snippets]

    def highlight(self, snippet: Snippet, marker: str = "**") -> str:
        """Wrap each query term in *snippet.text* with *marker*.

        E.g. with marker="**": "hello world" → "hello **world**"
        The replacement is case-insensitive; original casing is preserved.
        """
        text = snippet.text
        for term in snippet.query_terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text = pattern.sub(lambda m: f"{marker}{m.group()}{marker}", text)
        return text


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SNIPPET_EXTRACTOR_REGISTRY: dict[str, type] = {"default": SnippetExtractor}
