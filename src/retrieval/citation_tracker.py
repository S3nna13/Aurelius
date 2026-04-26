"""Citation tracking for RAG outputs.

Given an LLM output and the set of retrieved sources, compute a citation
audit trail: which output spans are supported by which source spans, and
which segments of the output are uncited (potential hallucinations).

Pure-stdlib (``dataclasses``, ``re``). No foreign imports. Deterministic.

The primary entry point is :class:`CitationTracker`.  Its ``track`` method
scans each source for exact substring matches (length >= ``min_exact_len``)
against the output text, records a :class:`CitationSpan` for each match,
and computes a coverage ratio plus the list of uncited runs of length
>= ``min_uncited_run``.  An optional ``similarity_fn`` may be supplied to
enable a fuzzy-match fallback for sources that have no exact-match
evidence.  Caller-declared citations (e.g. model-emitted ``[1]`` markers
resolved to source ids) can be merged via
:meth:`CitationTracker.track_with_declared`; when a declared citation
conflicts with an exact match for the same output region, the exact
match wins on confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_MIN_EXACT_LEN: int = 20
DEFAULT_MIN_UNCITED_RUN: int = 40

_METHODS = ("exact_match", "fuzzy_match", "semantic_match", "declared")


@dataclass(frozen=True)
class Source:
    """A retrieved source document under consideration for citation."""

    id: str
    text: str
    origin: str
    retrieved_at: str


@dataclass(frozen=True)
class CitationSpan:
    """A supported output region paired with its source evidence span."""

    output_start: int
    output_end: int
    source_id: str
    source_start: int
    source_end: int
    confidence: float
    method: str


@dataclass(frozen=True)
class CitationReport:
    """Audit artefact produced by :class:`CitationTracker`."""

    sources: tuple[Source, ...]
    spans: tuple[CitationSpan, ...]
    coverage_ratio: float
    uncited_segments: tuple[tuple[int, int], ...] = field(default_factory=tuple)


def _find_all_occurrences(haystack: str, needle: str) -> list[int]:
    """Return all (possibly overlapping-safe) start indices of ``needle``."""

    if not needle:
        return []
    out: list[int] = []
    start = 0
    step = max(1, len(needle))
    while True:
        i = haystack.find(needle, start)
        if i < 0:
            break
        out.append(i)
        # non-overlapping advance, matches re.finditer semantics for fixed
        # literal strings and keeps behaviour deterministic.
        start = i + step
    return out


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent half-open [start, end) intervals."""

    if not intervals:
        return []
    sorted_iv = sorted(intervals)
    merged: list[tuple[int, int]] = [sorted_iv[0]]
    for s, e in sorted_iv[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _covered_length(intervals: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in _merge_intervals(intervals))


def _uncited_runs(
    total_len: int,
    intervals: list[tuple[int, int]],
    min_run: int,
) -> list[tuple[int, int]]:
    if total_len <= 0:
        return []
    merged = _merge_intervals(intervals)
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for s, e in merged:
        if s > cursor:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total_len:
        gaps.append((cursor, total_len))
    return [(s, e) for s, e in gaps if (e - s) >= min_run]


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _sentence_windows(text: str) -> list[tuple[int, int]]:
    """Yield rough (start, end) windows for sentence-like chunks."""

    if not text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"[.!?\n]+", text):
        end = m.end()
        if end > start:
            spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


@dataclass
class CitationTracker:
    """Stateless citation tracker.

    Parameters
    ----------
    similarity_fn:
        Optional callable ``(a, b) -> float in [0, 1]``.  When provided,
        sources that yield zero exact matches are checked against output
        sentence windows; the best-scoring window above ``fuzzy_threshold``
        becomes a ``fuzzy_match`` citation.
    min_exact_len:
        Minimum length (chars) for an exact substring match to be recorded.
    min_uncited_run:
        Minimum contiguous uncited run (chars) to report.
    fuzzy_threshold:
        Score threshold for ``similarity_fn`` matches.
    """

    similarity_fn: object | None = None
    min_exact_len: int = DEFAULT_MIN_EXACT_LEN
    min_uncited_run: int = DEFAULT_MIN_UNCITED_RUN
    fuzzy_threshold: float = 0.75

    # ------------------------------------------------------------------ core

    def _exact_spans_for_source(self, output_text: str, src: Source) -> list[CitationSpan]:
        if not src.text or not output_text:
            return []
        # Greedy longest-match-first: enumerate candidate substrings of the
        # source in descending length, but cap to a reasonable working set
        # by scanning output for each source substring of length >= min.
        # Practical implementation: slide the *source* as the needle would
        # be too slow; instead enumerate output substrings of length equal
        # to min_exact_len, then grow.
        spans: list[CitationSpan] = []
        min_len = max(1, self.min_exact_len)
        # Strategy: for every position in output, find the longest prefix
        # that also appears as a substring of the source, then if its
        # length >= min_len record a span and skip past it.
        i = 0
        n = len(output_text)
        src_text = src.text
        while i < n:
            # find the longest j such that output_text[i:j] in src_text.
            # start by checking [i:i+min_len]; if absent, advance by 1.
            if i + min_len > n:
                break
            window = output_text[i : i + min_len]
            src_idx = src_text.find(window)
            if src_idx < 0:
                i += 1
                continue
            # Extend greedily.
            j = i + min_len
            while j < n:
                ext = output_text[i : j + 1]
                nxt = src_text.find(ext)
                if nxt < 0:
                    break
                src_idx = nxt
                j += 1
            spans.append(
                CitationSpan(
                    output_start=i,
                    output_end=j,
                    source_id=src.id,
                    source_start=src_idx,
                    source_end=src_idx + (j - i),
                    confidence=1.0,
                    method="exact_match",
                )
            )
            i = j
        return spans

    def _fuzzy_spans_for_source(self, output_text: str, src: Source) -> list[CitationSpan]:
        if self.similarity_fn is None or not src.text or not output_text:
            return []
        best: tuple[float, int, int] | None = None
        for s, e in _sentence_windows(output_text):
            chunk = output_text[s:e].strip()
            if len(chunk) < self.min_exact_len:
                continue
            try:
                score = float(self.similarity_fn(chunk, src.text))  # type: ignore[misc]
            except Exception:
                score = 0.0
            if score >= self.fuzzy_threshold and (best is None or score > best[0]):
                best = (score, s, e)
        if best is None:
            return []
        score, s, e = best
        return [
            CitationSpan(
                output_start=s,
                output_end=e,
                source_id=src.id,
                source_start=0,
                source_end=len(src.text),
                confidence=score,
                method="fuzzy_match",
            )
        ]

    def track(self, output_text: str, sources: list[Source]) -> CitationReport:
        spans: list[CitationSpan] = []
        for src in sources:
            src_spans = self._exact_spans_for_source(output_text, src)
            if not src_spans:
                src_spans = self._fuzzy_spans_for_source(output_text, src)
            spans.extend(src_spans)
        return self._finalize(output_text, tuple(sources), spans)

    def track_with_declared(
        self,
        output_text: str,
        sources: list[Source],
        declared_cites: list[dict],
    ) -> CitationReport:
        # First collect exact/fuzzy evidence as in ``track``.
        evidence: list[CitationSpan] = []
        for src in sources:
            src_spans = self._exact_spans_for_source(output_text, src)
            if not src_spans:
                src_spans = self._fuzzy_spans_for_source(output_text, src)
            evidence.extend(src_spans)
        # Build an interval tree of regions already covered by exact matches;
        # a declared citation is suppressed if it fully overlaps an exact
        # match (exact wins by confidence). Partial-overlap declared spans
        # are kept as-is.
        exact_intervals = _merge_intervals(
            [(s.output_start, s.output_end) for s in evidence if s.method == "exact_match"]
        )
        src_by_id = {s.id: s for s in sources}
        merged: list[CitationSpan] = list(evidence)
        for d in declared_cites:
            sid = d.get("source_id")
            if sid is None or sid not in src_by_id:
                continue
            o_start = int(d.get("output_start", 0))
            o_end = int(d.get("output_end", 0))
            if o_end <= o_start:
                continue
            o_start = max(0, o_start)
            o_end = min(len(output_text), o_end)
            if o_end <= o_start:
                continue
            if _interval_fully_contained((o_start, o_end), exact_intervals):
                # Exact match already covers this region with higher
                # confidence; skip.
                continue
            conf = float(d.get("confidence", 0.5))
            src = src_by_id[sid]
            s_start = int(d.get("source_start", 0))
            s_end = int(d.get("source_end", len(src.text)))
            merged.append(
                CitationSpan(
                    output_start=o_start,
                    output_end=o_end,
                    source_id=sid,
                    source_start=max(0, s_start),
                    source_end=max(s_start, s_end),
                    confidence=conf,
                    method="declared",
                )
            )
        return self._finalize(output_text, tuple(sources), merged)

    # ----------------------------------------------------------------- helpers

    def _finalize(
        self,
        output_text: str,
        sources: tuple[Source, ...],
        spans: list[CitationSpan],
    ) -> CitationReport:
        # Deterministic order: by (output_start, output_end, source_id, method).
        spans_sorted = tuple(
            sorted(
                spans,
                key=lambda s: (
                    s.output_start,
                    s.output_end,
                    s.source_id,
                    s.method,
                ),
            )
        )
        n = len(output_text)
        if n == 0:
            return CitationReport(
                sources=sources,
                spans=spans_sorted,
                coverage_ratio=0.0,
                uncited_segments=(),
            )
        intervals = [(s.output_start, s.output_end) for s in spans_sorted]
        covered = _covered_length(intervals)
        coverage = covered / n if n else 0.0
        uncited = tuple(_uncited_runs(n, intervals, self.min_uncited_run))
        return CitationReport(
            sources=sources,
            spans=spans_sorted,
            coverage_ratio=coverage,
            uncited_segments=uncited,
        )


def _interval_fully_contained(iv: tuple[int, int], covered: list[tuple[int, int]]) -> bool:
    s, e = iv
    for cs, ce in covered:
        if cs <= s and ce >= e:
            return True
    return False


__all__ = [
    "DEFAULT_MIN_EXACT_LEN",
    "DEFAULT_MIN_UNCITED_RUN",
    "Source",
    "CitationSpan",
    "CitationReport",
    "CitationTracker",
]
