"""Unit tests for the citation tracker."""

from __future__ import annotations

import pytest

from src.retrieval.citation_tracker import (
    DEFAULT_MIN_EXACT_LEN,
    DEFAULT_MIN_UNCITED_RUN,
    CitationReport,
    CitationTracker,
    Source,
)


def _src(sid: str, text: str) -> Source:
    return Source(id=sid, text=text, origin="unit", retrieved_at="2026-04-20T00:00:00Z")


def test_module_constants():
    assert DEFAULT_MIN_EXACT_LEN == 20
    assert DEFAULT_MIN_UNCITED_RUN == 40


def test_exact_match_found():
    src = _src("s1", "The quick brown fox jumps over the lazy dog repeatedly.")
    out = "Intro. The quick brown fox jumps over the lazy dog repeatedly. Tail."
    tr = CitationTracker()
    rep = tr.track(out, [src])
    assert isinstance(rep, CitationReport)
    assert len(rep.spans) == 1
    sp = rep.spans[0]
    assert sp.method == "exact_match"
    assert sp.source_id == "s1"
    assert out[sp.output_start : sp.output_end] == src.text
    assert sp.confidence == 1.0


def test_short_match_below_min_len_ignored():
    # 10-char overlap only
    src = _src("s1", "hello world")
    out = "prefix hello world suffix with many other characters added here."
    tr = CitationTracker(min_exact_len=DEFAULT_MIN_EXACT_LEN)
    rep = tr.track(out, [src])
    assert rep.spans == ()
    assert rep.coverage_ratio == 0.0


def test_multiple_sources_each_contribute_spans():
    s1 = _src("a", "alpha beta gamma delta epsilon zeta")
    s2 = _src("b", "one two three four five six seven eight")
    out = "alpha beta gamma delta epsilon zeta :: one two three four five six seven eight"
    tr = CitationTracker()
    rep = tr.track(out, [s1, s2])
    ids = {sp.source_id for sp in rep.spans}
    assert ids == {"a", "b"}


def test_no_match_yields_empty():
    src = _src("s1", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    out = "completely different output text with nothing in common at all."
    tr = CitationTracker()
    rep = tr.track(out, [src])
    assert rep.spans == ()
    assert rep.coverage_ratio == 0.0


def test_empty_output():
    src = _src("s1", "anything at all here for coverage tests")
    tr = CitationTracker()
    rep = tr.track("", [src])
    assert rep.spans == ()
    assert rep.coverage_ratio == 0.0
    assert rep.uncited_segments == ()


def test_empty_sources():
    tr = CitationTracker()
    rep = tr.track("output text that is not short at all really", [])
    assert rep.spans == ()
    assert rep.coverage_ratio == 0.0


def test_uncited_segment_detection():
    src = _src("s1", "CITED PORTION OF THE OUTPUT HERE IS LONG ENOUGH")
    out = "CITED PORTION OF THE OUTPUT HERE IS LONG ENOUGH " + "X" * 60
    tr = CitationTracker()
    rep = tr.track(out, [src])
    assert len(rep.spans) == 1
    assert len(rep.uncited_segments) == 1
    s, e = rep.uncited_segments[0]
    assert (e - s) >= DEFAULT_MIN_UNCITED_RUN


def test_coverage_ratio_correct():
    src_text = "A" * 50
    src = _src("s1", src_text)
    out = src_text + ("B" * 50)
    tr = CitationTracker()
    rep = tr.track(out, [src])
    # covered 50 of 100
    assert rep.coverage_ratio == pytest.approx(0.5, abs=1e-9)


def test_declared_citations_merge():
    s1 = _src("s1", "background material about topic X that is lengthy")
    out = "Our model predicts [1] results with high reliability end."
    tr = CitationTracker()
    declared = [
        {
            "source_id": "s1",
            "output_start": 0,
            "output_end": len(out),
            "confidence": 0.6,
        }
    ]
    rep = tr.track_with_declared(out, [s1], declared)
    methods = {sp.method for sp in rep.spans}
    assert "declared" in methods


def test_declared_conflict_resolves_to_exact():
    # Source text appears exactly in output, and also a declared span covers
    # the same region. Exact should win (declared suppressed).
    src_text = "exact matching phrase that is plenty long enough"
    src = _src("s1", src_text)
    out = src_text + " tail content."
    tr = CitationTracker()
    declared = [
        {
            "source_id": "s1",
            "output_start": 0,
            "output_end": len(src_text),
            "confidence": 0.5,
        }
    ]
    rep = tr.track_with_declared(out, [src], declared)
    # Only one span, and it must be exact_match.
    assert len(rep.spans) == 1
    assert rep.spans[0].method == "exact_match"
    assert rep.spans[0].confidence == 1.0


def test_fuzzy_match_used_when_similarity_fn_provided():
    src = _src("s1", "The stock market closed higher on Tuesday.")
    out = "Equities finished up on Tuesday across most sectors today fully."

    def sim(a: str, b: str) -> float:
        return 0.9  # always a match

    tr = CitationTracker(similarity_fn=sim, fuzzy_threshold=0.75)
    rep = tr.track(out, [src])
    methods = {sp.method for sp in rep.spans}
    assert "fuzzy_match" in methods


def test_fuzzy_ignored_when_similarity_fn_none():
    src = _src("s1", "nothing in common whatsoever really.")
    out = "totally unrelated output string of sufficient length here ok."
    tr = CitationTracker(similarity_fn=None)
    rep = tr.track(out, [src])
    assert rep.spans == ()


def test_unicode_text():
    src = _src("s1", "café naïve résumé über München größer straße!")
    out = "Prefix: café naïve résumé über München größer straße! tail."
    tr = CitationTracker()
    rep = tr.track(out, [src])
    assert len(rep.spans) == 1
    sp = rep.spans[0]
    assert out[sp.output_start : sp.output_end] == src.text


def test_determinism():
    s1 = _src("a", "alpha beta gamma delta epsilon zeta eta theta")
    s2 = _src("b", "one two three four five six seven eight nine ten")
    out = (
        "alpha beta gamma delta epsilon zeta eta theta "
        "one two three four five six seven eight nine ten"
    )
    tr = CitationTracker()
    r1 = tr.track(out, [s1, s2])
    r2 = tr.track(out, [s1, s2])
    assert r1 == r2


def test_min_exact_len_knob_honored():
    src = _src("s1", "short phrase!")  # 13 chars
    out = "lead short phrase! trailing extension text for bulk padding here."
    # default 20 -> no match
    assert CitationTracker().track(out, [src]).spans == ()
    # lowered to 10 -> matched
    rep = CitationTracker(min_exact_len=10).track(out, [src])
    assert len(rep.spans) == 1


def test_min_uncited_run_knob_honored():
    src = _src("s1", "CITED PORTION OF OUTPUT HERE IS LONG OK")
    out = "CITED PORTION OF OUTPUT HERE IS LONG OK " + ("y" * 20)
    # default 40 -> 20 chars of y's do not count
    rep1 = CitationTracker().track(out, [src])
    assert rep1.uncited_segments == ()
    # lowered to 10 -> they count
    rep2 = CitationTracker(min_uncited_run=10).track(out, [src])
    assert len(rep2.uncited_segments) == 1


def test_overlapping_spans_handled():
    # Two sources share overlapping content with the output; coverage should
    # dedupe the overlap (no double counting).
    src_text = "overlap region of sufficient length here please"
    s1 = _src("a", src_text)
    s2 = _src("b", src_text)
    out = src_text + " trailing."
    tr = CitationTracker()
    rep = tr.track(out, [s1, s2])
    # Two spans recorded...
    assert len(rep.spans) == 2
    # ...but coverage is capped at len(src_text)/len(out), not doubled.
    expected = len(src_text) / len(out)
    assert rep.coverage_ratio == pytest.approx(expected, abs=1e-9)
    assert rep.coverage_ratio <= 1.0
