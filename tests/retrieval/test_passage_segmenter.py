"""Tests for src/retrieval/passage_segmenter.py (10+ tests)."""

from __future__ import annotations

from src.retrieval.passage_segmenter import (
    RETRIEVAL_REGISTRY,
    PassageSegmenter,
    Segment,
    SegmentStrategy,
)

SEGMENTER = PassageSegmenter()

LONG_TEXT = " ".join(["word"] * 600)  # 600 words ~ 2400 chars


# ---------------------------------------------------------------------------
# 1. FIXED_SIZE: produces segments
# ---------------------------------------------------------------------------
def test_fixed_size_produces_segments():
    segs = SEGMENTER.segment(LONG_TEXT, SegmentStrategy.FIXED_SIZE, max_tokens=100)
    assert len(segs) > 1


# ---------------------------------------------------------------------------
# 2. FIXED_SIZE: no segment exceeds max_chars
# ---------------------------------------------------------------------------
def test_fixed_size_respects_max_chars():
    segs = SEGMENTER.segment(LONG_TEXT, SegmentStrategy.FIXED_SIZE, max_tokens=100)
    max_chars = 100 * 4
    for s in segs:
        assert len(s.text) <= max_chars + 20, f"Segment too long: {len(s.text)}"


# ---------------------------------------------------------------------------
# 3. FIXED_SIZE: segment_ids are sequential
# ---------------------------------------------------------------------------
def test_fixed_size_sequential_ids():
    segs = SEGMENTER.segment(LONG_TEXT, SegmentStrategy.FIXED_SIZE, max_tokens=100)
    for i, s in enumerate(segs):
        assert s.segment_id == i


# ---------------------------------------------------------------------------
# 4. SENTENCE: splits on sentence boundaries
# ---------------------------------------------------------------------------
def test_sentence_splits():
    text = "Hello world. How are you? I am fine! What a day."
    segs = SEGMENTER.segment(text, SegmentStrategy.SENTENCE, max_tokens=512)
    assert len(segs) >= 1
    # Each segment should end at a sentence boundary or be a merged group
    for s in segs:
        assert s.text.strip()


# ---------------------------------------------------------------------------
# 5. SENTENCE: small max_tokens forces more splits
# ---------------------------------------------------------------------------
def test_sentence_small_max_tokens_more_splits():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."
    segs_big = SEGMENTER.segment(text, SegmentStrategy.SENTENCE, max_tokens=512)
    segs_small = SEGMENTER.segment(text, SegmentStrategy.SENTENCE, max_tokens=10)
    assert len(segs_small) >= len(segs_big)


# ---------------------------------------------------------------------------
# 6. PARAGRAPH: splits on double newlines
# ---------------------------------------------------------------------------
def test_paragraph_splits_on_double_newline():
    text = "Para one.\n\nPara two.\n\nPara three."
    segs = SEGMENTER.segment(text, SegmentStrategy.PARAGRAPH)
    assert len(segs) == 3
    assert segs[0].text == "Para one."
    assert segs[1].text == "Para two."
    assert segs[2].text == "Para three."


# ---------------------------------------------------------------------------
# 7. PARAGRAPH: single block returns one segment
# ---------------------------------------------------------------------------
def test_paragraph_single_block():
    text = "No paragraph breaks here at all."
    segs = SEGMENTER.segment(text, SegmentStrategy.PARAGRAPH)
    assert len(segs) == 1
    assert segs[0].text == text


# ---------------------------------------------------------------------------
# 8. SEMANTIC_BOUNDARY: merges short paragraphs
# ---------------------------------------------------------------------------
def test_semantic_boundary_merges_short():
    # Each paragraph is 3 chars + 2 for \n\n = 5; max_tokens=512 -> max_chars=2048
    # All three should merge into one segment
    text = "Hi.\n\nBye.\n\nOk."
    segs = SEGMENTER.segment(text, SegmentStrategy.SEMANTIC_BOUNDARY, max_tokens=512)
    assert len(segs) == 1


# ---------------------------------------------------------------------------
# 9. SEMANTIC_BOUNDARY: does not merge past max_tokens
# ---------------------------------------------------------------------------
def test_semantic_boundary_respects_max_tokens():
    para = "x" * 200 + " "  # ~50 tokens each
    text = (para + "\n\n") * 10
    segs = SEGMENTER.segment(text, SegmentStrategy.SEMANTIC_BOUNDARY, max_tokens=50)
    # With max_chars=200, single paras already hit the limit
    assert len(segs) >= 5


# ---------------------------------------------------------------------------
# 10. overlap_segments: result has same count, non-first segments are longer
# ---------------------------------------------------------------------------
def test_overlap_same_count():
    segs = SEGMENTER.segment(LONG_TEXT, SegmentStrategy.FIXED_SIZE, max_tokens=100)
    overlapped = SEGMENTER.overlap_segments(segs, overlap=10)
    assert len(overlapped) == len(segs)


def test_overlap_non_first_longer():
    segs = SEGMENTER.segment(LONG_TEXT, SegmentStrategy.FIXED_SIZE, max_tokens=100)
    overlapped = SEGMENTER.overlap_segments(segs, overlap=10)
    # Each segment from index 1 onwards should be longer than or equal to original
    for orig, ov in zip(segs[1:], overlapped[1:]):
        assert len(ov.text) >= len(orig.text)


# ---------------------------------------------------------------------------
# 11. overlap_segments: single segment unchanged
# ---------------------------------------------------------------------------
def test_overlap_single_segment_unchanged():
    segs = [Segment(text="Hello world.", start_char=0, end_char=12, segment_id=0)]
    result = SEGMENTER.overlap_segments(segs, overlap=50)
    assert len(result) == 1
    assert result[0].text == "Hello world."


# ---------------------------------------------------------------------------
# 12. RETRIEVAL_REGISTRY entry
# ---------------------------------------------------------------------------
def test_registry_entry():
    assert "passage_segmenter" in RETRIEVAL_REGISTRY
    assert isinstance(RETRIEVAL_REGISTRY["passage_segmenter"], PassageSegmenter)
