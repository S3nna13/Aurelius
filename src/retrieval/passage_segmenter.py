"""Passage segmentation for Aurelius retrieval pipeline.

Splits raw text into retrievable segments using multiple strategies:
FIXED_SIZE, SENTENCE, PARAGRAPH, and SEMANTIC_BOUNDARY.

Supports token-level overlap between adjacent segments to reduce boundary
information loss.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List

RETRIEVAL_REGISTRY: dict = {}


class SegmentStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC_BOUNDARY = "semantic_boundary"


@dataclass
class Segment:
    text: str
    start_char: int
    end_char: int
    segment_id: int


class PassageSegmenter:
    """Segments a passage of text into retrievable chunks."""

    # rough chars-per-token estimate used for FIXED_SIZE strategy
    _CHARS_PER_TOKEN: int = 4

    def segment(
        self,
        text: str,
        strategy: SegmentStrategy,
        max_tokens: int = 512,
    ) -> List[Segment]:
        """Return a list of Segment objects for *text* under *strategy*."""
        if strategy == SegmentStrategy.FIXED_SIZE:
            return self._fixed_size(text, max_tokens)
        if strategy == SegmentStrategy.SENTENCE:
            return self._sentence(text, max_tokens)
        if strategy == SegmentStrategy.PARAGRAPH:
            return self._paragraph(text, max_tokens)
        if strategy == SegmentStrategy.SEMANTIC_BOUNDARY:
            return self._semantic_boundary(text, max_tokens)
        raise ValueError(f"Unknown strategy: {strategy!r}")

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _make_segments(self, spans: list[tuple[int, int]], text: str) -> List[Segment]:
        result: List[Segment] = []
        for idx, (start, end) in enumerate(spans):
            chunk = text[start:end]
            if chunk.strip():
                result.append(Segment(text=chunk, start_char=start, end_char=end, segment_id=idx))
        return result

    def _fixed_size(self, text: str, max_tokens: int) -> List[Segment]:
        max_chars = max_tokens * self._CHARS_PER_TOKEN
        words = text.split(" ")
        spans: list[tuple[int, int]] = []
        current_start = 0
        current_chars = 0
        current_words: list[str] = []

        char_offset = 0
        word_offsets: list[int] = []
        for w in words:
            word_offsets.append(char_offset)
            char_offset += len(w) + 1  # +1 for the space

        seg_start_char = 0
        seg_word_start = 0

        for i, (w, off) in enumerate(zip(words, word_offsets)):
            if current_chars + len(w) + (1 if current_words else 0) > max_chars and current_words:
                end_char = word_offsets[i - 1] + len(words[i - 1])
                spans.append((word_offsets[seg_word_start], end_char))
                seg_word_start = i
                current_words = [w]
                current_chars = len(w)
            else:
                if current_words:
                    current_chars += 1 + len(w)
                else:
                    current_chars = len(w)
                current_words.append(w)

        if current_words:
            end_char = word_offsets[seg_word_start] + sum(len(ww) + 1 for ww in current_words) - 1
            end_char = min(end_char, len(text))
            spans.append((word_offsets[seg_word_start], end_char))

        return self._make_segments(spans, text)

    def _sentence_spans(self, text: str) -> list[tuple[int, int]]:
        """Split text into sentence spans using '. ', '! ', '? ' boundaries."""
        pattern = re.compile(r'(?<=[.!?])\s+')
        spans: list[tuple[int, int]] = []
        last = 0
        for m in pattern.finditer(text):
            spans.append((last, m.start() + 1))  # include the punctuation
            last = m.end()
        if last < len(text):
            spans.append((last, len(text)))
        return spans

    def _sentence(self, text: str, max_tokens: int) -> List[Segment]:
        max_chars = max_tokens * self._CHARS_PER_TOKEN
        raw_spans = self._sentence_spans(text)
        # merge short sentences to fill up to max_chars
        merged: list[tuple[int, int]] = []
        current_start: int | None = None
        current_end: int = 0
        current_len = 0
        for start, end in raw_spans:
            span_len = end - start
            if current_start is None:
                current_start = start
                current_end = end
                current_len = span_len
            elif current_len + span_len <= max_chars:
                current_end = end
                current_len += span_len
            else:
                merged.append((current_start, current_end))
                current_start = start
                current_end = end
                current_len = span_len
        if current_start is not None:
            merged.append((current_start, current_end))
        return self._make_segments(merged, text)

    def _paragraph(self, text: str, max_tokens: int) -> List[Segment]:
        parts = re.split(r'\n\n+', text)
        spans: list[tuple[int, int]] = []
        offset = 0
        for p in parts:
            start = offset
            end = offset + len(p)
            spans.append((start, end))
            offset = end + 2  # account for "\n\n"
        return self._make_segments(spans, text)

    def _semantic_boundary(self, text: str, max_tokens: int) -> List[Segment]:
        """Split on paragraph boundaries; merge short ones until max_tokens."""
        max_chars = max_tokens * self._CHARS_PER_TOKEN
        parts = re.split(r'\n\n+', text)
        # collect raw paragraph spans
        raw: list[tuple[int, int]] = []
        offset = 0
        for p in parts:
            raw.append((offset, offset + len(p)))
            offset += len(p) + 2
        # merge until max_chars
        merged: list[tuple[int, int]] = []
        current_start: int | None = None
        current_end: int = 0
        current_len = 0
        for start, end in raw:
            span_len = end - start
            if current_start is None:
                current_start = start
                current_end = end
                current_len = span_len
            elif current_len + span_len <= max_chars:
                current_end = end
                current_len += span_len
            else:
                merged.append((current_start, current_end))
                current_start = start
                current_end = end
                current_len = span_len
        if current_start is not None:
            merged.append((current_start, current_end))
        return self._make_segments(merged, text)

    def overlap_segments(
        self,
        segments: List[Segment],
        overlap: int = 50,
    ) -> List[Segment]:
        """Return new segments where each non-first segment prepends up to
        *overlap* tokens worth of text from the previous segment boundary."""
        if len(segments) <= 1:
            return list(segments)

        overlap_chars = overlap * self._CHARS_PER_TOKEN
        result: List[Segment] = [segments[0]]
        for idx in range(1, len(segments)):
            prev = segments[idx - 1]
            curr = segments[idx]
            prefix = prev.text[-overlap_chars:] if overlap_chars < len(prev.text) else prev.text
            new_text = prefix + curr.text
            # start_char is shifted back by the prefix length
            new_start = max(0, curr.start_char - len(prefix))
            result.append(
                Segment(
                    text=new_text,
                    start_char=new_start,
                    end_char=curr.end_char,
                    segment_id=idx,
                )
            )
        return result


RETRIEVAL_REGISTRY["passage_segmenter"] = PassageSegmenter()
