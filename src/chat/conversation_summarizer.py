"""Conversation summarizer: extractive summary, key topic extraction, length control."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum

_STOPWORDS = {
    "a", "an", "the", "is", "in", "to", "of", "and", "or", "for",
    "with", "that", "this", "it", "be", "was", "are", "were", "has", "have",
}


class SummaryMode(str, Enum):
    EXTRACTIVE = "extractive"
    ABSTRACTIVE_STUB = "abstractive_stub"
    BULLETS = "bullets"


@dataclass
class ConversationSummary:
    original_turn_count: int
    summary_text: str
    key_topics: list[str]
    compression_ratio: float
    mode: SummaryMode


class ConversationSummarizer:
    def __init__(
        self,
        mode: SummaryMode = SummaryMode.EXTRACTIVE,
        max_sentences: int = 5,
    ) -> None:
        self.mode = mode
        self.max_sentences = max_sentences

    def extract_topics(self, messages: list[str]) -> list[str]:
        all_text = " ".join(messages).lower()
        words = re.findall(r"[a-z]+", all_text)
        filtered = [w for w in words if w not in _STOPWORDS]
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(5)]

    def _select_messages(self, messages: list[str]) -> list[str]:
        """Select up to max_sentences messages: first + last (max_sentences-2) middle + last."""
        if not messages:
            return []
        if len(messages) == 1:
            return [messages[0]]
        if len(messages) <= self.max_sentences:
            return messages

        selected: list[str] = [messages[0]]
        # last (max_sentences - 2) messages before the final
        tail_count = self.max_sentences - 2
        if tail_count > 0:
            middle_tail = messages[-(tail_count + 1):-1]
            for m in middle_tail:
                if m not in selected:
                    selected.append(m)
        # always include last
        if messages[-1] not in selected:
            selected.append(messages[-1])
        return selected

    def summarize_extractive(self, messages: list[str]) -> str:
        selected = self._select_messages(messages)
        return " | ".join(selected)

    def summarize_bullets(self, messages: list[str]) -> str:
        selected = self._select_messages(messages)
        lines = [f"• {m[:60]}" for m in selected]
        return "\n".join(lines)

    def summarize(self, messages: list[str]) -> ConversationSummary:
        n = len(messages)
        topics = self.extract_topics(messages)
        total_len = max(1, sum(len(m) for m in messages))

        if self.mode == SummaryMode.EXTRACTIVE:
            text = self.summarize_extractive(messages)
        elif self.mode == SummaryMode.BULLETS:
            text = self.summarize_bullets(messages)
        else:  # ABSTRACTIVE_STUB
            first_100 = messages[0][:100] if messages else ""
            text = f"Summary of {n} messages: {first_100}..."

        ratio = len(text) / total_len
        return ConversationSummary(
            original_turn_count=n,
            summary_text=text,
            key_topics=topics,
            compression_ratio=ratio,
            mode=self.mode,
        )
