"""Hierarchical Context Management — GLM-5 §6.2 (arXiv:2602.15763).

keep-recent-k strategy with discard-all fallback.
BrowseComp: 55.3% → 62.0% with hierarchical context management.

During long-horizon agentic tasks, conversation context grows unboundedly.
GLM-5 uses two strategies, tried in order:

Strategy 1 — keep-recent-k:
    Keep only the last k turns of the conversation.
    Older turns are discarded.
    BrowseComp improvement: 55.3% → 62.0%

Strategy 2 — discard-all fallback:
    When quality_score < threshold (summarization would introduce more error
    than discarding): discard all turns except the very last one.

Trigger condition: total_tokens > max_len * trigger_ratio (default 0.8)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class Turn(TypedDict):
    role: str
    content: str
    tokens: int


@dataclass
class HierarchicalContextManager:
    """GLM-5 §6.2 hierarchical context manager.

    Attributes:
        max_len: Maximum context length in tokens.
        trigger_ratio: Fraction of max_len at which truncation is triggered.
        keep_k: Number of most-recent turns to retain under keep-recent-k.
        quality_threshold: Quality score below which discard-all fallback fires.
    """

    max_len: int = 8192
    trigger_ratio: float = 0.8
    keep_k: int = 10
    quality_threshold: float = 0.3

    def manage(
        self,
        turns: list[Turn],
        quality_score: float = 1.0,
    ) -> list[Turn]:
        """Apply hierarchical context management to a list of turns.

        Args:
            turns: Conversation turns, each with role, content, and tokens.
            quality_score: Estimated quality of context summarization.
                           Values below quality_threshold trigger discard-all.

        Returns:
            Pruned list of turns. Original list is not mutated.
        """
        if not turns:
            return []

        total = sum(t["tokens"] for t in turns)
        trigger = self.max_len * self.trigger_ratio

        if total < trigger:
            return turns  # no truncation needed

        if quality_score < self.quality_threshold:
            # Discard-all fallback: keep only last turn
            return turns[-1:]

        # keep-recent-k strategy
        if self.keep_k <= 0:
            return []
        return turns[-self.keep_k :]

    def token_count(self, turns: list[Turn]) -> int:
        """Return the total token count for a list of turns."""
        return sum(t["tokens"] for t in turns)

    def utilization(self, turns: list[Turn]) -> float:
        """Return the fraction of max_len used by the given turns."""
        return self.token_count(turns) / max(self.max_len, 1)
