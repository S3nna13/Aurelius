"""Context window manager for Aurelius.

Provides policies for fitting long token sequences into a finite context
window: truncation, middle summarization, and sliding window.
"""

from __future__ import annotations

import math
from enum import StrEnum


class WindowPolicy(StrEnum):
    TRUNCATE_LEFT = "truncate_left"
    TRUNCATE_RIGHT = "truncate_right"
    SUMMARIZE_MIDDLE = "summarize_middle"
    SLIDING = "sliding"


class ContextWindowManager:
    """Manages how token sequences are fitted into a fixed context window.

    Parameters
    ----------
    max_tokens:
        Maximum number of tokens that fit in the context window (default 4096).
    """

    def __init__(self, max_tokens: int = 4096) -> None:
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fits(self, token_ids: list[int]) -> bool:
        """Return True if *token_ids* fits in the context window."""
        return len(token_ids) <= self.max_tokens

    def apply_policy(
        self,
        token_ids: list[int],
        policy: WindowPolicy,
        summary_size: int = 128,
    ) -> list[int]:
        """Apply *policy* to *token_ids* and return the trimmed sequence.

        Parameters
        ----------
        token_ids:
            Input token IDs (may exceed ``max_tokens``).
        policy:
            Which WindowPolicy to apply.
        summary_size:
            Number of placeholder tokens used to represent a summarized middle
            segment under SUMMARIZE_MIDDLE (default 128).

        Returns
        -------
        list[int]
            Token IDs fitted to the context window.
        """
        n = len(token_ids)
        mt = self.max_tokens

        if policy == WindowPolicy.TRUNCATE_LEFT:
            return token_ids[-mt:]

        if policy == WindowPolicy.TRUNCATE_RIGHT:
            return token_ids[:mt]

        if policy == WindowPolicy.SUMMARIZE_MIDDLE:
            keep_head = mt // 4
            keep_tail = mt // 4
            head = token_ids[:keep_head]
            tail = token_ids[-keep_tail:] if keep_tail > 0 else []
            placeholder = [-1] * summary_size
            return head + placeholder + tail

        if policy == WindowPolicy.SLIDING:
            step = mt // 2
            if step == 0:
                step = 1
            # Return the last complete window
            if n <= mt:
                return token_ids
            # Number of full windows
            num_windows = math.ceil((n - mt) / step) + 1
            start = (num_windows - 1) * step
            return token_ids[start : start + mt]

        raise ValueError(f"Unknown policy: {policy!r}")

    def token_count_estimate(self, text: str) -> int:
        """Rough token count estimate: word count * 1.3."""
        return round(len(text.split()) * 1.3)
