"""
context_compressor.py – Compresses agent context window by summarizing old turns.
Stdlib-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    role: str
    content: str
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if self.token_estimate == 0:
            self.token_estimate = len(self.content.split())


@dataclass(frozen=True)
class CompressionConfig:
    max_tokens: int = 4096
    keep_recent: int = 4
    summary_ratio: float = 0.3


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


class ContextCompressor:
    """Compresses a list of conversation turns to fit within a token budget."""

    def __init__(self, config: CompressionConfig | None = None) -> None:
        self._config: CompressionConfig = config if config is not None else CompressionConfig()

    # --- helpers ----------------------------------------------------------

    def estimate_tokens(self, turns: list[Turn]) -> int:
        return sum(t.token_estimate for t in turns)

    def compress(self, turns: list[Turn]) -> list[Turn]:
        cfg = self._config
        total = self.estimate_tokens(turns)
        if total <= cfg.max_tokens:
            return list(turns)

        # Split: recent vs. older
        recent = turns[-cfg.keep_recent :] if cfg.keep_recent > 0 else []
        older = turns[: len(turns) - cfg.keep_recent] if cfg.keep_recent > 0 else turns

        # Summarize older turns by extracting every Nth word
        n = math.ceil(1.0 / cfg.summary_ratio)
        all_words: list[str] = []
        for t in older:
            all_words.extend(t.content.split())
        summarized_words = [w for idx, w in enumerate(all_words) if idx % n == 0]
        summarized_text = " ".join(summarized_words)

        summary_turn = Turn(role="summary", content=summarized_text)
        return [summary_turn] + list(recent)

    def compression_ratio(self, original: list[Turn], compressed: list[Turn]) -> float:
        original_tokens = self.estimate_tokens(original)
        compressed_tokens = self.estimate_tokens(compressed)
        return original_tokens / max(compressed_tokens, 1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONTEXT_COMPRESSOR_REGISTRY: dict[str, type] = {"default": ContextCompressor}
