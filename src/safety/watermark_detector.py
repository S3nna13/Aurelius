"""Watermark Detector — statistical watermark detection for LLM outputs.

Implements z-score based detection for several watermarking schemes.
No external ML frameworks required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

# ---------------------------------------------------------------------------
# Enums / dataclasses
# ---------------------------------------------------------------------------


class WatermarkScheme(StrEnum):
    GREEN_LIST = "green_list"
    RED_GREEN = "red_green"
    UNIGRAM = "unigram"
    MULTI_HASH = "multi_hash"


@dataclass
class WatermarkConfig:
    scheme: WatermarkScheme
    key: int = 42
    gamma: float = 0.25
    delta: float = 2.0


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def _green_list_member(token_id: int, key: int) -> bool:
    """Return True if token_id is in the green list defined by key.

    Green list: hash(token_id XOR key) % 2 == 0, approximating gamma=0.5.
    For adjustable gamma the caller rescales using WatermarkConfig.gamma.
    """
    return ((token_id ^ key) * 2654435761) % (2**32) % 2 == 0


def _extended_green_list_member(token_id: int, key: int, vocab_size: int, gamma: float) -> bool:
    """Green list membership using gamma fraction of vocabulary."""
    bucket = int(vocab_size * gamma)
    h = ((token_id ^ key) * 2654435761) % (2**32)
    return (h % vocab_size) < bucket


class WatermarkDetector:
    """Detects LLM watermarks via statistical z-score analysis."""

    def detect(
        self,
        token_ids: list[int],
        vocab_size: int,
        config: WatermarkConfig,
    ) -> float:
        """Compute a watermark z-score for the given token sequence.

        For GREEN_LIST / RED_GREEN / UNIGRAM schemes: counts the fraction of
        tokens that fall in the green list (defined by config.key and
        config.gamma) and converts to a z-score.

        For MULTI_HASH: uses a per-position hash that also incorporates the
        preceding token id, then applies the same z-score formula.

        Args:
            token_ids: Sequence of integer token IDs to analyse.
            vocab_size: Total vocabulary size (used for bucket sizing).
            config: WatermarkConfig specifying scheme, key, gamma, delta.

        Returns:
            Float z-score. Values > 4.0 are considered likely watermarked.
        """
        n = len(token_ids)
        if n == 0:
            return 0.0

        gamma = config.gamma
        key = config.key

        if config.scheme in (
            WatermarkScheme.GREEN_LIST,
            WatermarkScheme.RED_GREEN,
            WatermarkScheme.UNIGRAM,
        ):
            count_green = sum(
                1 for t in token_ids if _extended_green_list_member(t, key, vocab_size, gamma)
            )
        elif config.scheme == WatermarkScheme.MULTI_HASH:
            count_green = 0
            for i, t in enumerate(token_ids):
                prev = token_ids[i - 1] if i > 0 else 0
                combined_key = key ^ prev
                if _extended_green_list_member(t, combined_key, vocab_size, gamma):
                    count_green += 1
        else:
            count_green = sum(
                1 for t in token_ids if _extended_green_list_member(t, key, vocab_size, gamma)
            )

        # z = (observed_fraction - gamma) / sqrt(gamma * (1 - gamma) / n)
        denominator = math.sqrt(gamma * (1.0 - gamma) / n)
        if denominator < 1e-12:
            return 0.0

        z = (count_green / n - gamma) / denominator
        return z

    def is_watermarked(
        self,
        token_ids: list[int],
        vocab_size: int,
        config: WatermarkConfig,
        z_threshold: float = 4.0,
    ) -> bool:
        """Return True if the sequence is likely watermarked.

        Args:
            token_ids: Sequence of token IDs.
            vocab_size: Vocabulary size.
            config: Watermark configuration.
            z_threshold: Z-score threshold; default 4.0.

        Returns:
            True if z-score >= z_threshold.
        """
        return self.detect(token_ids, vocab_size, config) >= z_threshold


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SAFETY_REGISTRY: dict = {}
SAFETY_REGISTRY["watermark_detector"] = WatermarkDetector()
