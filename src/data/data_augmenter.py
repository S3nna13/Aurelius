"""
data_augmenter.py
Text data augmentation strategies for training diversity.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AugmentationStrategy(Enum):
    SYNONYM_SWAP = "synonym_swap"
    DELETION = "deletion"
    INSERTION = "insertion"
    SWAP_WORDS = "swap_words"
    LOWERCASE = "lowercase"
    UPPERCASE = "uppercase"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentedSample:
    """Immutable record of a single augmentation result."""

    original: str
    augmented: str
    strategy: AugmentationStrategy
    seed: int


# ---------------------------------------------------------------------------
# Augmenter
# ---------------------------------------------------------------------------


class DataAugmenter:
    """Apply text augmentation strategies to individual texts or batches."""

    def __init__(self, strategies: Optional[List[AugmentationStrategy]] = None) -> None:
        if strategies is None:
            self._strategies: List[AugmentationStrategy] = list(AugmentationStrategy)
        else:
            self._strategies = list(strategies)

    # ------------------------------------------------------------------
    # Internal strategy implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _synonym_swap(text: str, seed: int) -> str:
        """Replace every 3rd word (1-indexed) with 'SYNONYM_' + word."""
        words = text.split()
        result = []
        for idx, word in enumerate(words):
            # 1-indexed: positions 3, 6, 9, …
            if (idx + 1) % 3 == 0:
                result.append(f"SYNONYM_{word}")
            else:
                result.append(word)
        return " ".join(result)

    @staticmethod
    def _deletion(text: str, seed: int) -> str:
        """Randomly delete ~10% of words; always keep at least 1 word."""
        words = text.split()
        if len(words) <= 1:
            return text
        rng = random.Random(seed)
        kept = [w for w in words if rng.random() >= 0.10]
        if not kept:
            # Guarantee at least one word survives
            kept = [rng.choice(words)]
        return " ".join(kept)

    @staticmethod
    def _insertion(text: str, seed: int) -> str:
        """Insert 'FILLER' after every 5th word (1-indexed)."""
        words = text.split()
        result = []
        for idx, word in enumerate(words):
            result.append(word)
            if (idx + 1) % 5 == 0:
                result.append("FILLER")
        return " ".join(result)

    @staticmethod
    def _swap_words(text: str, seed: int) -> str:
        """Swap adjacent word pairs: (0↔1), (2↔3), …"""
        words = text.split()
        result = list(words)
        for i in range(0, len(result) - 1, 2):
            result[i], result[i + 1] = result[i + 1], result[i]
        return " ".join(result)

    @staticmethod
    def _lowercase(text: str, seed: int) -> str:
        return text.lower()

    @staticmethod
    def _uppercase(text: str, seed: int) -> str:
        return text.upper()

    # ------------------------------------------------------------------
    # Dispatch table
    # ------------------------------------------------------------------

    _STRATEGY_FN = {
        AugmentationStrategy.SYNONYM_SWAP: _synonym_swap.__func__,  # type: ignore[attr-defined]
        AugmentationStrategy.DELETION: _deletion.__func__,          # type: ignore[attr-defined]
        AugmentationStrategy.INSERTION: _insertion.__func__,        # type: ignore[attr-defined]
        AugmentationStrategy.SWAP_WORDS: _swap_words.__func__,      # type: ignore[attr-defined]
        AugmentationStrategy.LOWERCASE: _lowercase.__func__,        # type: ignore[attr-defined]
        AugmentationStrategy.UPPERCASE: _uppercase.__func__,        # type: ignore[attr-defined]
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def augment(
        self,
        text: str,
        strategy: AugmentationStrategy,
        seed: int = 42,
    ) -> AugmentedSample:
        """Apply *strategy* to *text* and return an AugmentedSample."""
        fn = self._STRATEGY_FN[strategy]
        augmented = fn(text, seed)
        return AugmentedSample(
            original=text,
            augmented=augmented,
            strategy=strategy,
            seed=seed,
        )

    def augment_batch(
        self,
        texts: List[str],
        strategy: AugmentationStrategy,
    ) -> List[AugmentedSample]:
        """Apply *strategy* to each text in *texts*, using seed=index."""
        return [self.augment(text, strategy, seed=idx) for idx, text in enumerate(texts)]

    def apply_all(self, text: str) -> List[AugmentedSample]:
        """Apply every strategy to *text* with seed=42 and return all results."""
        return [self.augment(text, strategy, seed=42) for strategy in AugmentationStrategy]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATA_AUGMENTER_REGISTRY: dict = {"default": DataAugmenter}
