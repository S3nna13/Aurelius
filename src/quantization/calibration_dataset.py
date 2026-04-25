"""Calibration dataset for post-training quantization (PTQ) in Aurelius.

Provides:

* :class:`CalibrationSample`  — frozen dataclass representing a single sample
* :class:`CalibrationDataset` — bounded, iterable collection of samples with
  statistics and sub-sampling utilities
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationSample:
    """A single calibration sample.

    Args:
        text:   Raw text of the sample (may be empty for token-only data).
        tokens: Pre-tokenised integer token IDs.
        source: Optional provenance tag (dataset name, file path, etc.).
    """

    text: str
    tokens: List[int]
    source: str = ""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CalibrationDataset:
    """Bounded, iterable container of :class:`CalibrationSample` objects.

    Args:
        max_samples: Hard upper bound on the number of samples this dataset
                     may hold.  Defaults to 512.

    Example::

        ds = CalibrationDataset(max_samples=128)
        ds.add(CalibrationSample(text="hello", tokens=[1, 2, 3]))
        print(ds.stats())
    """

    def __init__(self, max_samples: int = 512) -> None:
        if max_samples < 0:
            raise ValueError("max_samples must be non-negative.")
        self._max_samples: int = max_samples
        self._samples: List[CalibrationSample] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, sample: CalibrationSample) -> None:
        """Append *sample* to the dataset.

        Raises:
            ValueError: If the dataset is already at capacity
                        (``len(self) == self.max_samples``).
        """
        if len(self._samples) >= self._max_samples:
            raise ValueError(
                f"CalibrationDataset is at capacity ({self._max_samples} samples). "
                "Increase max_samples or use a new dataset."
            )
        self._samples.append(sample)

    # ------------------------------------------------------------------
    # Sequence-like interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[CalibrationSample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> CalibrationSample:
        return self._samples[idx]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def token_counts(self) -> List[int]:
        """Return a list of token counts, one per sample."""
        return [len(s.tokens) for s in self._samples]

    def stats(self) -> dict:
        """Compute summary statistics over the dataset.

        Returns:
            Dict with keys ``count``, ``mean_tokens``, ``max_tokens``,
            ``min_tokens``.

        Raises:
            ValueError: If the dataset is empty (no samples to aggregate).
        """
        n = len(self._samples)
        if n == 0:
            raise ValueError("Cannot compute stats on an empty CalibrationDataset.")
        counts = self.token_counts()
        return {
            "count": n,
            "mean_tokens": sum(counts) / n,
            "max_tokens": max(counts),
            "min_tokens": min(counts),
        }

    # ------------------------------------------------------------------
    # Sub-sampling
    # ------------------------------------------------------------------

    def subsample(self, n: int, seed: int = 42) -> "CalibrationDataset":
        """Return a new :class:`CalibrationDataset` with *n* random samples.

        The selection is performed without replacement using
        ``random.Random(seed)``.  If *n* ≥ ``len(self)`` the returned dataset
        contains all current samples (in a reproducible shuffled order).

        Args:
            n:    Number of samples to select.
            seed: RNG seed for reproducibility.

        Returns:
            A new :class:`CalibrationDataset` with ``max_samples`` set to *n*
            (or ``len(self)`` when *n* > current size).

        Raises:
            ValueError: If *n* is negative.
        """
        if n < 0:
            raise ValueError("n must be non-negative.")

        rng = random.Random(seed)
        population = list(self._samples)
        k = min(n, len(population))

        selected = rng.sample(population, k)

        new_ds = CalibrationDataset(max_samples=max(n, k))
        for s in selected:
            new_ds.add(s)
        return new_ds

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CalibrationDataset(count={len(self._samples)}, "
            f"max_samples={self._max_samples})"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CALIBRATION_DATASET_REGISTRY: dict[str, type] = {"default": CalibrationDataset}
