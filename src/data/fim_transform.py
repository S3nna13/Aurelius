"""Fill-in-the-Middle (FIM) transformation for code data.

Implements the FIM objective described in *Efficient Training of Language
Models to Fill in the Middle* (Bavarian et al., 2022).  50 % of code
examples are FIM-transformed; of those, half use Prefix-Suffix-Middle
(PSM) format and half use Suffix-Prefix-Middle (SPM) format.

Use ``fim_transform`` for a single example or ``fim_transform_batch`` for
a list.  Each uses a per-example seeded RNG for reproducibility.

Special tokens
--------------
- ``<|fim_prefix|>``
- ``<|fim_middle|>``
- ``<|fim_suffix|>``
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

FIM_PREFIX: str = "<|fim_prefix|>"
FIM_MIDDLE: str = "<|fim_middle|>"
FIM_SUFFIX: str = "<|fim_suffix|>"


class FIMMode(Enum):
    """Which FIM serialization order to use."""

    PSM = auto()  # Prefix-Suffix-Middle
    SPM = auto()  # Suffix-Prefix-Middle


@dataclass(slots=True)
class FIMConfig:
    """Knobs for the FIM transformation."""

    fim_rate: float = 0.50       # fraction of examples that get FIM
    psm_rate: float = 0.50       # of FIM examples, fraction using PSM
    min_prefix_len: int = 1      # minimum characters in prefix
    min_suffix_len: int = 1      # minimum characters in suffix
    min_middle_len: int = 1      # minimum characters in middle
    seed: int | None = None      # reproducible randomness

    def __post_init__(self) -> None:
        if not 0.0 <= self.fim_rate <= 1.0:
            raise ValueError(f"fim_rate must be in [0, 1], got {self.fim_rate}")
        if not 0.0 <= self.psm_rate <= 1.0:
            raise ValueError(f"psm_rate must be in [0, 1], got {self.psm_rate}")


# ---------------------------------------------------------------------------
# Core transformation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FIMResult:
    """Result of a single FIM transformation."""

    text: str
    mode: FIMMode | None  # None when the example was *not* FIM-transformed
    prefix: str
    middle: str
    suffix: str
    was_transformed: bool


def _choose_split_point(
    text: str,
    rng: random.Random,
    min_prefix: int,
    min_suffix: int,
    min_middle: int,
) -> tuple[int, int] | None:
    """Return (prefix_end, suffix_start) indices or None if text is too short."""
    min_len = min_prefix + min_middle + min_suffix
    if len(text) < min_len:
        return None

    # Pick two random cut points that respect the minimums.
    prefix_end = rng.randint(min_prefix, len(text) - min_suffix - min_middle)
    suffix_start = rng.randint(prefix_end + min_middle, len(text) - min_suffix)
    return prefix_end, suffix_start


def fim_transform(
    text: str,
    *,
    config: FIMConfig | None = None,
    rng: random.Random | None = None,
) -> FIMResult:
    """Apply the FIM transformation to a single code example.

    Parameters
    ----------
    text:
        Raw source code string.
    config:
        FIM hyper-parameters.  Uses defaults if *None*.
    rng:
        Pre-seeded random generator.  A new one is created from
        ``config.seed`` when not supplied.

    Returns
    -------
    FIMResult
        The (possibly transformed) text together with metadata.
    """
    if config is None:
        config = FIMConfig()
    if rng is None:
        rng = random.Random(config.seed)

    # Decide whether to apply FIM at all.
    if rng.random() >= config.fim_rate:
        return FIMResult(
            text=text,
            mode=None,
            prefix=text,
            middle="",
            suffix="",
            was_transformed=False,
        )

    # Choose random split points.
    points = _choose_split_point(
        text,
        rng,
        config.min_prefix_len,
        config.min_suffix_len,
        config.min_middle_len,
    )
    if points is None:
        # Text too short -- return untransformed.
        return FIMResult(
            text=text,
            mode=None,
            prefix=text,
            middle="",
            suffix="",
            was_transformed=False,
        )

    prefix_end, suffix_start = points
    prefix = text[:prefix_end]
    middle = text[prefix_end:suffix_start]
    suffix = text[suffix_start:]

    # Choose PSM vs SPM.
    mode = FIMMode.PSM if rng.random() < config.psm_rate else FIMMode.SPM

    if mode is FIMMode.PSM:
        transformed = (
            f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"
        )
    else:
        transformed = (
            f"{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}{middle}"
        )

    return FIMResult(
        text=transformed,
        mode=mode,
        prefix=prefix,
        middle=middle,
        suffix=suffix,
        was_transformed=True,
    )


# ---------------------------------------------------------------------------
# Batch / streaming helpers
# ---------------------------------------------------------------------------

def fim_transform_batch(
    texts: list[str],
    *,
    config: FIMConfig | None = None,
    seed: int = 42,
) -> list[FIMResult]:
    """Apply FIM to a list of code strings with deterministic randomness.

    Each example gets its own RNG derived from ``seed + index`` so that
    results are reproducible regardless of parallelism.
    """
    if config is None:
        config = FIMConfig()
    results: list[FIMResult] = []
    for idx, text in enumerate(texts):
        rng = random.Random(seed + idx)
        results.append(fim_transform(text, config=config, rng=rng))
    return results


def apply_fim(text: str, seed: int = 42) -> str:
    """Convenience wrapper: apply FIM and return the transformed string."""
    rng = random.Random(seed)
    result = fim_transform(text, rng=rng)
    return result.text
