"""
Synthetic augmentation for training data using model-free token-level techniques.
Pure Python stdlib + PyTorch only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SyntheticAugConfig:
    """Configuration for synthetic augmentation."""

    p_mask: float = 0.15
    p_insert: float = 0.1
    p_replace: float = 0.1
    vocab_size: int = 32000
    mask_token_id: int = 103
    n_augments: int = 3
    seed: int | None = None


# ---------------------------------------------------------------------------
# Standalone augmentation functions
# ---------------------------------------------------------------------------


def mask_tokens(
    token_ids: list[int],
    p: float,
    mask_id: int,
    rng: random.Random,
) -> list[int]:
    """Replace each token with mask_id with probability p.

    Guarantees at least one token remains unmasked.
    """
    result = list(token_ids)
    n = len(result)
    if n == 0:
        return result

    masked_indices: list[int] = []
    for i in range(n):
        if rng.random() < p:
            masked_indices.append(i)

    # If everything would be masked, keep one token unmasked
    if len(masked_indices) == n:
        keep = rng.randint(0, n - 1)
        masked_indices = [i for i in masked_indices if i != keep]

    for i in masked_indices:
        result[i] = mask_id

    return result


def random_token_insertion(
    token_ids: list[int],
    p: float,
    vocab_size: int,
    rng: random.Random,
) -> list[int]:
    """Insert a random token after each position with probability p.

    The resulting list may be longer than the original.
    """
    result: list[int] = []
    for tok in token_ids:
        result.append(tok)
        if rng.random() < p:
            result.append(rng.randint(0, vocab_size - 1))
    return result


def random_token_replacement(
    token_ids: list[int],
    p: float,
    vocab_size: int,
    rng: random.Random,
) -> list[int]:
    """Replace each token with a uniformly random token with probability p."""
    return [rng.randint(0, vocab_size - 1) if rng.random() < p else tok for tok in token_ids]


def sentence_order_permutation(
    token_ids: list[int],
    sep_id: int = 13,
) -> list[int]:
    """Split at sep_id boundaries, shuffle sentence segments, rejoin with sep_id.

    If fewer than 2 segments are present, returns the sequence unchanged.
    """
    if not token_ids:
        return list(token_ids)

    # Split into segments on sep_id; sep_id tokens are boundaries
    segments: list[list[int]] = []
    current: list[int] = []
    for tok in token_ids:
        if tok == sep_id:
            segments.append(current)
            current = []
        else:
            current.append(tok)
    segments.append(current)

    # Filter to non-empty segments; if fewer than 2, return unchanged
    non_empty = [s for s in segments if s]
    if len(non_empty) < 2:
        return list(token_ids)

    # Shuffle a copy using Python's built-in (deterministic if caller seeded)
    shuffled = non_empty[:]
    random.shuffle(shuffled)

    # Rejoin with sep_id
    result: list[int] = []
    for idx, seg in enumerate(shuffled):
        result.extend(seg)
        if idx < len(shuffled) - 1:
            result.append(sep_id)
    return result


def span_masking(
    token_ids: list[int],
    mask_ratio: float,
    max_span_len: int = 5,
    mask_id: int = 103,
    rng: random.Random = None,  # type: ignore[assignment]
) -> list[int]:
    """Mask contiguous spans totalling approximately mask_ratio * len(tokens).

    Selects random spans of length 1..max_span_len until the budget is met or
    no unmasked tokens remain.
    """
    if rng is None:
        rng = random.Random()

    result = list(token_ids)
    n = len(result)
    if n == 0 or mask_ratio <= 0.0:
        return result

    budget = max(1, round(mask_ratio * n))
    masked = [False] * n

    attempts = 0
    max_attempts = n * 10
    while budget > 0 and attempts < max_attempts:
        attempts += 1
        span_len = rng.randint(1, min(max_span_len, budget, n))
        start = rng.randint(0, n - span_len)
        # Mask this span
        newly_masked = 0
        for i in range(start, start + span_len):
            if not masked[i]:
                masked[i] = True
                result[i] = mask_id
                newly_masked += 1
        budget -= newly_masked

    return result


def compute_token_overlap(seq_a: list[int], seq_b: list[int]) -> float:
    """Jaccard similarity of token sets: |A ∩ B| / |A ∪ B|."""
    set_a = set(seq_a)
    set_b = set(seq_b)
    union = set_a | set_b
    if not union:
        return 1.0  # both empty → identical
    intersection = set_a & set_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# SyntheticAugmentor class
# ---------------------------------------------------------------------------


class SyntheticAugmentor:
    """Applies a configurable pipeline of model-free token augmentations."""

    _DEFAULT_METHODS: list[str] = ["mask", "replace"]

    def __init__(self, config: SyntheticAugConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)

    def augment(
        self,
        token_ids: list[int],
        methods: list[str] | None = None,
    ) -> list[int]:
        """Apply configured augmentation methods in order.

        Available methods: "mask", "insert", "replace", "span_mask", "permute".
        Default: ["mask", "replace"].
        """
        if methods is None:
            methods = self._DEFAULT_METHODS

        result = list(token_ids)
        cfg = self.config

        for method in methods:
            if method == "mask":
                result = mask_tokens(result, cfg.p_mask, cfg.mask_token_id, self._rng)
            elif method == "insert":
                result = random_token_insertion(result, cfg.p_insert, cfg.vocab_size, self._rng)
            elif method == "replace":
                result = random_token_replacement(result, cfg.p_replace, cfg.vocab_size, self._rng)
            elif method == "span_mask":
                result = span_masking(
                    result,
                    mask_ratio=cfg.p_mask,
                    mask_id=cfg.mask_token_id,
                    rng=self._rng,
                )
            elif method == "permute":
                result = sentence_order_permutation(result)
            else:
                raise ValueError(f"Unknown augmentation method: {method!r}")

        return result

    def augment_batch(
        self,
        batch: list[list[int]],
        n_augments: int = 1,
    ) -> list[list[int]]:
        """Augment each sequence n_augments times.

        Returns originals followed by all augmented versions:
        total length = len(batch) * (1 + n_augments).
        """
        result: list[list[int]] = list(batch)
        for seq in batch:
            for _ in range(n_augments):
                result.append(self.augment(seq))
        return result

    def get_stats(
        self,
        original: list[int],
        augmented: list[int],
    ) -> dict[str, float]:
        """Return statistics comparing original and augmented sequences."""
        original_len = float(len(original))
        augmented_len = float(len(augmented))
        overlap = compute_token_overlap(original, augmented)

        if original_len > 0:
            change_ratio = 1.0 - overlap
        else:
            change_ratio = 0.0

        return {
            "original_len": original_len,
            "augmented_len": augmented_len,
            "overlap": overlap,
            "change_ratio": change_ratio,
        }
