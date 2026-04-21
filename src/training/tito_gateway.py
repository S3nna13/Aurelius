"""TITO (Token-in-Token-out) Gateway — GLM-5 §4.1 (arXiv:2602.15763).

Eliminates re-tokenization mismatches at the inference/training engine boundary.
Validates token IDs and canonicalizes across engine vocabularies.

In off-policy RL, the inference engine generates completions using one tokenizer
state, and the training engine consumes them using a potentially different tokenizer
state.  Re-tokenization at the boundary can cause:
  - Off-by-one token ID mismatches
  - Padding token ID differences between engines
  - Vocabulary boundary errors when engines have different vocab_size

The TITO Gateway sits at this boundary and canonicalizes token IDs so both engines
speak the same token "language."  It validates all IDs are within [0, vocab_size) and
raises loudly if not (no silent fallbacks).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TITOConfig:
    """Configuration for the TITO Gateway.

    Attributes:
        vocab_size: Total vocabulary size.  Valid token IDs are [0, vocab_size).
        pad_id: Token ID used for padding.
        unk_id: Token ID used for unknown tokens.
        id_remap: Optional mapping {inference_id: training_id} applied before
            validation.  Allows bridging engines whose special-token IDs differ.
    """

    vocab_size: int
    pad_id: int = 0
    unk_id: int = 1
    # Optional ID remapping: {inference_id: training_id}
    id_remap: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id_remap is None:
            self.id_remap = {}


class TITOGateway:
    """Token-in-Token-out boundary gateway (GLM-5 §4.1).

    Canonicalizes token IDs crossing the inference→training boundary so that
    both engines operate on a consistent vocabulary.  Any ID outside
    [0, vocab_size) — even after remapping — raises a ``ValueError`` immediately;
    there are no silent fallbacks.

    Example::

        config = TITOConfig(vocab_size=32000, pad_id=0, unk_id=1,
                            id_remap={50256: 2})   # bridge GPT-2 EOS → training EOS
        gw = TITOGateway(config)
        ids = gw.canonicalize([0, 1, 50256, 999])   # [0, 1, 2, 999]
    """

    def __init__(self, config: TITOConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def canonicalize(self, token_ids: list[int]) -> list[int]:
        """Validate and canonicalize a single sequence of token IDs.

        Steps for each token ID:
        1. Apply ``config.id_remap`` if the ID is present in the mapping.
        2. Validate that the (possibly remapped) ID is in ``[0, vocab_size)``.
        3. Append to the output list.

        Args:
            token_ids: Raw token IDs from the inference engine.

        Returns:
            Canonicalized token IDs safe for the training engine.

        Raises:
            ValueError: If any token ID (after remapping) is outside
                ``[0, vocab_size)``.
        """
        if not token_ids:
            return []

        vocab_size = self.config.vocab_size
        id_remap = self.config.id_remap
        out: list[int] = []

        for t in token_ids:
            # Apply remapping first
            t = id_remap.get(t, t)
            if t < 0 or t >= vocab_size:
                raise ValueError(
                    f"Token ID {t} out of range [0, {vocab_size}). "
                    f"This indicates a tokenizer mismatch between inference and "
                    f"training engines."
                )
            out.append(t)

        return out

    def wrap_batch(self, batch: list[list[int]]) -> list[list[int]]:
        """Canonicalize a batch of token-ID sequences.

        Args:
            batch: List of token-ID sequences from the inference engine.

        Returns:
            List of canonicalized token-ID sequences for the training engine.

        Raises:
            ValueError: If any token ID in any sequence is out of range.
        """
        return [self.canonicalize(seq) for seq in batch]

    def validate_only(self, token_ids: list[int]) -> bool:
        """Check that all IDs in *token_ids* are valid without returning them.

        Args:
            token_ids: Token IDs to validate.

        Returns:
            ``True`` if every ID is in range (after any remapping).

        Raises:
            ValueError: On the first out-of-range ID encountered.
        """
        self.canonicalize(token_ids)
        return True
