"""Fill-in-the-Middle (FIM) Language Model — document reordering for code infilling.

Reference: Bavarian et al., 2022, "Efficient Training of Language Models to Fill
in the Middle". https://arxiv.org/abs/2207.14255

FIM enables code infilling by rearranging a training document's token sequence so
the model learns to predict the *middle* of a document given both the prefix and
suffix as context.  Two reordering modes are supported:

    PSM (Prefix-Suffix-Middle):
        [FIM_PRE] prefix [FIM_SUF] suffix [FIM_MID] middle [EOT]

    SPM (Suffix-Prefix-Middle):
        [FIM_SUF] suffix [FIM_PRE] prefix [FIM_MID] middle [EOT]

During training, each document is independently sampled:
    - With probability ``fim_rate`` it is converted to FIM format.
    - Of those, ``spm_rate`` fraction use SPM; the rest use PSM.

The loss is computed only over the *middle* segment (tokens after FIM_MID up to
and including EOT), controlled by ``FIMLossFilter.make_loss_mask``.

Usage::

    config = FIMConfig(fim_rate=0.5, spm_rate=0.5)
    transformer = FIMTransformer(config)
    rng = random.Random(42)

    token_ids = list(range(200))
    fim_ids = transformer.transform(token_ids, rng)

    loss_filter = FIMLossFilter(config)
    mask = loss_filter.make_loss_mask(fim_ids)  # True for tokens to predict
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FIMConfig:
    """Configuration for FIMTransformer and FIMLossFilter.

    Args:
        fim_prefix_token_id:  Token id for the ``<fim_prefix>`` special token.
        fim_suffix_token_id:  Token id for the ``<fim_suffix>`` special token.
        fim_middle_token_id:  Token id for the ``<fim_middle>`` special token.
        eot_token_id:         Token id for the ``<|endoftext|>`` / EOT token.
        fim_rate:             Fraction of documents to transform into FIM
                              format (0.0–1.0).
        spm_rate:             Of the FIM-transformed documents, fraction to use
                              Suffix-Prefix-Middle (SPM) ordering vs
                              Prefix-Suffix-Middle (PSM) ordering.
        max_seq_len:          Maximum token sequence length; sequences are
                              truncated to this length after transformation.
        seed:                 Default RNG seed for ``transform_batch`` when
                              no explicit seed is provided.
    """

    fim_prefix_token_id: int = 100257  # <fim_prefix>
    fim_suffix_token_id: int = 100258  # <fim_suffix>
    fim_middle_token_id: int = 100259  # <fim_middle>
    eot_token_id: int = 100260        # <|endoftext|>
    fim_rate: float = 0.5
    spm_rate: float = 0.5
    max_seq_len: int = 8192
    seed: int = 42


# ---------------------------------------------------------------------------
# FIMDocument — holds the three segments of a split document
# ---------------------------------------------------------------------------


@dataclass
class FIMDocument:
    """Three-way split of a token sequence for FIM training.

    Attributes:
        prefix_ids:  Tokens before the split point (the visible prefix).
        suffix_ids:  Tokens after the second split point (the visible suffix).
        middle_ids:  Tokens between the two split points (the target to predict).
        mode:        Reordering mode: ``"spm"`` or ``"psm"``.
    """

    prefix_ids: List[int]
    suffix_ids: List[int]
    middle_ids: List[int]
    mode: str  # "spm" or "psm"


# ---------------------------------------------------------------------------
# FIMTransformer
# ---------------------------------------------------------------------------


class FIMTransformer:
    """Transforms token sequences into FIM format for training.

    Given a flat list of token ids, ``FIMTransformer`` optionally applies the
    fill-in-the-middle document reordering described by Bavarian et al. (2022).

    Splitting strategy:
        Two split points ``i`` and ``j`` (with ``i <= j``) are drawn uniformly
        at random from ``[0, len(token_ids)]``.  The document is then divided:
            prefix  = token_ids[:i]
            middle  = token_ids[i:j]
            suffix  = token_ids[j:]

        All three segments may be empty.  This matches the uniform-split
        sampling described in the paper.

    Args:
        config: ``FIMConfig`` instance.
    """

    def __init__(self, config: FIMConfig) -> None:
        self.config = config

    # ---------------------------------------------------------------------- #
    # Core split / reorder helpers                                            #
    # ---------------------------------------------------------------------- #

    def split_document(
        self,
        token_ids: List[int],
        rng: random.Random,
    ) -> FIMDocument:
        """Split ``token_ids`` into prefix / middle / suffix at two random points.

        Both split points are drawn independently and uniformly from
        ``[0, len(token_ids)]``, then sorted so that ``i <= j``.

        Args:
            token_ids: Flat list of integer token ids.
            rng:       ``random.Random`` instance for reproducible sampling.

        Returns:
            ``FIMDocument`` with the three segments and mode assigned based on
            ``config.spm_rate``.
        """
        n = len(token_ids)
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i > j:
            i, j = j, i

        prefix_ids = token_ids[:i]
        middle_ids = token_ids[i:j]
        suffix_ids = token_ids[j:]

        mode = "spm" if rng.random() < self.config.spm_rate else "psm"

        return FIMDocument(
            prefix_ids=prefix_ids,
            suffix_ids=suffix_ids,
            middle_ids=middle_ids,
            mode=mode,
        )

    def to_spm(self, doc: FIMDocument) -> List[int]:
        """Reorder ``doc`` using Suffix-Prefix-Middle (SPM) format.

        Output layout::

            [FIM_SUF] suffix [FIM_PRE] prefix [FIM_MID] middle [EOT]

        Args:
            doc: ``FIMDocument`` with prefix / middle / suffix segments.

        Returns:
            Flat list of token ids in SPM order.
        """
        cfg = self.config
        return (
            [cfg.fim_suffix_token_id]
            + doc.suffix_ids
            + [cfg.fim_prefix_token_id]
            + doc.prefix_ids
            + [cfg.fim_middle_token_id]
            + doc.middle_ids
            + [cfg.eot_token_id]
        )

    def to_psm(self, doc: FIMDocument) -> List[int]:
        """Reorder ``doc`` using Prefix-Suffix-Middle (PSM) format.

        Output layout::

            [FIM_PRE] prefix [FIM_SUF] suffix [FIM_MID] middle [EOT]

        Args:
            doc: ``FIMDocument`` with prefix / middle / suffix segments.

        Returns:
            Flat list of token ids in PSM order.
        """
        cfg = self.config
        return (
            [cfg.fim_prefix_token_id]
            + doc.prefix_ids
            + [cfg.fim_suffix_token_id]
            + doc.suffix_ids
            + [cfg.fim_middle_token_id]
            + doc.middle_ids
            + [cfg.eot_token_id]
        )

    # ---------------------------------------------------------------------- #
    # Public transform API                                                    #
    # ---------------------------------------------------------------------- #

    def transform(
        self,
        token_ids: List[int],
        rng: random.Random,
    ) -> List[int]:
        """Optionally apply FIM reordering to a single document.

        With probability ``config.fim_rate`` the document is split and reordered
        (using SPM or PSM depending on ``config.spm_rate``).  Otherwise the
        original tokens are returned with ``[EOT]`` appended.

        Args:
            token_ids: Flat list of integer token ids.
            rng:       ``random.Random`` instance for reproducible sampling.

        Returns:
            Transformed (or original + EOT) flat list of token ids, truncated
            to ``config.max_seq_len``.
        """
        if rng.random() < self.config.fim_rate:
            doc = self.split_document(token_ids, rng)
            if doc.mode == "spm":
                result = self.to_spm(doc)
            else:
                result = self.to_psm(doc)
        else:
            result = token_ids + [self.config.eot_token_id]

        return self.truncate_to_max(result)

    def transform_batch(
        self,
        batch: List[List[int]],
        seed: Optional[int] = None,
    ) -> List[List[int]]:
        """Transform a batch of token sequences.

        Each sequence is processed independently with its own per-document RNG
        derived from ``seed`` (or ``config.seed`` if ``seed`` is ``None``).

        Args:
            batch: List of flat token-id lists to transform.
            seed:  Optional integer seed; falls back to ``config.seed``.

        Returns:
            List of transformed token-id lists, one per input document.
        """
        effective_seed = seed if seed is not None else self.config.seed
        base_rng = random.Random(effective_seed)

        results: List[List[int]] = []
        for token_ids in batch:
            # Give each document a deterministic but independent sub-seed.
            doc_seed = base_rng.randint(0, 2**31 - 1)
            doc_rng = random.Random(doc_seed)
            results.append(self.transform(token_ids, doc_rng))

        return results

    def truncate_to_max(self, token_ids: List[int]) -> List[int]:
        """Truncate ``token_ids`` to ``config.max_seq_len`` tokens.

        Args:
            token_ids: Flat list of token ids.

        Returns:
            List of at most ``config.max_seq_len`` tokens.
        """
        return token_ids[: self.config.max_seq_len]


# ---------------------------------------------------------------------------
# FIMLossFilter
# ---------------------------------------------------------------------------


class FIMLossFilter:
    """Computes a per-token loss mask for FIM-formatted sequences.

    In FIM training the model is only trained to predict the *middle* segment
    (tokens that appear after the ``FIM_MID`` special token).  All tokens
    before and including ``FIM_MID`` are masked out (``False``).

    For non-FIM sequences (i.e. sequences that do not contain ``FIM_MID``) the
    standard causal language-model objective applies and every token is
    predicted (all ``True``).

    Args:
        config: ``FIMConfig`` instance.
    """

    def __init__(self, config: FIMConfig) -> None:
        self.config = config

    def make_loss_mask(self, token_ids: List[int]) -> List[bool]:
        """Build a boolean loss mask for ``token_ids``.

        Args:
            token_ids: Flat list of integer token ids (possibly FIM-formatted).

        Returns:
            List of ``bool`` values, one per token.  ``True`` means the token
            contributes to the loss.

            - If ``FIM_MID`` is found: tokens *after* the first ``FIM_MID``
              occurrence get ``True``; all others (including ``FIM_MID``
              itself) get ``False``.
            - If ``FIM_MID`` is not found: all tokens get ``True`` (standard
              causal LM behaviour).
        """
        fim_mid = self.config.fim_middle_token_id
        try:
            mid_pos = token_ids.index(fim_mid)
        except ValueError:
            # No FIM_MID present — standard language model; predict every token.
            return [True] * len(token_ids)

        # Predict only the middle segment (tokens after FIM_MID).
        mask = [False] * len(token_ids)
        for idx in range(mid_pos + 1, len(token_ids)):
            mask[idx] = True
        return mask


# ---------------------------------------------------------------------------
# Registry registration — see src/model/__init__.py
# MODEL_COMPONENT_REGISTRY["fim_transformer"] = FIMTransformer
# (registered there to avoid circular-import issues)
