"""Greedy sequence packing for efficient LLM training.

Concatenates multiple short sequences into fixed-length chunks to eliminate
padding waste. Tracks original sequence boundaries via sequence_ids, supports
per-sequence position ids, and builds block-diagonal attention masks that
prevent cross-sequence attention contamination.

Terminology used throughout this module:
    chunk       — a fixed-length output sample of exactly max_seq_len tokens
    sequence    — one input document / example (variable length)
    sequence_id — integer index of the source sequence; -1 for padding / EOS separators
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PackConfig:
    """Configuration for sequence packing.

    Attributes:
        max_seq_len:        Number of tokens per output chunk.
        pad_token_id:       Token id used to fill unused positions.
        eos_token_id:       Token id appended between packed sequences.
        add_eos:            Whether to insert an EOS token between sequences.
        loss_mask_padding:  Whether to set labels to -100 at padding positions.
    """

    max_seq_len: int = 2048
    pad_token_id: int = 0
    eos_token_id: int = 2
    add_eos: bool = True
    loss_mask_padding: bool = True


# ---------------------------------------------------------------------------
# Core packing logic
# ---------------------------------------------------------------------------


def pack_sequences(
    sequences: list[list[int]],
    config: PackConfig,
) -> list[dict]:
    """Greedily pack variable-length sequences into fixed-length chunks.

    Each sequence is optionally followed by an EOS token (config.add_eos).
    When a sequence (+ optional EOS) no longer fits in the current chunk, that
    chunk is closed and a new one is opened.  Sequences longer than max_seq_len
    are silently truncated to fit.

    Args:
        sequences: List of token-id lists, one per document / example.
        config:    Packing configuration.

    Returns:
        List of dicts, one per chunk, each containing:
            input_ids      (List[int]): token ids, length == max_seq_len
            attention_mask (List[int]): 1 for real tokens, 0 for padding
            labels         (List[int]): input_ids shifted for CLM;
                                        -100 at padding positions when
                                        config.loss_mask_padding is True
            sequence_ids   (List[int]): source sequence index per token;
                                        -1 for EOS separators and padding
    """
    chunks: list[dict] = []

    # Buffers for the chunk currently being built
    buf_tokens: list[int] = []
    buf_seq_ids: list[int] = []

    def _flush() -> None:
        """Pad the current buffer to max_seq_len and emit a chunk."""
        if not buf_tokens:
            return
        L = len(buf_tokens)
        pad_len = config.max_seq_len - L

        input_ids = buf_tokens + [config.pad_token_id] * pad_len
        attention_mask = [1] * L + [0] * pad_len
        sequence_ids = buf_seq_ids + [-1] * pad_len

        # Labels for causal LM: predict the *next* token at each position.
        # labels[t] = input_ids[t+1]; the last real position predicts pad/eos,
        # but that is masked out anyway.  We implement a simple copy-shifted
        # scheme: labels[t] = input_ids[t] (standard cross-entropy target for
        # next-token prediction in practice uses the raw ids; the dataloader or
        # model shifts internally).  The spec says "input_ids shifted right
        # (for CLM)", so we set labels == input_ids and let the model shift.
        labels = list(input_ids)

        # Mask padding in labels
        if config.loss_mask_padding:
            for t in range(L, config.max_seq_len):
                labels[t] = -100

        chunks.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "sequence_ids": sequence_ids,
            }
        )
        buf_tokens.clear()
        buf_seq_ids.clear()

    for seq_idx, seq in enumerate(sequences):
        # Build the segment: [seq_tokens] + optionally [EOS]
        segment_tokens: list[int] = list(seq)
        segment_seq_ids: list[int] = [seq_idx] * len(seq)

        if config.add_eos:
            segment_tokens.append(config.eos_token_id)
            segment_seq_ids.append(-1)  # EOS belongs to no sequence

        # Truncate if even a fresh chunk cannot hold this segment
        if len(segment_tokens) > config.max_seq_len:
            segment_tokens = segment_tokens[: config.max_seq_len]
            segment_seq_ids = segment_seq_ids[: config.max_seq_len]

        needed = len(segment_tokens)

        # Flush if this segment won't fit in the current chunk
        if buf_tokens and len(buf_tokens) + needed > config.max_seq_len:
            _flush()

        buf_tokens.extend(segment_tokens)
        buf_seq_ids.extend(segment_seq_ids)

    # Flush any remaining content
    _flush()

    return chunks


# ---------------------------------------------------------------------------
# Efficiency metric
# ---------------------------------------------------------------------------


def compute_packing_efficiency(
    sequences: list[list[int]],
    config: PackConfig,
) -> float:
    """Return the fraction of chunk tokens that carry real (non-padding) content.

    A value of 1.0 means perfectly packed (no padding); lower values indicate
    wasted capacity.

    Args:
        sequences: Input sequences (same format as pack_sequences).
        config:    Packing configuration.

    Returns:
        Float in (0, 1].  Returns 1.0 for empty input to avoid division by zero.
    """
    if not sequences:
        return 1.0

    chunks = pack_sequences(sequences, config)
    if not chunks:
        return 1.0

    total_positions = len(chunks) * config.max_seq_len
    real_tokens = sum(sum(c["attention_mask"]) for c in chunks)
    return real_tokens / total_positions


# ---------------------------------------------------------------------------
# Position ids
# ---------------------------------------------------------------------------


def create_position_ids(sequence_ids: list[int]) -> list[int]:
    """Create per-token position ids that reset at each new sequence boundary.

    The position counter resets to 0 whenever a new (non -1) sequence begins.
    EOS separators (sequence_id == -1) and padding keep position 0.

    Args:
        sequence_ids: Per-token sequence index list; -1 for EOS/padding.

    Returns:
        List of int, same length as sequence_ids.
    """
    position_ids: list[int] = []
    pos = 0
    prev_sid = None  # sentinel: no previous token

    for sid in sequence_ids:
        if sid == -1:
            # EOS separator or padding: position 0, do not advance counter
            position_ids.append(0)
            prev_sid = sid
        elif prev_sid is None or prev_sid == -1 or sid != prev_sid:
            # Start of a new (real) sequence
            pos = 0
            position_ids.append(pos)
            pos += 1
            prev_sid = sid
        else:
            # Continuing the same sequence
            position_ids.append(pos)
            pos += 1
            prev_sid = sid

    return position_ids


# ---------------------------------------------------------------------------
# Document attention mask
# ---------------------------------------------------------------------------


def build_document_attention_mask(sequence_ids: list[int]) -> Tensor:
    """Build a (T, T) boolean attention mask for a packed chunk.

    mask[i, j] is True iff token i can attend to token j, which requires:
        - sequence_ids[i] == sequence_ids[j]  (same document)
        - sequence_ids[i] != -1               (neither is padding/EOS)

    Note: this is a *bidirectional* within-document mask.  Layer-level causal
    masking is applied separately by the model.

    Args:
        sequence_ids: Per-token sequence index list; -1 for EOS/padding.

    Returns:
        Boolean tensor of shape (T, T).
    """
    T = len(sequence_ids)
    ids = torch.tensor(sequence_ids, dtype=torch.long)  # (T,)

    # ids_i[i, j] == ids[i], ids_j[i, j] == ids[j]
    ids_i = ids.unsqueeze(1).expand(T, T)  # (T, T)
    ids_j = ids.unsqueeze(0).expand(T, T)  # (T, T)

    same_seq = ids_i == ids_j  # same sequence id
    not_pad = ids_i != -1  # neither is padding (ids_i covers row; if sid[i]==-1 → blocked)

    return same_seq & not_pad  # (T, T) bool


# ---------------------------------------------------------------------------
# SequencePacker class
# ---------------------------------------------------------------------------


class SequencePacker:
    """Stateless wrapper around the sequence packing utilities.

    Example::

        cfg = PackConfig(max_seq_len=512)
        packer = SequencePacker(cfg)
        packed = packer.pack([[1, 2, 3], [4, 5]])
        orig   = packer.unpack(packed[0])
    """

    def __init__(self, config: PackConfig) -> None:
        self.config = config

    def pack(self, sequences: list[list[int]]) -> list[dict]:
        """Pack sequences into fixed-length chunks.

        Args:
            sequences: List of token-id lists.

        Returns:
            List of chunk dicts (see pack_sequences for field descriptions).
        """
        return pack_sequences(sequences, self.config)

    def efficiency(self, sequences: list[list[int]]) -> float:
        """Compute packing efficiency (non-padding token fraction).

        Args:
            sequences: List of token-id lists.

        Returns:
            Float in (0, 1].
        """
        return compute_packing_efficiency(sequences, self.config)

    def unpack(self, packed: dict) -> list[list[int]]:
        """Recover the original sequences from a packed chunk.

        EOS separators (sequence_id == -1) and padding tokens are excluded from
        the recovered sequences.

        Args:
            packed: A single chunk dict as returned by pack / pack_sequences.

        Returns:
            List of token-id lists, one per source sequence found in this chunk.
        """
        input_ids = packed["input_ids"]
        sequence_ids = packed["sequence_ids"]

        # Collect tokens grouped by sequence_id, preserving order
        recovered: dict[int, list[int]] = {}
        for token, sid in zip(input_ids, sequence_ids):
            if sid == -1:
                continue  # EOS / padding — skip
            if sid not in recovered:
                recovered[sid] = []
            recovered[sid].append(token)

        # Return in ascending order of sequence_id
        return [recovered[sid] for sid in sorted(recovered)]
