"""Sequence packing for efficient LLM training (a.k.a. sample packing / multipack).

Bins multiple short sequences into fixed-length chunks to eliminate padding waste.
Uses first-fit-decreasing (FFD) bin packing: sort sequences by length descending,
then greedily assign each sequence to the first bin where it fits.

References:
  - OpenAI packing approach (GPT-3 SFT)
  - LLaMA-Factory multipack sampler
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PackedSequence:
    """A single packed chunk containing one or more concatenated sequences.

    Attributes:
        token_ids:      1D tensor of concatenated token ids (length == max_length
                        after padding).
        position_ids:   1D tensor; position counter resets to 0 at each sub-sequence
                        start, so sub-sequence boundaries are visible.
        seq_boundaries: Start index (in token_ids) of each sub-sequence.
        labels:         Optional 1D tensor aligned with token_ids; padding positions
                        are filled with -100 so they are ignored by cross-entropy.
    """
    token_ids: Tensor
    position_ids: Tensor
    seq_boundaries: List[int]
    labels: Optional[Tensor] = field(default=None)


# ---------------------------------------------------------------------------
# Core packer
# ---------------------------------------------------------------------------

class SequencePacker:
    """Bins a list of 1-D token-id tensors into packed chunks via FFD.

    Args:
        max_length:    Maximum number of tokens per packed chunk (the bin capacity).
        pad_token_id:  Token id used for padding (default 0).
    """

    def __init__(self, max_length: int, pad_token_id: int = 0) -> None:
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_bins(self, sequences: List[Tensor]) -> List[List[int]]:
        """First-fit decreasing bin packing.

        Returns a list of bins, where each bin is a list of original indices
        (into `sequences`) assigned to that bin.  Sequences longer than
        max_length are truncated before assignment.
        """
        # Compute (capped) lengths and sort descending
        lengths = [min(len(s), self.max_length) for s in sequences]
        order = sorted(range(len(sequences)), key=lambda i: lengths[i], reverse=True)

        bin_contents: List[List[int]] = []   # list of [seq_idx, ...]
        bin_remaining: List[int] = []        # remaining capacity per bin

        for idx in order:
            seq_len = lengths[idx]
            placed = False
            for b in range(len(bin_contents)):
                if bin_remaining[b] >= seq_len:
                    bin_contents[b].append(idx)
                    bin_remaining[b] -= seq_len
                    placed = True
                    break
            if not placed:
                bin_contents.append([idx])
                bin_remaining.append(self.max_length - seq_len)

        return bin_contents

    def _assemble_packed(
        self,
        bin_indices: List[int],
        sequences: List[Tensor],
        labels_list: Optional[List[Tensor]] = None,
    ) -> PackedSequence:
        """Concatenate sequences in a bin and pad to max_length."""
        parts_tokens: List[Tensor] = []
        parts_pos: List[Tensor] = []
        parts_labels: List[Tensor] = []
        seq_boundaries: List[int] = []
        cursor = 0

        for idx in bin_indices:
            seq = sequences[idx][:self.max_length]  # truncate if > max_length
            seq_len = len(seq)
            seq_boundaries.append(cursor)

            parts_tokens.append(seq)
            parts_pos.append(torch.arange(seq_len, dtype=torch.long))

            if labels_list is not None:
                lbl = labels_list[idx][:self.max_length]
                parts_labels.append(lbl)

            cursor += seq_len

        # Concatenate real tokens
        token_ids = torch.cat(parts_tokens)
        position_ids = torch.cat(parts_pos)

        real_len = len(token_ids)
        pad_len = self.max_length - real_len

        # Pad token_ids and position_ids
        if pad_len > 0:
            token_ids = torch.cat([
                token_ids,
                torch.full((pad_len,), self.pad_token_id, dtype=token_ids.dtype),
            ])
            # Position ids for padding: continue counting (or 0 — we use 0)
            position_ids = torch.cat([
                position_ids,
                torch.zeros(pad_len, dtype=torch.long),
            ])

        # Handle labels
        labels: Optional[Tensor] = None
        if labels_list is not None:
            labels = torch.cat(parts_labels)
            if pad_len > 0:
                labels = torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=labels.dtype),
                ])

        return PackedSequence(
            token_ids=token_ids,
            position_ids=position_ids,
            seq_boundaries=seq_boundaries,
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pack(self, sequences: List[Tensor]) -> List[PackedSequence]:
        """Pack a list of 1-D token-id tensors into PackedSequence chunks.

        Each chunk has length exactly max_length (padded if necessary).
        Position IDs reset to 0 at the start of each sub-sequence.

        Args:
            sequences: List of 1-D LongTensors (variable length).

        Returns:
            List of PackedSequence objects, one per bin.
        """
        if not sequences:
            return []

        bins = self._build_bins(sequences)
        return [self._assemble_packed(b, sequences, None) for b in bins]

    def pack_with_labels(
        self,
        sequences: List[Tensor],
        labels: List[Tensor],
    ) -> List[PackedSequence]:
        """Pack token-id tensors together with aligned label tensors.

        Padding positions in labels are filled with -100 (ignored by CE loss).

        Args:
            sequences: List of 1-D LongTensors (token ids).
            labels:    List of 1-D LongTensors aligned to sequences; same lengths.

        Returns:
            List of PackedSequence objects with the labels field populated.
        """
        if len(sequences) != len(labels):
            raise ValueError(
                f"sequences and labels must have the same length, "
                f"got {len(sequences)} vs {len(labels)}"
            )
        if not sequences:
            return []

        bins = self._build_bins(sequences)
        return [self._assemble_packed(b, sequences, labels) for b in bins]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class PackingStats:
    """Packing efficiency statistics.

    Attributes:
        n_sequences:    Total number of input sequences.
        n_bins:         Number of bins (packed chunks) produced.
        total_tokens:   Sum of all input sequence lengths (after capping at max_length).
        padding_tokens: Number of padding tokens added (n_bins * max_length - total_tokens).
        efficiency:     Fraction of non-padding tokens = total_tokens / (n_bins * max_length).
    """
    n_sequences: int
    n_bins: int
    total_tokens: int
    padding_tokens: int
    efficiency: float

    def __init__(
        self,
        sequences: List[Tensor],
        bins: List[List[int]],
        max_length: int,
    ) -> None:
        self.n_sequences = len(sequences)
        self.n_bins = len(bins)
        self.total_tokens = sum(min(len(s), max_length) for s in sequences)
        capacity = self.n_bins * max_length
        self.padding_tokens = capacity - self.total_tokens
        self.efficiency = self.total_tokens / capacity if capacity > 0 else 0.0

    @classmethod
    def from_packer(
        cls,
        packer: SequencePacker,
        sequences: List[Tensor],
    ) -> "PackingStats":
        """Convenience factory: compute stats from a packer and sequences."""
        bins = packer._build_bins(sequences)
        return cls(sequences, bins, packer.max_length)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class PackedBatchCollator:
    """PyTorch collate_fn that packs a batch of dict items into padded tensors.

    Each item in the batch must have:
        'input_ids': List[int] or 1-D LongTensor of token ids.
        'labels':    (optional) List[int] or 1-D LongTensor aligned with input_ids.

    The collator packs all items, then returns a dict with:
        'input_ids':      (B, L) LongTensor
        'position_ids':   (B, L) LongTensor
        'attention_mask': (B, L) BoolTensor — True for real tokens, False for padding
        'labels':         (B, L) LongTensor — -100 for padding positions

    where B = number of bins and L = packer.max_length.

    Args:
        packer: A configured SequencePacker instance.
    """

    def __init__(self, packer: SequencePacker) -> None:
        self.packer = packer

    def __call__(self, batch: List[Dict]) -> Dict[str, Tensor]:
        sequences: List[Tensor] = []
        labels_list: Optional[List[Tensor]] = []
        has_labels = any("labels" in item for item in batch)

        for item in batch:
            ids = item["input_ids"]
            if not isinstance(ids, Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            else:
                ids = ids.long()
            sequences.append(ids)

            if has_labels:
                lbl = item.get("labels", None)
                if lbl is None:
                    # Fill with -100 when not provided for this item
                    lbl = torch.full_like(ids, -100)
                elif not isinstance(lbl, Tensor):
                    lbl = torch.tensor(lbl, dtype=torch.long)
                else:
                    lbl = lbl.long()
                labels_list.append(lbl)

        if has_labels:
            packed = self.packer.pack_with_labels(sequences, labels_list)
        else:
            packed = self.packer.pack(sequences)

        max_length = self.packer.max_length
        B = len(packed)

        input_ids_out = torch.zeros(B, max_length, dtype=torch.long)
        position_ids_out = torch.zeros(B, max_length, dtype=torch.long)
        attention_mask_out = torch.zeros(B, max_length, dtype=torch.bool)
        labels_out = torch.full((B, max_length), -100, dtype=torch.long)

        for i, ps in enumerate(packed):
            input_ids_out[i] = ps.token_ids
            position_ids_out[i] = ps.position_ids

            # Build attention mask from seq_boundaries and token_ids
            # Real tokens are those not equal to pad_token_id in padding positions.
            # More precisely: real positions are everything up to total real length,
            # which we can derive from seq_boundaries + individual seq lengths.
            # We mark padding positions (token_id == pad_token_id AND position > last real token).
            # Simplest: a token is real iff it belongs to a sub-sequence (not padding).
            # We reconstruct real_len from seq_boundaries + the packed sequences lengths.
            real_len = self._compute_real_len(ps, sequences, packed, i)
            attention_mask_out[i, :real_len] = True

            if ps.labels is not None:
                labels_out[i] = ps.labels
            else:
                # No labels provided: fill real positions with -100 too
                pass  # already -100 everywhere

        return {
            "input_ids": input_ids_out,
            "position_ids": position_ids_out,
            "attention_mask": attention_mask_out,
            "labels": labels_out,
        }

    def _compute_real_len(
        self,
        ps: PackedSequence,
        sequences: List[Tensor],
        packed: List[PackedSequence],
        bin_idx: int,
    ) -> int:
        """Determine number of real (non-padding) tokens in this packed sequence.

        We use position_ids: padding tokens have position_id == 0 and sit after
        all real tokens. But that's ambiguous when the last real token also has
        position 0 (single-token sequences). Instead we look at the token_ids
        directly: the padding region is the suffix filled with pad_token_id
        beyond the real content.

        Robust approach: count tokens until we hit the first padding suffix.
        We iterate backwards from max_length.
        """
        tids = ps.token_ids
        L = len(tids)
        # Walk backward to find where real tokens end
        real_len = L
        for j in range(L - 1, -1, -1):
            if tids[j].item() == self.packer.pad_token_id:
                real_len = j
            else:
                break
        return real_len
