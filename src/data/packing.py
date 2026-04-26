"""Intra-document masking for sequence packing.

When multiple short documents are packed into a single training sequence,
standard causal masking allows tokens in later documents to attend to tokens
in earlier documents (cross-contamination). This module provides utilities
to build document-boundary-aware attention masks that prevent this.

Reference: SmolLM3 (HuggingFace, 2025), SkyLadder (2025).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PackedSequence:
    """A sequence packed from multiple documents with boundary metadata."""

    input_ids: list[int]
    doc_ids: list[int]  # document index for each token position
    doc_lengths: list[int]  # length of each original document


def pack_sequences(
    token_sequences: list[list[int]],
    max_len: int,
    pad_token_id: int = 0,
) -> list[PackedSequence]:
    """Pack variable-length token sequences into fixed-length chunks.

    Greedily fills each chunk to max_len. Documents that don't fit start
    a new chunk. Does NOT split documents across chunk boundaries.

    Args:
        token_sequences: List of tokenized documents (list of int).
        max_len: Maximum sequence length per packed chunk.
        pad_token_id: Token ID used for padding the final chunk.

    Returns:
        List of PackedSequence objects, each of length <= max_len.
    """
    packed: list[PackedSequence] = []
    current_ids: list[int] = []
    current_doc_ids: list[int] = []
    current_doc_lengths: list[int] = []
    doc_idx = 0

    for seq in token_sequences:
        if len(seq) > max_len:
            # Document too long: truncate and emit as a standalone chunk
            seq = seq[:max_len]

        if len(current_ids) + len(seq) > max_len:
            # Current chunk is full: emit it and start a new one
            if current_ids:
                packed.append(
                    PackedSequence(
                        input_ids=current_ids,
                        doc_ids=current_doc_ids,
                        doc_lengths=current_doc_lengths,
                    )
                )
            current_ids = []
            current_doc_ids = []
            current_doc_lengths = []
            doc_idx = 0

        current_doc_ids.extend([doc_idx] * len(seq))
        current_doc_lengths.append(len(seq))
        current_ids.extend(seq)
        doc_idx += 1

    # Emit remaining tokens
    if current_ids:
        packed.append(
            PackedSequence(
                input_ids=current_ids,
                doc_ids=current_doc_ids,
                doc_lengths=current_doc_lengths,
            )
        )

    return packed


def build_intra_doc_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Build a causal attention mask that respects document boundaries.

    Produces a boolean mask of shape (S, S) where mask[i, j] = True means
    token i CAN attend to token j. Token i can attend to token j iff:
        (1) j <= i (causal)
        (2) doc_ids[i] == doc_ids[j] (same document)

    This is the correct mask to pass to F.scaled_dot_product_attention
    as attn_mask (where True = allowed, False = masked out).

    Args:
        doc_ids: 1D integer tensor of shape (S,) mapping each position to a doc index.

    Returns:
        Boolean tensor of shape (S, S). True = can attend.
    """
    S = doc_ids.shape[0]
    # Causal mask: lower triangle
    causal = torch.ones(S, S, dtype=torch.bool, device=doc_ids.device).tril()
    # Same-document mask
    same_doc = doc_ids.unsqueeze(0) == doc_ids.unsqueeze(1)  # (S, S)
    return causal & same_doc


def collate_packed_batch(
    packed_sequences: list[PackedSequence],
    max_len: int,
    pad_token_id: int = 0,
    return_mask: bool = True,
) -> dict[str, Any]:
    """Collate a batch of PackedSequences into tensors.

    Args:
        packed_sequences: List of PackedSequence objects.
        max_len: Sequence length to pad/truncate to.
        pad_token_id: Token used for right-padding.
        return_mask: If True, include per-sequence intra-doc attention masks.

    Returns:
        Dict with keys:
            "input_ids": (B, max_len) int64 tensor
            "labels":    (B, max_len) int64 tensor (same as input_ids, -100 on pad)
            "attention_mask": (B, max_len, max_len) bool tensor (if return_mask=True)
    """
    B = len(packed_sequences)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    masks = [] if return_mask else None

    for i, ps in enumerate(packed_sequences):
        ids = ps.input_ids[:max_len]
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        labels[i, :L] = torch.tensor(ids, dtype=torch.long)

        if return_mask:
            doc_ids_tensor = torch.tensor(ps.doc_ids[:max_len], dtype=torch.long)
            # Pad doc_ids with unique sentinels so padded positions don't attend to anything
            if L < max_len:
                sentinel = doc_ids_tensor.max().item() + 1 if L > 0 else 0
                # Each pad position gets a unique doc_id so it only attends to itself
                pad_doc_ids = torch.arange(max_len - L, dtype=torch.long) + (sentinel + 1)
                doc_ids_tensor = torch.cat([doc_ids_tensor, pad_doc_ids])
            masks.append(build_intra_doc_mask(doc_ids_tensor))

    result: dict[str, Any] = {"input_ids": input_ids, "labels": labels}
    if return_mask:
        result["attention_mask"] = torch.stack(masks)  # (B, max_len, max_len)
    return result
