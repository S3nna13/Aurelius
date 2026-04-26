"""Greedy bin-packing of variable-length sequences into fixed-length chunks.

Implements document boundary tracking and cross-document loss masking for
efficient LLM training.  Multiple short documents are packed into a single
max_seq_len chunk separated by EOS tokens; a block-diagonal causal attention
mask prevents cross-document attention contamination.

References:
    - SmolLM3 (HuggingFace, 2025)
    - SkyLadder (2025)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration & result types
# ---------------------------------------------------------------------------


@dataclass
class PackingConfig:
    max_seq_len: int = 8192
    pad_token_id: int = 0
    eos_token_id: int = 2
    loss_on_eos: bool = True  # include EOS positions in loss
    cross_doc_loss_mask: bool = True  # mask first token of each new doc


@dataclass
class PackedBatch:
    input_ids: Tensor  # (n_chunks, max_seq_len)
    labels: Tensor  # (n_chunks, max_seq_len) — -100 for masked
    attention_mask: Tensor  # (n_chunks, max_seq_len) — 0 for padding, 1 real
    doc_ids: Tensor  # (n_chunks, max_seq_len) — document index per token
    n_docs_packed: int  # total documents packed across all chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_chunk_tensors(
    chunk_tokens: list[int],
    chunk_doc_ids: list[int],
    cfg: PackingConfig,
    global_doc_offset: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Pad a raw chunk to max_seq_len and return (input_ids, labels, attn_mask, doc_ids).

    All four lists have length max_seq_len.
    """
    L = len(chunk_tokens)
    pad_len = cfg.max_seq_len - L

    # input_ids
    input_ids = chunk_tokens + [cfg.pad_token_id] * pad_len

    # doc_ids — pad positions get sentinel value (max_doc_id + 1)
    max_doc_id = max(chunk_doc_ids) if chunk_doc_ids else 0
    sentinel_doc = max_doc_id + 1
    doc_ids_out = chunk_doc_ids + [sentinel_doc] * pad_len

    # attention_mask: 1 for real tokens, 0 for padding
    attn_mask = [1] * L + [0] * pad_len

    # labels: start as copy of input_ids, then mask
    labels = list(input_ids)

    # Mask padding
    for t in range(L, cfg.max_seq_len):
        labels[t] = -100

    # Mask EOS from loss if not loss_on_eos
    if not cfg.loss_on_eos:
        for t in range(L):
            if input_ids[t] == cfg.eos_token_id:
                labels[t] = -100

    # Cross-doc loss masking: mask first token of each new document
    if cfg.cross_doc_loss_mask:
        for t in range(1, L):
            if chunk_doc_ids[t] != chunk_doc_ids[t - 1]:
                labels[t] = -100

    return input_ids, labels, attn_mask, doc_ids_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def greedy_pack(
    sequences: list[Tensor],
    cfg: PackingConfig,
) -> PackedBatch:
    """Greedy bin-packing: fill each chunk greedily with EOS separator between docs.

    Layout per chunk: [doc1_tokens, EOS, doc2_tokens, EOS, ..., PAD, PAD, ...]

    Args:
        sequences: List of 1-D token-id tensors (variable length).
        cfg: Packing configuration.

    Returns:
        PackedBatch with tensors of shape (n_chunks, max_seq_len).
    """
    all_input_ids: list[list[int]] = []
    all_labels: list[list[int]] = []
    all_attn: list[list[int]] = []
    all_doc_ids: list[list[int]] = []

    chunk_tokens: list[int] = []
    chunk_doc_ids: list[int] = []
    doc_counter_in_chunk = 0  # local doc index within current chunk

    def _flush_chunk() -> None:
        """Pad and append the current chunk to the output lists."""
        if not chunk_tokens:
            return
        inp, lab, atm, dids = _build_chunk_tensors(
            chunk_tokens, chunk_doc_ids, cfg, global_doc_offset=0
        )
        all_input_ids.append(inp)
        all_labels.append(lab)
        all_attn.append(atm)
        all_doc_ids.append(dids)

    for seq in sequences:
        seq_list: list[int] = seq.tolist()

        # A single document plus its trailing EOS occupies (len + 1) positions
        doc_with_eos = seq_list + [cfg.eos_token_id]
        needed = len(doc_with_eos)

        # If it won't fit in the current chunk, flush and start a new one
        if chunk_tokens and len(chunk_tokens) + needed > cfg.max_seq_len:
            _flush_chunk()
            chunk_tokens = []
            chunk_doc_ids = []
            doc_counter_in_chunk = 0

        # If even a fresh chunk can't hold this document, truncate it
        if needed > cfg.max_seq_len:
            doc_with_eos = doc_with_eos[: cfg.max_seq_len]
            needed = cfg.max_seq_len

        chunk_doc_ids.extend([doc_counter_in_chunk] * needed)
        chunk_tokens.extend(doc_with_eos)
        doc_counter_in_chunk += 1

    # Flush the final chunk
    _flush_chunk()

    if not all_input_ids:
        # Edge case: no sequences provided
        empty = torch.zeros(0, cfg.max_seq_len, dtype=torch.long)
        return PackedBatch(
            input_ids=empty,
            labels=empty.clone(),
            attention_mask=empty.clone(),
            doc_ids=empty.clone(),
            n_docs_packed=0,
        )

    return PackedBatch(
        input_ids=torch.tensor(all_input_ids, dtype=torch.long),
        labels=torch.tensor(all_labels, dtype=torch.long),
        attention_mask=torch.tensor(all_attn, dtype=torch.long),
        doc_ids=torch.tensor(all_doc_ids, dtype=torch.long),
        n_docs_packed=len(sequences),
    )


def compute_document_mask(doc_ids: Tensor) -> Tensor:
    """Return a boolean boundary mask of shape (n_chunks, max_seq_len).

    True at position t means a document boundary occurs there (doc_ids[i,t] !=
    doc_ids[i,t-1]).  Position 0 is always False.

    Args:
        doc_ids: (n_chunks, max_seq_len) integer tensor.

    Returns:
        Boolean tensor of the same shape.
    """
    # Shift right to compare adjacent positions
    # boundary[i, t] = (doc_ids[i, t] != doc_ids[i, t-1])
    boundary = torch.zeros_like(doc_ids, dtype=torch.bool)
    boundary[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    return boundary


def create_block_diagonal_attention_mask(doc_ids: Tensor) -> Tensor:
    """Build a block-diagonal causal attention mask for packed sequences.

    Each chunk gets a (max_seq_len, max_seq_len) float mask where:
        - 0.0  means "allowed to attend" (same document, causal)
        - -inf means "blocked"           (different document, or future token)

    Args:
        doc_ids: (n_chunks, max_seq_len) integer tensor.

    Returns:
        Float tensor of shape (n_chunks, max_seq_len, max_seq_len).
    """
    n_chunks, seq_len = doc_ids.shape
    NEG_INF = float("-inf")

    # same_doc[b, i, j] = True if doc_ids[b,i] == doc_ids[b,j]
    # Shape: (n_chunks, seq_len, seq_len)
    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)

    # Causal mask: lower-triangle (i >= j)
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=doc_ids.device).tril()

    # Combined: attend iff same doc AND causal
    attend = same_doc & causal.unsqueeze(0)  # (n_chunks, seq_len, seq_len)

    # Convert to additive float mask
    mask = torch.where(
        attend,
        torch.zeros(1, device=doc_ids.device),
        torch.full((1,), NEG_INF, device=doc_ids.device),
    )
    return mask


def pack_dataset(
    token_ids_list: list[Tensor],
    cfg: PackingConfig,
) -> list[PackedBatch]:
    """Convenience wrapper: pack all sequences and return a list of PackedBatch.

    If everything fits in a single call to greedy_pack, the list has one entry.

    Args:
        token_ids_list: List of 1-D token-id tensors.
        cfg: Packing configuration.

    Returns:
        List containing one or more PackedBatch objects.
    """
    batch = greedy_pack(token_ids_list, cfg)
    return [batch]
