"""Dynamic batching: group sequences of similar length to minimize padding waste.

Sequences are sorted by length and packed greedily so that each batch respects
a maximum token budget.  This reduces padding overhead compared to random
shuffling, which is important when training on variable-length text corpora.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BatchConfig:
    max_tokens_per_batch: int = 2048
    max_batch_size: int = 32
    pad_token_id: int = 0
    sort_by_length: bool = True
    drop_last: bool = False


# ---------------------------------------------------------------------------
# Standalone utilities
# ---------------------------------------------------------------------------


def pad_sequence_batch(
    sequences: list[Tensor],
    pad_value: int = 0,
) -> tuple[Tensor, Tensor]:
    """Pad a list of 1-D sequences to the length of the longest one.

    Args:
        sequences: List of 1-D integer tensors of variable length.
        pad_value: Token id used for padding positions.

    Returns:
        padded_batch: (B, T_max) long tensor.
        attention_mask: (B, T_max) long tensor; 1 for real tokens, 0 for padding.
    """
    if not sequences:
        empty = torch.zeros(0, 0, dtype=torch.long)
        return empty, empty

    max_len = max(s.shape[0] for s in sequences)
    B = len(sequences)

    padded = torch.full((B, max_len), fill_value=pad_value, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)

    for i, seq in enumerate(sequences):
        L = seq.shape[0]
        padded[i, :L] = seq
        attention_mask[i, :L] = 1

    return padded, attention_mask


def compute_padding_ratio(sequences: list[Tensor]) -> float:
    """Compute the average fraction of padding tokens if sequences were batched together.

    The batch is padded to the length of the longest sequence.  The ratio is
    defined as ``(total_padding_positions) / (B * T_max)`` where ``T_max`` is the
    maximum sequence length in the batch.

    Returns:
        Float in [0.0, 1.0].  0.0 means no padding needed; 1.0 would mean
        every token position is padding (degenerate case of empty sequences).
    """
    if not sequences:
        return 0.0

    max_len = max(s.shape[0] for s in sequences)
    if max_len == 0:
        return 0.0

    total_positions = len(sequences) * max_len
    real_tokens = sum(s.shape[0] for s in sequences)
    padding_tokens = total_positions - real_tokens
    return padding_tokens / total_positions


def bucket_by_length(
    sequences: list[Tensor],
    n_buckets: int,
) -> list[list[Tensor]]:
    """Sort sequences by length and split into n_buckets equal-sized groups.

    If ``len(sequences)`` is not divisible by ``n_buckets``, the final bucket
    receives the remaining sequences.

    Args:
        sequences: List of 1-D tensors.
        n_buckets: Number of buckets to create.

    Returns:
        List of ``n_buckets`` lists, each containing roughly the same number
        of sequences, sorted by ascending length within each bucket.
    """
    if n_buckets <= 0:
        raise ValueError("n_buckets must be >= 1")
    if not sequences:
        return [[] for _ in range(n_buckets)]

    sorted_seqs = sorted(sequences, key=lambda s: s.shape[0])
    bucket_size = max(1, len(sorted_seqs) // n_buckets)

    buckets: list[list[Tensor]] = []
    for i in range(n_buckets):
        start = i * bucket_size
        if i < n_buckets - 1:
            end = start + bucket_size
        else:
            end = len(sorted_seqs)  # last bucket gets the remainder
        buckets.append(sorted_seqs[start:end])

    return buckets


def greedy_pack(
    sequences: list[Tensor],
    max_tokens: int,
) -> list[list[Tensor]]:
    """Greedily pack sequences into bins where total tokens <= max_tokens.

    Sequences are sorted by descending length before packing so that longer
    sequences are placed first (first-fit decreasing heuristic), which tends
    to produce fewer, fuller bins.

    Args:
        sequences: List of 1-D tensors.
        max_tokens: Maximum total tokens per bin.

    Returns:
        List of bins, each bin being a list of tensors.
    """
    if not sequences:
        return []

    sorted_seqs = sorted(sequences, key=lambda s: s.shape[0], reverse=True)

    bins: list[list[Tensor]] = []
    bin_sizes: list[int] = []

    for seq in sorted_seqs:
        L = seq.shape[0]
        # Try to fit into an existing bin
        placed = False
        for idx, current_size in enumerate(bin_sizes):
            if current_size + L <= max_tokens:
                bins[idx].append(seq)
                bin_sizes[idx] += L
                placed = True
                break
        if not placed:
            # Open a new bin (even if the sequence alone exceeds max_tokens,
            # it must still go somewhere — avoid silently dropping data).
            bins.append([seq])
            bin_sizes.append(L)

    return bins


# ---------------------------------------------------------------------------
# DynamicBatcher
# ---------------------------------------------------------------------------


class DynamicBatcher:
    """Groups variable-length sequences into padded batches while respecting
    a per-batch token budget.

    Args:
        config: BatchConfig instance controlling behaviour.
    """

    def __init__(self, config: BatchConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def collate(self, sequences: list[Tensor]) -> dict[str, Tensor]:
        """Pad a list of sequences into a single batch dict.

        If ``config.sort_by_length`` is True the sequences are sorted by
        ascending length before padding (this has no effect on efficiency when
        all sequences are the same length, but it keeps within-batch ordering
        deterministic).

        Args:
            sequences: List of 1-D token-id tensors.

        Returns:
            dict with keys:
                ``"input_ids"``:      (B, T_max) long tensor.
                ``"attention_mask"``: (B, T_max) long tensor.
        """
        if self.config.sort_by_length:
            sequences = sorted(sequences, key=lambda s: s.shape[0])

        padded, attention_mask = pad_sequence_batch(sequences, pad_value=self.config.pad_token_id)
        return {"input_ids": padded, "attention_mask": attention_mask}

    def create_batches(self, sequences: list[Tensor]) -> list[dict[str, Tensor]]:
        """Partition sequences into batches that respect the token budget.

        Algorithm:
          1. Sort all sequences by ascending length to minimise padding.
          2. Use a greedy scan: keep adding sequences to the current group
             until ``max_tokens_per_batch`` or ``max_batch_size`` would be
             exceeded, then start a new group.
          3. Collate each group.
          4. If ``drop_last`` is True the final (potentially smaller) batch
             is discarded.

        Args:
            sequences: List of 1-D token-id tensors (the full dataset).

        Returns:
            List of batch dicts, each with ``"input_ids"`` and
            ``"attention_mask"``.
        """
        if not sequences:
            return []

        # Sort by ascending length for padding efficiency
        sorted_seqs = sorted(sequences, key=lambda s: s.shape[0])

        batches: list[dict[str, Tensor]] = []
        current_group: list[Tensor] = []
        current_max_len = 0

        for seq in sorted_seqs:
            L = seq.shape[0]
            # After adding this sequence the total token footprint would be
            # (current_max_len + L) * (len(current_group) + 1) — actually, since
            # we pad to T_max, the cost is T_max * B.  Use the prospective T_max.
            prospective_max_len = max(current_max_len, L)
            prospective_b = len(current_group) + 1
            prospective_tokens = prospective_max_len * prospective_b

            would_exceed_tokens = prospective_tokens > self.config.max_tokens_per_batch
            would_exceed_size = prospective_b > self.config.max_batch_size

            if current_group and (would_exceed_tokens or would_exceed_size):
                batches.append(self.collate(current_group))
                current_group = []
                current_max_len = 0

            current_group.append(seq)
            current_max_len = max(current_max_len, L)

        # Handle the final (possibly partial) batch
        if current_group:
            if not self.config.drop_last:
                batches.append(self.collate(current_group))

        return batches

    def compute_efficiency(self, batches: list[dict[str, Tensor]]) -> float:
        """Compute padding efficiency across all batches.

        Efficiency = real_tokens / total_tokens, where ``total_tokens``
        includes padding positions.

        Args:
            batches: List of batch dicts as returned by ``create_batches``.

        Returns:
            Float in (0.0, 1.0].  1.0 means zero padding waste.
        """
        if not batches:
            return 1.0

        total_tokens = 0
        real_tokens = 0
        for batch in batches:
            mask: Tensor = batch["attention_mask"]
            total_tokens += mask.numel()
            real_tokens += int(mask.sum().item())

        if total_tokens == 0:
            return 1.0

        return real_tokens / total_tokens
