"""Sequence parallelism simulation: partition sequences across virtual ranks for long-context training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SeqParallelConfig:
    """Configuration for sequence parallelism simulation."""

    world_size: int = 4
    overlap_tokens: int = 0
    ring_attn: bool = True
    load_balance: bool = True


def partition_sequence(
    input_ids: Tensor, world_size: int, overlap: int = 0
) -> list[Tensor]:
    """Split (B, T) tensor into *world_size* chunks along the T dimension.

    If *overlap* > 0, each chunk (except the first) is extended by *overlap*
    tokens from the preceding chunk so that attention has surrounding context.

    When T is not evenly divisible by *world_size* the last chunk absorbs the
    remainder (it may be smaller or larger depending on rounding).
    """
    B, T = input_ids.shape
    chunk_size = T // world_size
    chunks: list[Tensor] = []

    for i in range(world_size):
        start = i * chunk_size
        if i == world_size - 1:
            end = T  # last chunk gets remainder
        else:
            end = start + chunk_size

        # extend with overlap from previous chunk
        if overlap > 0 and i > 0:
            overlap_start = max(0, start - overlap)
            chunk = input_ids[:, overlap_start:end]
        else:
            chunk = input_ids[:, start:end]

        chunks.append(chunk)

    return chunks


def gather_sequence(chunks: list[Tensor], overlap: int = 0) -> Tensor:
    """Inverse of :func:`partition_sequence` — concatenate chunks, removing overlaps.

    For each chunk after the first, the leading *overlap* tokens are stripped
    before concatenation so the result matches the original sequence.
    """
    if overlap == 0:
        return torch.cat(chunks, dim=1)

    parts: list[Tensor] = [chunks[0]]
    for chunk in chunks[1:]:
        parts.append(chunk[:, overlap:])
    return torch.cat(parts, dim=1)


def compute_chunk_attention(
    query_chunk: Tensor,
    key_chunks: list[Tensor],
    value_chunks: list[Tensor],
) -> Tensor:
    """Simulate ring attention for a single query chunk.

    Args:
        query_chunk: ``(B, T_q, d)``
        key_chunks: list of ``(B, T_k_i, d)`` tensors
        value_chunks: list of ``(B, T_v_i, d)`` tensors

    Returns:
        ``(B, T_q, d)`` — attended output.
    """
    K = torch.cat(key_chunks, dim=1)   # (B, sum(T_k), d)
    V = torch.cat(value_chunks, dim=1)  # (B, sum(T_v), d)

    d = query_chunk.shape[-1]
    scale = math.sqrt(d)

    scores = torch.matmul(query_chunk, K.transpose(-2, -1)) / scale  # (B, T_q, T_k)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)  # (B, T_q, d)
    return output


class RingAttentionSimulator:
    """Simulate a full ring attention pass across virtual ranks."""

    def __init__(self, config: SeqParallelConfig, d_model: int) -> None:
        self.config = config
        self.d_model = d_model

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        """Run ring attention simulation.

        Args:
            Q, K, V: ``(B, T, d)``

        Returns:
            ``(B, T, d)``
        """
        ws = self.config.world_size
        B, T, d = Q.shape
        chunk_size = T // ws

        # Partition Q, K, V
        def _split(x: Tensor) -> list[Tensor]:
            chunks: list[Tensor] = []
            for i in range(ws):
                start = i * chunk_size
                end = T if i == ws - 1 else start + chunk_size
                chunks.append(x[:, start:end])
            return chunks

        q_chunks = _split(Q)
        k_chunks = _split(K)
        v_chunks = _split(V)

        output_chunks: list[Tensor] = []
        for i, q_c in enumerate(q_chunks):
            if self.config.ring_attn:
                # ring: attend to neighbours [i-1, i, i+1] with wrapping
                indices = [
                    (i - 1) % ws,
                    i,
                    (i + 1) % ws,
                ]
                # deduplicate while preserving order
                seen: set[int] = set()
                unique_indices: list[int] = []
                for idx in indices:
                    if idx not in seen:
                        seen.add(idx)
                        unique_indices.append(idx)
                indices = unique_indices
            else:
                # causal: attend to chunks [0..i]
                indices = list(range(i + 1))

            kc = [k_chunks[j] for j in indices]
            vc = [v_chunks[j] for j in indices]
            out = compute_chunk_attention(q_c, kc, vc)
            output_chunks.append(out)

        return torch.cat(output_chunks, dim=1)


def compute_communication_cost(
    seq_len: int, config: SeqParallelConfig, d_model: int = 0
) -> dict[str, int]:
    """Estimate communication costs for sequence parallelism.

    Returns a dict with:
        - ``tokens_per_rank``: base tokens each rank processes
        - ``overlap_tokens_total``: total overlap tokens across all ranks
        - ``all_reduce_volume``: d_model * tokens communicated in all-reduce
    """
    tokens_per_rank = seq_len // config.world_size
    overlap_total = config.overlap_tokens * max(0, config.world_size - 1)
    all_reduce_volume = d_model * tokens_per_rank

    return {
        "tokens_per_rank": tokens_per_rank,
        "overlap_tokens_total": overlap_total,
        "all_reduce_volume": all_reduce_volume,
    }


class SequenceParallelTrainer:
    """Train a model by partitioning sequences across virtual ranks."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: SeqParallelConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config

    def train_step(self, input_ids: Tensor) -> dict[str, float]:
        """Simulate one distributed training step.

        The sequence is partitioned into chunks and each chunk is forwarded
        through the model sequentially (simulating distributed execution).
        Gradients are accumulated across chunks before a single optimizer step.
        """
        self.model.train()
        self.optimizer.zero_grad()

        chunks = partition_sequence(
            input_ids, self.config.world_size, self.config.overlap_tokens
        )

        total_loss = 0.0
        n_chunks = len(chunks)
        tokens_per_chunk = 0

        for chunk in chunks:
            tokens_per_chunk = chunk.shape[1]
            # Build labels: shift by one position for next-token prediction
            if chunk.shape[1] < 2:
                continue
            labels = chunk.clone()
            loss, _logits, _past = self.model(input_ids=chunk, labels=labels)
            if loss is not None:
                (loss / n_chunks).backward()
                total_loss += loss.item()

        self.optimizer.step()

        return {
            "loss": total_loss / max(n_chunks, 1),
            "n_chunks": n_chunks,
            "tokens_per_chunk": tokens_per_chunk,
        }
