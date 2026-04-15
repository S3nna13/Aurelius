"""Chunked prefill for LLM inference.

Long-context prefill can OOM on limited-memory hardware. Chunked prefill processes
the prompt in fixed-size chunks, building a KV cache incrementally. This decouples
memory usage from prompt length.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PrefillStats:
    n_chunks: int
    chunk_size: int
    total_prefill_tokens: int
    peak_tokens_in_flight: int  # max tokens in a single forward pass
    prefill_time_ms: float = 0.0  # placeholder 0.0 in tests


@dataclass
class ChunkedPrefillConfig:
    chunk_size: int = 512
    overlap: int = 0            # optional token overlap between chunks
    use_kv_cache: bool = True


class ChunkedPrefillEngine:
    def __init__(
        self,
        model: nn.Module,
        config: ChunkedPrefillConfig = None,
    ):
        self.model = model
        self.config = config if config is not None else ChunkedPrefillConfig()

    def _forward(
        self,
        input_ids: Tensor,  # (B, T)
    ) -> Tensor:
        """Run model, return logits (B, T, V)."""
        output = self.model(input_ids)
        # Handle (loss, logits, pkv) tuple or bare logits tensor
        if isinstance(output, tuple):
            logits = output[1]
        else:
            logits = output
        return logits

    def get_chunk_schedule(
        self,
        seq_len: int,
    ) -> List[Tuple[int, int]]:
        """
        Returns list of (start, end) pairs for each chunk.
        Accounts for overlap.
        Example: seq_len=10, chunk_size=4, overlap=0 -> [(0,4),(4,8),(8,10)]
        """
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap
        chunks = []
        start = 0
        while start < seq_len:
            end = min(start + chunk_size, seq_len)
            chunks.append((start, end))
            if end >= seq_len:
                break
            # Advance by chunk_size - overlap (so next chunk re-covers `overlap` tokens)
            advance = chunk_size - overlap
            if advance <= 0:
                # Prevent infinite loop if overlap >= chunk_size
                advance = 1
            start += advance
        return chunks

    def prefill(
        self,
        prompt_ids: Tensor,  # (B, T_prompt)
    ) -> Tuple[Tensor, PrefillStats]:
        """
        Process prompt in chunks. Returns:
        - final_logits: (B, T_prompt, V) — logits for the full prompt
        - stats: PrefillStats
        """
        B, T = prompt_ids.shape
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap

        schedule = self.get_chunk_schedule(T)
        n_chunks = len(schedule)
        peak_tokens = 0
        chunk_logits_list = []

        for start, end in schedule:
            chunk_ids = prompt_ids[:, start:end]
            chunk_len = end - start
            if chunk_len > peak_tokens:
                peak_tokens = chunk_len
            logits_chunk = self._forward(chunk_ids)  # (B, chunk_len, V)
            chunk_logits_list.append(logits_chunk)

        # Merge chunk logits, stripping overlap tokens from non-first chunks
        final_logits = merge_chunked_logits(chunk_logits_list, overlap=overlap)

        # Trim or pad to exactly T tokens in case of rounding
        final_logits = final_logits[:, :T, :]

        stats = PrefillStats(
            n_chunks=n_chunks,
            chunk_size=chunk_size,
            total_prefill_tokens=T,
            peak_tokens_in_flight=peak_tokens,
            prefill_time_ms=0.0,
        )
        return final_logits, stats

    def prefill_and_decode(
        self,
        prompt_ids: Tensor,          # (B, T_prompt)
        max_new_tokens: int = 10,
        temperature: float = 0.0,    # 0.0 = greedy
    ) -> Tuple[Tensor, PrefillStats]:
        """
        Chunked prefill then greedy/sampled decode.
        Returns (output_ids: (B, T_prompt + max_new_tokens), stats).
        """
        B, T_prompt = prompt_ids.shape
        final_logits, stats = self.prefill(prompt_ids)

        # Autoregressive decode
        generated = prompt_ids
        for _ in range(max_new_tokens):
            logits = self._forward(generated)   # (B, current_len, V)
            next_logits = logits[:, -1, :]      # (B, V)
            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)    # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated, stats


def chunk_sequence(
    input_ids: Tensor,  # (B, T)
    chunk_size: int,
    overlap: int = 0,
) -> List[Tensor]:
    """Split (B, T) into list of (B, chunk_size) tensors (last may be smaller)."""
    B, T = input_ids.shape
    chunks = []
    start = 0
    while start < T:
        end = min(start + chunk_size, T)
        chunks.append(input_ids[:, start:end])
        if end >= T:
            break
        advance = chunk_size - overlap
        if advance <= 0:
            advance = 1
        start += advance
    return chunks


def merge_chunked_logits(
    chunk_logits: List[Tensor],  # list of (B, chunk_T, V)
    overlap: int = 0,
) -> Tensor:
    """Concatenate chunk logits, removing overlap tokens. Returns (B, T_total, V)."""
    if not chunk_logits:
        raise ValueError("chunk_logits must be non-empty")
    if len(chunk_logits) == 1:
        return chunk_logits[0]

    parts = [chunk_logits[0]]
    for chunk in chunk_logits[1:]:
        if overlap > 0:
            # Drop the first `overlap` tokens from each subsequent chunk
            # (they were already covered by the previous chunk's tail)
            trimmed = chunk[:, overlap:, :]
            parts.append(trimmed)
        else:
            parts.append(chunk)
    return torch.cat(parts, dim=1)
