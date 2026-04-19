"""Instruction-tuning sample packer with boundary-aware attention masks.

Packs SFT samples into fixed-length sequences using first-fit decreasing
(FFD) bin packing. Each packed sequence carries:

  * input_ids        -- [1, L]       token ids, padded with pad_token_id
  * attention_mask   -- [1, L, L]    block-diagonal (respect_sample_boundaries=True)
                                      or upper-triangular causal (False); pad rows/cols masked.
  * loss_mask        -- [1, L]       True only on response (assistant) tokens.
  * segment_ids      -- [1, L]       integer id per sample within the bin;
                                      pad tokens use segment id 0 with attention_mask=False.

Pure PyTorch. No foreign imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, List

import torch


@dataclass
class InstructionSample:
    """A single instruction-tuning sample.

    prompt_token_ids:   tokens for the user/system prompt (loss masked OUT).
    response_token_ids: tokens for the assistant response (loss ON).
    """

    prompt_token_ids: List[int]
    response_token_ids: List[int]

    def __len__(self) -> int:
        return len(self.prompt_token_ids) + len(self.response_token_ids)


@dataclass
class PackedBatch:
    """A single packed training sequence (batch dim = 1)."""

    input_ids: torch.Tensor      # [1, L] long
    attention_mask: torch.Tensor # [1, L, L] bool
    loss_mask: torch.Tensor      # [1, L] bool
    segment_ids: torch.Tensor    # [1, L] long


class InstructionDatasetPacker:
    """First-fit-decreasing packer for instruction samples.

    Args:
        max_seq_len: fixed length of each packed sequence. Must be > 0.
        pad_token_id: id used to pad the remainder of under-full bins.
        respect_sample_boundaries: if True, build block-diagonal 2D attention
            masks so one sample cannot attend to another in the same bin.
            If False, build a standard upper-triangular causal mask over the
            entire packed sequence (samples may attend across boundaries).
    """

    def __init__(
        self,
        max_seq_len: int = 4096,
        pad_token_id: int = 0,
        respect_sample_boundaries: bool = True,
    ) -> None:
        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be a positive int, got {max_seq_len!r}"
            )
        if not isinstance(pad_token_id, int):
            raise TypeError(
                f"pad_token_id must be int, got {type(pad_token_id).__name__}"
            )
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.respect_sample_boundaries = bool(respect_sample_boundaries)

    # ----------------------------------------------------------------- bin packing

    def _ffd(self, samples: List[InstructionSample]) -> List[List[InstructionSample]]:
        """First-fit decreasing. Raises ValueError on oversized samples."""
        for i, s in enumerate(samples):
            n = len(s)
            if n == 0:
                raise ValueError(f"sample {i} is empty (0 tokens)")
            if n > self.max_seq_len:
                raise ValueError(
                    f"sample {i} has length {n} > max_seq_len {self.max_seq_len}"
                )

        # Stable sort by length descending; ties preserve input order.
        order = sorted(
            range(len(samples)), key=lambda i: (-len(samples[i]), i)
        )
        bins: List[List[InstructionSample]] = []
        bin_lens: List[int] = []
        for idx in order:
            s = samples[idx]
            n = len(s)
            placed = False
            for b, used in enumerate(bin_lens):
                if used + n <= self.max_seq_len:
                    bins[b].append(s)
                    bin_lens[b] = used + n
                    placed = True
                    break
            if not placed:
                bins.append([s])
                bin_lens.append(n)
        return bins

    # ----------------------------------------------------------------- materialization

    def _materialize(self, bin_samples: List[InstructionSample]) -> PackedBatch:
        L = self.max_seq_len
        input_ids = torch.full((L,), self.pad_token_id, dtype=torch.long)
        loss_mask = torch.zeros(L, dtype=torch.bool)
        segment_ids = torch.zeros(L, dtype=torch.long)
        # real-token mask -- used for both block-diagonal and causal cases.
        real = torch.zeros(L, dtype=torch.bool)

        offset = 0
        for seg_idx, sample in enumerate(bin_samples, start=1):
            p = sample.prompt_token_ids
            r = sample.response_token_ids
            np_ = len(p)
            nr = len(r)
            n = np_ + nr
            if p:
                input_ids[offset : offset + np_] = torch.tensor(p, dtype=torch.long)
            if r:
                input_ids[offset + np_ : offset + n] = torch.tensor(r, dtype=torch.long)
                loss_mask[offset + np_ : offset + n] = True
            segment_ids[offset : offset + n] = seg_idx
            real[offset : offset + n] = True
            offset += n

        # Build 2D attention mask.
        if self.respect_sample_boundaries:
            # Same-segment AND causal AND both-real.
            seg_eq = segment_ids.unsqueeze(0) == segment_ids.unsqueeze(1)  # [L,L]
            pos = torch.arange(L)
            causal = pos.unsqueeze(1) >= pos.unsqueeze(0)  # [L,L] row>=col
            both_real = real.unsqueeze(0) & real.unsqueeze(1)
            attn = seg_eq & causal & both_real
            # Diagonal for pad positions stays False -- pads attend to nothing.
        else:
            pos = torch.arange(L)
            causal = pos.unsqueeze(1) >= pos.unsqueeze(0)
            both_real = real.unsqueeze(0) & real.unsqueeze(1)
            attn = causal & both_real

        return PackedBatch(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attn.unsqueeze(0),
            loss_mask=loss_mask.unsqueeze(0),
            segment_ids=segment_ids.unsqueeze(0),
        )

    # ----------------------------------------------------------------- public api

    def pack(self, samples: List[InstructionSample]) -> List[PackedBatch]:
        if not samples:
            return []
        bins = self._ffd(list(samples))
        return [self._materialize(b) for b in bins]

    def pack_iter(
        self, iterable: Iterable[InstructionSample], batch_size: int
    ) -> Iterator[PackedBatch]:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive int, got {batch_size!r}"
            )
        buf: List[InstructionSample] = []
        for s in iterable:
            buf.append(s)
            if len(buf) >= batch_size:
                for pb in self.pack(buf):
                    yield pb
                buf = []
        if buf:
            for pb in self.pack(buf):
                yield pb


__all__ = ["InstructionSample", "PackedBatch", "InstructionDatasetPacker"]
