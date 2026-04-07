from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class TokenizedShardDataset(Dataset):
    """Memory-mapped dataset over pre-tokenized .npy shards.

    Each shard is a 1-D numpy array of uint16 token ids, stored as a .npy file.
    This dataset presents fixed-length windows of tokens as (input_ids, labels)
    pairs suitable for causal language model training.

    Args:
        shard_paths: List of paths to .npy token shard files.
        seq_len: Context length for each training example.
        stride: Step between consecutive windows. Defaults to seq_len (no overlap).
    """

    def __init__(
        self,
        shard_paths: list[str | Path],
        seq_len: int,
        stride: int | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        # Load memory-mapped arrays for each shard.
        # np.load with mmap_mode='r' correctly handles the .npy header and
        # returns a view into just the data region, unlike np.memmap which
        # would include header bytes and produce an incorrect length.
        self._memmaps: list[np.ndarray] = []
        for path in shard_paths:
            mm = np.load(path, mmap_mode="r")
            self._memmaps.append(mm)

        # Compute per-shard window counts and cumulative offsets for indexing
        self._windows_per_shard: list[int] = []
        for mm in self._memmaps:
            n = len(mm)
            needed = seq_len + 1  # seq_len input tokens + 1 label token
            if n < needed:
                count = 0
            else:
                count = (n - needed) // self.stride + 1
            self._windows_per_shard.append(count)

        # Cumulative counts for O(log n) shard lookup via searchsorted
        self._cumulative = np.cumsum([0] + self._windows_per_shard)

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, labels) where labels = input_ids shifted left by 1.

        input_ids:  tokens[start : start + seq_len]
        labels:     tokens[start + 1 : start + seq_len + 1]

        Both are int64 tensors of shape (seq_len,).
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        # Find which shard this global index falls in
        shard_idx = int(np.searchsorted(self._cumulative[1:], idx, side="right"))
        local_idx = idx - int(self._cumulative[shard_idx])

        start = local_idx * self.stride
        end = start + self.seq_len + 1  # +1 for the label shift

        chunk = self._memmaps[shard_idx][start:end].astype(np.int64)
        input_ids = torch.from_numpy(chunk[:-1])
        labels = torch.from_numpy(chunk[1:])

        return input_ids, labels
