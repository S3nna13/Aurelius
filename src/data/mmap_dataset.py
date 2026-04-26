from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MmapConfig:
    seq_len: int = 2048
    dtype: str = "uint16"
    stride: int | None = None  # None => seq_len (non-overlapping)


class MmapDataset(Dataset):
    """Memory-mapped numpy array dataset for efficient large-scale training."""

    def __init__(
        self,
        data_path: str | None = None,
        config: MmapConfig | None = None,
    ) -> None:
        self.config = config or MmapConfig()
        self.seq_len = self.config.seq_len
        self.stride = self.config.stride if self.config.stride is not None else self.seq_len

        if data_path is None or not __import__("os").path.exists(data_path):
            rng = np.random.default_rng(42)
            self._tokens = rng.integers(0, 50257, size=10_000, dtype=np.uint16)
            self._is_stub = True
        else:
            self._tokens = np.memmap(data_path, dtype=self.config.dtype, mode="r")
            self._is_stub = False

    def __len__(self) -> int:
        n = len(self._tokens)
        if n < self.seq_len + 1:
            return 0
        return (n - self.seq_len - 1) // self.stride + 1

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for dataset of length {len(self)}")
        start = idx * self.stride
        chunk = self._tokens[start : start + self.seq_len + 1]
        input_ids = torch.tensor(chunk[:-1].astype(np.int64), dtype=torch.long)
        labels = torch.tensor(chunk[1:].astype(np.int64), dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    def total_tokens(self) -> int:
        return int(len(self._tokens))

    def estimate_steps(self, batch_size: int, epochs: int = 1) -> int:
        return (len(self) * epochs) // batch_size
