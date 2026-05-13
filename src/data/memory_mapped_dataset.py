"""Memory-mapped dataset utilities for massive corpora.

Provides :class:`MemoryMappedDataset` which uses :mod:`numpy.memmap` to
stream extremely large token shards (dozens of GB+) without loading the
entire array into RAM.  The API mimics :class:`torch.utils.data.Dataset`
and integrates directly with the :mod:`torch.utils.data.DataLoader`
pipeline already used throughout ``src.data``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset


class MemoryMappedDataset(Dataset):
    """Memory-mapped dataset of token IDs stored as a raw ``.npy`` file.

    Parameters
    ----------
    path: str | pathlib.Path
        Path to the ``.npy`` file containing a 1D ``uint16`` or ``int64``
        array of token IDs.  The file is opened in ``r`` mode and stays
        mapped for the lifetime of the object.
    seq_len: int
        Sequence length to return per ``__getitem__``.  Tokens are taken
        as contiguous slices; the dataset size is ``max(0, n_tokens - seq_len)``.
    stride: int, optional
        Stride between consecutive sequences (default ``seq_len`` → no overlap).
        Any positive value is accepted; ``stride < seq_len`` yields overlapping
        windows.
    dtype: torch.dtype, optional
        Output tensor dtype (default ``torch.long``).

    Notes
    -----
    - The underlying :class:`numpy.memmap` is read-only; any writes will
      raise.  Create new shards with :func:`numpy.save` or the existing
      data-pipeline scripts.
    - No caching is performed beyond the OS page cache; this class is
      designed to be used with a :class:`torch.utils.data.DataLoader`
      with ``num_workers > 0`` to hide I/O latency.
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        seq_len: int,
        stride: int | None = None,
        dtype: torch.dtype = torch.long,
    ) -> None:
        self.path = pathlib.Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Memory-mapped shard not found: {self.path}")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        self.dtype = dtype

        # Open memmap (read-only)
        self._mm = np.load(self.path, mmap_mode="r")
        if self._mm.ndim != 1:
            raise ValueError(f"Expected 1D array in {self.path}, got shape {self._mm.shape}")
        self._n = len(self._mm)

    def __len__(self) -> int:
        return max(0, (self._n - self.seq_len) // self.stride)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self)}")
        start = idx * self.stride
        end = start + self.seq_len
        slice_ = self._mm[start:end]
        return torch.from_numpy(slice_).to(self.dtype)

    def get_histogram(self, bins: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a histogram of token IDs across the entire shard.

        Useful for sanity-checking data distribution without loading it all.

        Parameters
        ----------
        bins: int
            Number of histogram bins.

        Returns
        -------
        counts: torch.Tensor
            Count per bin, shape ``(bins,)``.
        edges: torch.Tensor
            Bin edges (length ``bins + 1``).
        """
        data = self._mm[:]  # triggers page-fault reads; okay for sampling a
        # stats subset? For full scan may be slow
        # Use numpy histogram then convert
        counts, edges = np.histogram(data, bins=bins)
        return torch.from_numpy(counts), torch.from_numpy(edges)

    def close(self) -> None:
        """Close the underlying memmap and release OS resources."""
        if hasattr(self, "_mm"):
            del self._mm

    def __del__(self) -> None:
        self.close()


def make_mmap_dataset(
    shard_dir: str | pathlib.Path,
    *,
    seq_len: int,
    stride: int | None = None,
    dtype: torch.dtype = torch.long,
    pattern: str = "*.npy",
) -> torch.utils.data.ConcatDataset:
    """Construct a :class:`ConcatDataset` over all memory-mapped shards in a directory.

    Convenience helper used by training scripts to replace in-memory
    ``TensorDataset`` with a streaming variant.

    Parameters
    ----------
    shard_dir: str | pathlib.Path
        Directory containing ``.npy`` token shards.
    seq_len: int
        Sequence length for each sample.
    stride: int, optional
        Stride between sequences within each shard.
    dtype: torch.dtype, optional
        Output tensor type.
    pattern: str, optional
        Glob pattern for shard files (default ``"*.npy"``).

    Returns
    -------
    ConcatDataset
        Concatenated dataset over all matching shards.  Iteration order is
        deterministic: sorted filenames, shards concatenated sequentially.
    """
    root = pathlib.Path(shard_dir)
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shards matching {pattern} in {shard_dir}")
    datasets = [MemoryMappedDataset(p, seq_len=seq_len, stride=stride, dtype=dtype) for p in files]
    return torch.utils.data.ConcatDataset(datasets)

