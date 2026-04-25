from __future__ import annotations

import os
import struct
import tempfile

import numpy as np
import pytest
import torch

from src.data.mmap_dataset import MmapConfig, MmapDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tokens(path: str, n: int, dtype: str = "uint16") -> np.ndarray:
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 50257, size=n, dtype=dtype)
    tokens.tofile(path)
    return tokens


# ---------------------------------------------------------------------------
# MmapConfig tests
# ---------------------------------------------------------------------------

class TestMmapConfig:
    def test_defaults(self):
        cfg = MmapConfig()
        assert cfg.seq_len == 2048
        assert cfg.dtype == "uint16"
        assert cfg.stride is None

    def test_custom_values(self):
        cfg = MmapConfig(seq_len=512, dtype="int32", stride=256)
        assert cfg.seq_len == 512
        assert cfg.dtype == "int32"
        assert cfg.stride == 256


# ---------------------------------------------------------------------------
# Stub (no file) behaviour
# ---------------------------------------------------------------------------

class TestMmapDatasetStub:
    def setup_method(self):
        self.cfg = MmapConfig(seq_len=64)
        self.ds = MmapDataset(config=self.cfg)

    def test_stub_created_when_no_path(self):
        assert self.ds._is_stub is True

    def test_stub_created_when_path_missing(self):
        ds = MmapDataset(data_path="/nonexistent/file.bin", config=MmapConfig(seq_len=64))
        assert ds._is_stub is True

    def test_total_tokens_stub(self):
        assert self.ds.total_tokens() == 10_000

    def test_len_nonzero(self):
        assert len(self.ds) > 0

    def test_getitem_returns_dict(self):
        sample = self.ds[0]
        assert isinstance(sample, dict)
        assert "input_ids" in sample
        assert "labels" in sample

    def test_getitem_shapes(self):
        sample = self.ds[0]
        assert sample["input_ids"].shape == (64,)
        assert sample["labels"].shape == (64,)

    def test_getitem_dtypes(self):
        sample = self.ds[0]
        assert sample["input_ids"].dtype == torch.long
        assert sample["labels"].dtype == torch.long

    def test_labels_offset_by_one(self):
        ds = MmapDataset(config=MmapConfig(seq_len=64))
        s0 = ds[0]
        s1_input = ds._tokens[1:65].astype(np.int64)
        assert torch.all(s0["labels"] == torch.tensor(s1_input))

    def test_index_out_of_bounds_raises(self):
        with pytest.raises(IndexError):
            self.ds[len(self.ds)]

    def test_negative_index_raises(self):
        with pytest.raises(IndexError):
            self.ds[-1]

    def test_estimate_steps(self):
        steps = self.ds.estimate_steps(batch_size=4, epochs=2)
        assert steps == (len(self.ds) * 2) // 4


# ---------------------------------------------------------------------------
# Real file (memmap) behaviour
# ---------------------------------------------------------------------------

class TestMmapDatasetFile:
    def test_len_calculation_no_stride(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            n = 5000
            _write_tokens(path, n)
            cfg = MmapConfig(seq_len=128)
            ds = MmapDataset(data_path=path, config=cfg)
            expected = (n - 128 - 1) // 128 + 1
            assert len(ds) == expected
        finally:
            os.unlink(path)

    def test_len_calculation_with_stride(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            n = 5000
            _write_tokens(path, n)
            cfg = MmapConfig(seq_len=128, stride=64)
            ds = MmapDataset(data_path=path, config=cfg)
            expected = (n - 128 - 1) // 64 + 1
            assert len(ds) == expected
        finally:
            os.unlink(path)

    def test_getitem_values_match_file(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            tokens = _write_tokens(path, 1000)
            cfg = MmapConfig(seq_len=32)
            ds = MmapDataset(data_path=path, config=cfg)
            sample = ds[2]
            start = 2 * 32
            expected_input = torch.tensor(tokens[start : start + 32].astype(np.int64))
            expected_labels = torch.tensor(tokens[start + 1 : start + 33].astype(np.int64))
            assert torch.all(sample["input_ids"] == expected_input)
            assert torch.all(sample["labels"] == expected_labels)
        finally:
            os.unlink(path)

    def test_is_not_stub_for_existing_file(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            _write_tokens(path, 1000)
            ds = MmapDataset(data_path=path, config=MmapConfig(seq_len=32))
            assert ds._is_stub is False
        finally:
            os.unlink(path)
