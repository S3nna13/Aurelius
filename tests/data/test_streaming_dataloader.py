from __future__ import annotations

import torch

from src.data.mmap_dataset import MmapConfig, MmapDataset
from src.data.streaming_dataloader import StreamConfig, StreamingDataloader

# ---------------------------------------------------------------------------
# Minimal in-memory dataset for isolated tests
# ---------------------------------------------------------------------------


class _TinyDataset:
    """Fixed-size dataset returning sequential tensors for deterministic tests."""

    def __init__(self, n: int = 100, seq_len: int = 16):
        self.n = n
        self.seq_len = seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        t = torch.full((self.seq_len,), idx, dtype=torch.long)
        return {"input_ids": t, "labels": t}


# ---------------------------------------------------------------------------
# StreamConfig tests
# ---------------------------------------------------------------------------


class TestStreamConfig:
    def test_defaults(self):
        cfg = StreamConfig()
        assert cfg.batch_size == 8
        assert cfg.shuffle_buffer == 1000
        assert cfg.prefetch == 2
        assert cfg.drop_last is True

    def test_custom(self):
        cfg = StreamConfig(batch_size=16, shuffle_buffer=500, drop_last=False)
        assert cfg.batch_size == 16
        assert cfg.shuffle_buffer == 500


# ---------------------------------------------------------------------------
# StreamingDataloader tests
# ---------------------------------------------------------------------------


class TestStreamingDataloader:
    def _make(self, n=100, seq_len=16, batch_size=8, **kw):
        ds = _TinyDataset(n=n, seq_len=seq_len)
        cfg = StreamConfig(batch_size=batch_size, **kw)
        return StreamingDataloader(ds, cfg)

    def test_len_estimate(self):
        dl = self._make(n=100, batch_size=8)
        assert len(dl) == 12  # 100 // 8

    def test_yields_dicts(self):
        dl = self._make(n=50, batch_size=4)
        batch = next(iter(dl))
        assert isinstance(batch, dict)
        assert "input_ids" in batch and "labels" in batch

    def test_batch_shape(self):
        dl = self._make(n=50, seq_len=16, batch_size=4)
        batch = next(iter(dl))
        assert batch["input_ids"].shape == (4, 16)
        assert batch["labels"].shape == (4, 16)

    def test_batch_dtype(self):
        dl = self._make(n=50, batch_size=4)
        batch = next(iter(dl))
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long

    def test_drop_last_true_no_partial(self):
        dl = self._make(n=10, batch_size=4, drop_last=True, shuffle_buffer=10)
        batches = list(dl)
        for b in batches:
            assert b["input_ids"].shape[0] == 4

    def test_drop_last_false_allows_partial(self):
        dl = self._make(n=10, batch_size=4, drop_last=False, shuffle_buffer=10)
        batches = list(dl)
        sizes = [b["input_ids"].shape[0] for b in batches]
        assert any(s < 4 for s in sizes) or sum(sizes) == 10

    def test_all_samples_covered_drop_last_false(self):
        n = 17
        dl = self._make(n=n, batch_size=4, drop_last=False, shuffle_buffer=20)
        total = sum(b["input_ids"].shape[0] for b in dl)
        assert total == n

    def test_reservoir_sample_length(self):
        dl = self._make()
        indices = list(range(200))
        result = dl._reservoir_sample(indices, 50)
        assert len(result) == 50

    def test_reservoir_sample_subset(self):
        dl = self._make()
        indices = list(range(200))
        result = dl._reservoir_sample(indices, 50)
        assert all(v in indices for v in result)

    def test_collate_stacks_tensors(self):
        dl = self._make(seq_len=8)
        samples = [
            {
                "input_ids": torch.zeros(8, dtype=torch.long),
                "labels": torch.ones(8, dtype=torch.long),
            }
            for _ in range(4)
        ]
        out = dl.collate(samples)
        assert out["input_ids"].shape == (4, 8)
        assert out["labels"].shape == (4, 8)

    def test_integrates_with_mmap_dataset(self):
        ds = MmapDataset(config=MmapConfig(seq_len=32))
        cfg = StreamConfig(batch_size=4, shuffle_buffer=20)
        dl = StreamingDataloader(ds, cfg)
        batch = next(iter(dl))
        assert batch["input_ids"].shape == (4, 32)

    def test_iteration_exhausts_without_error(self):
        dl = self._make(n=30, batch_size=8, shuffle_buffer=30)
        batches = list(dl)
        assert len(batches) >= 1
