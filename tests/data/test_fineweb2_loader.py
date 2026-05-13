"""Tests for FineWeb2 data loader."""

from __future__ import annotations


def test_fineweb2_dataset_import():
    from src.data.fineweb2_loader import FineWeb2Dataset

    assert FineWeb2Dataset is not None


def test_fineweb2_dataset_init_signature():
    from unittest.mock import MagicMock

    from src.data.fineweb2_loader import FineWeb2Dataset

    tokenizer = MagicMock()
    ds = FineWeb2Dataset(tokenizer=tokenizer, seq_len=128, split="train")
    assert ds.seq_len == 128
    assert ds.split == "train"
    assert ds.synthetic_ratio == 0.3


def test_fineweb2_iter_does_not_raise():
    from unittest.mock import MagicMock

    from src.data.fineweb2_loader import FineWeb2Dataset

    tokenizer = MagicMock()
    tokenizer.encode.return_value = list(range(128))
    ds = FineWeb2Dataset(tokenizer=tokenizer, seq_len=8, use_hq=False, synthetic_ratio=0.0)
    # Just verify __iter__ is callable
    it = iter(ds)
    assert it is not None
