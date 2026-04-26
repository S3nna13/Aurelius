"""Tests for standalone perplexity evaluator."""

import math

import pytest
import torch

from src.eval.perplexity import compute_perplexity, perplexity_on_dataset
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    return AureliusTransformer(cfg)


def test_perplexity_finite(small_model):
    """compute_perplexity must return finite, positive PPL."""
    seqs = [list(range(16)), list(range(10, 26))]
    result = compute_perplexity(small_model, seqs)
    assert math.isfinite(result.perplexity)
    assert result.perplexity > 1.0
    assert result.total_tokens == 30  # 2 seqs of 16 tokens, each contributes 15 targets


def test_perplexity_repeated_sequence(small_model):
    """Same sequence repeated should give same PPL."""
    seq = list(range(16))
    r1 = compute_perplexity(small_model, [seq])
    r2 = compute_perplexity(small_model, [seq])
    assert abs(r1.perplexity - r2.perplexity) < 1e-3


def test_perplexity_long_sequence(small_model):
    """Long sequence (> max_seq_len) should be handled via sliding window."""
    # max_seq_len=32, create sequence of 50 tokens
    seq = list(range(50))
    result = compute_perplexity(small_model, [seq], stride=16)
    assert math.isfinite(result.perplexity)
    assert result.total_tokens > 0


def test_perplexity_on_dataset(small_model):
    """perplexity_on_dataset must process DataLoader batches."""
    from torch.utils.data import DataLoader, TensorDataset

    ids = torch.randint(0, 256, (4, 16))
    ds = TensorDataset(ids)

    def collate(batch):
        return {"input_ids": torch.stack([b[0] for b in batch])}

    loader = DataLoader(ds, batch_size=2, collate_fn=collate)
    result = perplexity_on_dataset(small_model, loader)
    assert math.isfinite(result.perplexity)
    assert result.n_sequences == 4
