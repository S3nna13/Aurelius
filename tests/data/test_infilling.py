"""Tests for src/data/infilling.py — span masking and FIM training."""

from __future__ import annotations

import random

import pytest
import torch

from src.data.infilling import (
    InfillingConfig,
    InfillingDataset,
    InfillingTrainer,
    apply_span_mask,
    fim_transform,
    sample_span_lengths,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
SEQ_LEN = 16
BATCH = 1

SMALL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=VOCAB_SIZE,
    max_seq_len=512,
)


def _rand_ids(length: int = SEQ_LEN) -> torch.Tensor:
    """Return a 1-D tensor of random token ids in [4, VOCAB_SIZE)."""
    return torch.randint(4, VOCAB_SIZE, (length,))


# ---------------------------------------------------------------------------
# InfillingConfig
# ---------------------------------------------------------------------------


def test_infilling_config_defaults():
    cfg = InfillingConfig()
    assert cfg.mask_prob == 0.15
    assert cfg.mean_span_length == 3.0
    assert cfg.fim_prob == 0.5
    assert cfg.prefix_token == 1
    assert cfg.suffix_token == 2
    assert cfg.middle_token == 3


# ---------------------------------------------------------------------------
# sample_span_lengths
# ---------------------------------------------------------------------------


def test_sample_span_lengths_returns_list_of_tuples():
    rng = random.Random(42)
    spans = sample_span_lengths(SEQ_LEN, 0.15, 3.0, rng=rng)
    assert isinstance(spans, list)
    for s in spans:
        assert isinstance(s, tuple)
        assert len(s) == 2


def test_sample_span_lengths_no_overlap():
    rng = random.Random(0)
    spans = sample_span_lengths(64, 0.30, 3.0, rng=rng)
    occupied: set[int] = set()
    for start, end in spans:
        candidate = set(range(start, end))
        assert candidate.isdisjoint(occupied), "spans must not overlap"
        occupied |= candidate


def test_sample_span_lengths_within_bounds():
    rng = random.Random(1)
    spans = sample_span_lengths(SEQ_LEN, 0.15, 3.0, rng=rng)
    for start, end in spans:
        assert 0 <= start < SEQ_LEN
        assert 0 < end <= SEQ_LEN
        assert start < end


# ---------------------------------------------------------------------------
# apply_span_mask
# ---------------------------------------------------------------------------


def test_apply_span_mask_correct_shapes():
    ids = _rand_ids()
    spans = [(2, 5), (10, 13)]
    masked_ids, labels = apply_span_mask(ids, spans)
    assert masked_ids.shape == ids.shape
    assert labels.shape == ids.shape


def test_apply_span_mask_labels_minus100_at_unmasked():
    ids = _rand_ids()
    spans = [(3, 6)]
    _, labels = apply_span_mask(ids, spans)
    for i in range(len(ids)):
        if not (3 <= i < 6):
            assert labels[i].item() == -100, f"position {i} should be -100"


def test_apply_span_mask_masked_positions_have_original_token():
    ids = _rand_ids()
    spans = [(1, 4), (8, 10)]
    masked_ids, labels = apply_span_mask(ids, spans)
    for start, end in spans:
        for i in range(start, end):
            assert labels[i].item() == ids[i].item()
            assert masked_ids[i].item() == 0


# ---------------------------------------------------------------------------
# fim_transform
# ---------------------------------------------------------------------------


def test_fim_transform_contains_special_tokens():
    ids = _rand_ids(SEQ_LEN)
    out = fim_transform(ids, prefix_token=1, suffix_token=2, middle_token=3, rng=random.Random(7))
    tokens = out.tolist()
    assert 1 in tokens, "prefix_token must be in output"
    assert 2 in tokens, "suffix_token must be in output"
    assert 3 in tokens, "middle_token must be in output"


def test_fim_transform_output_length():
    ids = _rand_ids(SEQ_LEN)
    out = fim_transform(ids, prefix_token=1, suffix_token=2, middle_token=3, rng=random.Random(9))
    assert out.shape[0] == SEQ_LEN + 3, f"expected length {SEQ_LEN + 3}, got {out.shape[0]}"


# ---------------------------------------------------------------------------
# InfillingDataset
# ---------------------------------------------------------------------------


def test_infilling_dataset_len():
    samples = [_rand_ids() for _ in range(10)]
    ds = InfillingDataset(samples)
    assert len(ds) == 10


def test_infilling_dataset_getitem_returns_tuple():
    samples = [_rand_ids() for _ in range(5)]
    ds = InfillingDataset(samples, InfillingConfig(fim_prob=0.0))
    item = ds[0]
    assert isinstance(item, tuple)
    assert len(item) == 2
    input_ids, labels = item
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(labels, torch.Tensor)


def test_infilling_dataset_labels_contain_minus100():
    """At least some label positions should be -100 (non-masked or FIM)."""
    samples = [_rand_ids(SEQ_LEN) for _ in range(20)]
    cfg = InfillingConfig(mask_prob=0.15, fim_prob=0.0)
    ds = InfillingDataset(samples, cfg)
    found_minus100 = False
    for i in range(len(ds)):
        _, labels = ds[i]
        if (labels == -100).any():
            found_minus100 = True
            break
    assert found_minus100, "Expected some -100 labels in dataset"


# ---------------------------------------------------------------------------
# InfillingTrainer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_model_and_opt():
    model = AureliusTransformer(SMALL_CFG)
    model.eval()
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    return model, opt


def test_infilling_trainer_train_step_keys(small_model_and_opt):
    model, opt = small_model_and_opt
    trainer = InfillingTrainer(model, InfillingConfig(), optimizer=opt)
    input_ids = torch.randint(4, VOCAB_SIZE, (BATCH, SEQ_LEN))
    result = trainer.train_step(input_ids)
    assert "loss" in result, "train_step must return 'loss'"
    assert "n_masked" in result, "train_step must return 'n_masked'"


def test_infilling_trainer_train_step_n_masked_positive(small_model_and_opt):
    model, opt = small_model_and_opt
    trainer = InfillingTrainer(model, InfillingConfig(mask_prob=0.50), optimizer=opt)
    input_ids = torch.randint(4, VOCAB_SIZE, (BATCH, SEQ_LEN))
    result = trainer.train_step(input_ids)
    assert result["n_masked"] > 0, "n_masked must be > 0"
