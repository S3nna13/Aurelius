"""Tests for DifficultyScorer."""
import math
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.data.difficulty_scorer import (
    DifficultyScore,
    DifficultyScoreConfig,
    DifficultyScorer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )


@pytest.fixture(scope="module")
def small_model(small_cfg):
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def scorer_cfg():
    return DifficultyScoreConfig(
        batch_size=4,
        device="cpu",
        percentile_easy=33.0,
        percentile_hard=66.0,
    )


@pytest.fixture(scope="module")
def scorer(small_model, scorer_cfg):
    return DifficultyScorer(small_model, scorer_cfg)


@pytest.fixture(scope="module")
def tiny_dataloader():
    """8 examples, seq_len=16, vocab_size=256."""
    torch.manual_seed(42)
    n, seq = 8, 16
    input_ids = torch.randint(0, 256, (n, seq))
    labels = input_ids.clone()
    dataset = TensorDataset(input_ids, labels)

    def collate(batch):
        ids = torch.stack([b[0] for b in batch])
        lbs = torch.stack([b[1] for b in batch])
        return {"input_ids": ids, "labels": lbs}

    return DataLoader(dataset, batch_size=4, collate_fn=collate)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_difficulty_score_dataclass():
    """DifficultyScore has index, perplexity, loss, n_tokens."""
    s = DifficultyScore(index=0, perplexity=10.0, loss=2.3, n_tokens=15)
    assert s.index == 0
    assert s.perplexity == 10.0
    assert s.loss == 2.3
    assert s.n_tokens == 15


def test_score_batch_returns_list(scorer):
    """score_batch returns a list of DifficultyScore."""
    torch.manual_seed(0)
    ids = torch.randint(0, 256, (2, 16))
    lbs = ids.clone()
    result = scorer.score_batch(ids, lbs)
    assert isinstance(result, list)
    assert all(isinstance(s, DifficultyScore) for s in result)


def test_score_batch_length(scorer):
    """One score per example in the batch."""
    torch.manual_seed(1)
    B = 3
    ids = torch.randint(0, 256, (B, 16))
    lbs = ids.clone()
    result = scorer.score_batch(ids, lbs)
    assert len(result) == B


def test_score_batch_perplexity_positive(scorer):
    """All perplexity values are > 0."""
    torch.manual_seed(2)
    ids = torch.randint(0, 256, (4, 16))
    lbs = ids.clone()
    result = scorer.score_batch(ids, lbs)
    for s in result:
        assert s.perplexity > 0.0


def test_score_batch_ppl_from_loss(scorer):
    """Perplexity is approximately exp(loss) within tolerance."""
    torch.manual_seed(3)
    ids = torch.randint(0, 256, (4, 16))
    lbs = ids.clone()
    result = scorer.score_batch(ids, lbs)
    for s in result:
        expected_ppl = math.exp(min(s.loss, 20.0))
        assert abs(s.perplexity - expected_ppl) < 1e-4, (
            f"ppl={s.perplexity:.6f} but exp(loss)={expected_ppl:.6f}"
        )


def test_score_dataset_length(scorer, tiny_dataloader):
    """Scores for all examples in the dataloader."""
    scores = scorer.score_dataset(tiny_dataloader)
    assert len(scores) == 8


def test_score_dataset_sorted_by_index(scorer, tiny_dataloader):
    """Indices are 0, 1, 2, ... in order."""
    scores = scorer.score_dataset(tiny_dataloader)
    indices = [s.index for s in scores]
    assert indices == list(range(len(scores)))


def test_partition_three_groups(scorer, tiny_dataloader):
    """easy + medium + hard covers all scores (no overlap, no missing)."""
    scores = scorer.score_dataset(tiny_dataloader)
    easy, medium, hard = scorer.partition(scores)
    total = len(easy) + len(medium) + len(hard)
    assert total == len(scores)
    # No duplicates
    all_indices = [s.index for s in easy + medium + hard]
    assert len(all_indices) == len(set(all_indices))


def test_sampling_weights_sum_to_one(scorer, tiny_dataloader):
    """Sum of sampling weights is approximately 1.0."""
    scores = scorer.score_dataset(tiny_dataloader)
    weights = scorer.get_sampling_weights(scores, temperature=1.0)
    assert abs(weights.sum().item() - 1.0) < 1e-5


def test_sampling_weights_hard_higher(scorer):
    """Hard examples (high ppl) have higher weight than easy ones (low ppl)."""
    easy_score = DifficultyScore(index=0, perplexity=2.0, loss=0.7, n_tokens=15)
    hard_score = DifficultyScore(index=1, perplexity=200.0, loss=5.3, n_tokens=15)
    scores = [easy_score, hard_score]
    weights = scorer.get_sampling_weights(scores, temperature=1.0)
    assert weights[1].item() > weights[0].item()


def test_save_load_roundtrip(scorer, tiny_dataloader, tmp_path):
    """Save then load returns same scores (index, perplexity, loss, n_tokens)."""
    scores = scorer.score_dataset(tiny_dataloader)
    save_path = tmp_path / "scores.npy"
    scorer.save_scores(scores, save_path)
    loaded = scorer.load_scores(save_path)

    assert len(loaded) == len(scores)
    for orig, ld in zip(scores, loaded):
        assert orig.index == ld.index
        assert abs(orig.perplexity - ld.perplexity) < 1e-9
        assert abs(orig.loss - ld.loss) < 1e-9
        assert orig.n_tokens == ld.n_tokens
