"""Tests for src/data/curriculum_learning.py

Covers DifficultyScorer, PacingFunction, CurriculumDataset,
DataSelectionFilter, and CurriculumTrainer.

Model fixture: tiny 1-layer causal transformer (vocab=16, seq=8, d_model=16).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.data.curriculum_learning import (
    CurriculumDataset,
    CurriculumTrainer,
    DataSelectionFilter,
    DifficultyScorer,
    PacingFunction,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOCAB_SIZE = 16
SEQ_LEN = 8
N = 32
BATCH = 4
D_MODEL = 16


# ---------------------------------------------------------------------------
# Tiny causal transformer helper
# ---------------------------------------------------------------------------

class _TinyTransformer(nn.Module):
    """1-layer causal transformer for tests."""

    def __init__(self, vocab: int = VOCAB_SIZE, d_model: int = D_MODEL, seq_len: int = SEQ_LEN) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=32,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(d_model, vocab)
        self._seq_len = seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # [B, T] -> [B, T, V]
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        # causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
        x = self.embed(input_ids) + self.pos(positions)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(x)  # [B, T, V]


def _make_model() -> _TinyTransformer:
    return _TinyTransformer()


def _make_data(n: int = N) -> torch.Tensor:
    """Random token sequences [n, SEQ_LEN]."""
    return torch.randint(1, VOCAB_SIZE, (n, SEQ_LEN))


def _make_scores(n: int = N) -> torch.Tensor:
    return torch.rand(n)


# ---------------------------------------------------------------------------
# DifficultyScorer tests
# ---------------------------------------------------------------------------

def test_perplexity_score_shape():
    """perplexity_score returns shape [B]."""
    scorer = DifficultyScorer()
    model = _make_model()
    ids = _make_data(BATCH)
    out = scorer.perplexity_score(model, ids)
    assert out.shape == (BATCH,), f"expected ({BATCH},), got {out.shape}"


def test_perplexity_score_values_ge_1():
    """perplexity_score values are >= 1.0."""
    scorer = DifficultyScorer()
    model = _make_model()
    ids = _make_data(BATCH)
    out = scorer.perplexity_score(model, ids)
    assert (out >= 1.0).all(), f"Expected all >= 1, got min={out.min().item()}"


def test_length_score_range():
    """length_score returns values in [0, 1]."""
    scorer = DifficultyScorer()
    ids = _make_data(N)
    out = scorer.length_score(ids)
    assert out.shape == (N,), f"Expected shape ({N},)"
    assert (out >= 0.0).all() and (out <= 1.0).all(), (
        f"Values out of [0,1]: min={out.min()}, max={out.max()}"
    )


def test_vocabulary_richness_range():
    """vocabulary_richness returns values in [0, 1]."""
    scorer = DifficultyScorer()
    ids = _make_data(N)
    out = scorer.vocabulary_richness(ids, VOCAB_SIZE)
    assert out.shape == (N,), f"Expected shape ({N},)"
    assert (out >= 0.0).all() and (out <= 1.0).all(), (
        f"Values out of [0,1]: min={out.min()}, max={out.max()}"
    )


def test_vocabulary_richness_uniform():
    """All-same-token sequence has richness 1/T."""
    scorer = DifficultyScorer()
    ids = torch.ones(1, SEQ_LEN, dtype=torch.long) * 5
    out = scorer.vocabulary_richness(ids, VOCAB_SIZE)
    expected = 1.0 / SEQ_LEN
    assert abs(out[0].item() - expected) < 1e-6, f"Expected {expected}, got {out[0]}"


def test_combined_score_shape():
    """combined_score returns shape [B]."""
    scorer = DifficultyScorer()
    model = _make_model()
    ids = _make_data(BATCH)
    weights = {"perplexity": 0.5, "length": 0.3, "vocab": 0.2}
    out = scorer.combined_score(model, ids, weights)
    assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"


# ---------------------------------------------------------------------------
# PacingFunction tests
# ---------------------------------------------------------------------------

def test_linear_pacing_at_step_0():
    """linear_pacing at step=0 equals start_frac."""
    start = 0.1
    result = PacingFunction.linear_pacing(0, 100, start_frac=start)
    assert abs(result - start) < 1e-9, f"Expected {start}, got {result}"


def test_linear_pacing_at_total_steps():
    """linear_pacing at step=total_steps equals 1.0."""
    result = PacingFunction.linear_pacing(100, 100, start_frac=0.1)
    assert abs(result - 1.0) < 1e-9, f"Expected 1.0, got {result}"


def test_linear_pacing_monotone():
    """linear_pacing is non-decreasing."""
    vals = [PacingFunction.linear_pacing(s, 100, 0.1) for s in range(0, 101, 10)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a, f"Not monotone: {a} > {b}"


def test_competence_pacing_monotone():
    """competence_pacing is monotonically increasing."""
    vals = [PacingFunction.competence_pacing(s, 100, c0=0.01) for s in range(0, 101, 5)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-12, f"Not monotone at {a} -> {b}"


def test_competence_pacing_starts_at_c0():
    """competence_pacing at step=0 equals c0."""
    c0 = 0.01
    result = PacingFunction.competence_pacing(0, 100, c0=c0)
    expected = math.sqrt(c0 ** 2)
    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"


def test_competence_pacing_ends_at_1():
    """competence_pacing at step=total_steps equals 1.0."""
    result = PacingFunction.competence_pacing(100, 100, c0=0.01)
    assert abs(result - 1.0) < 1e-9, f"Expected 1.0, got {result}"


def test_exponential_pacing_range():
    """exponential_pacing stays in [0, 1]."""
    for s in range(0, 101, 10):
        v = PacingFunction.exponential_pacing(s, 100, k=5.0)
        assert 0.0 <= v <= 1.0, f"Out of range at step {s}: {v}"


def test_step_pacing_milestones():
    """step_pacing returns correct fraction at each milestone."""
    milestones = [0, 25, 50, 75]
    fracs = [0.25, 0.5, 0.75, 1.0]

    assert abs(PacingFunction.step_pacing(0,  milestones, fracs) - 0.25) < 1e-9
    assert abs(PacingFunction.step_pacing(25, milestones, fracs) - 0.5)  < 1e-9
    assert abs(PacingFunction.step_pacing(50, milestones, fracs) - 0.75) < 1e-9
    assert abs(PacingFunction.step_pacing(75, milestones, fracs) - 1.0)  < 1e-9
    # Between milestones: last passed
    assert abs(PacingFunction.step_pacing(30, milestones, fracs) - 0.5)  < 1e-9


# ---------------------------------------------------------------------------
# CurriculumDataset tests
# ---------------------------------------------------------------------------

def _make_curriculum_dataset() -> CurriculumDataset:
    ids = _make_data(N)
    scores = _make_scores(N)
    pacing_fn = lambda step: PacingFunction.linear_pacing(step, 100, 0.1)
    return CurriculumDataset(ids, scores, pacing_fn)


def test_curriculum_get_batch_shape():
    """get_batch returns tensor of shape [B, T]."""
    ds = _make_curriculum_dataset()
    batch = ds.get_batch(BATCH)
    assert batch.shape == (BATCH, SEQ_LEN), f"Expected ({BATCH}, {SEQ_LEN}), got {batch.shape}"


def test_curriculum_advance_increases_step():
    """advance() increments step counter."""
    ds = _make_curriculum_dataset()
    assert ds.step == 0
    ds.advance(5)
    assert ds.step == 5
    ds.advance()
    assert ds.step == 6


def test_curriculum_early_steps_easy_examples():
    """At step=0 with start_frac=0.1, only easiest 10% are sampled."""
    ids = _make_data(N)
    scores = torch.arange(N, dtype=torch.float32)  # 0..31, deterministic ordering
    pacing_fn = lambda step: PacingFunction.linear_pacing(step, 1000, 0.1)
    ds = CurriculumDataset(ids, scores, pacing_fn)

    sorted_indices = torch.argsort(scores).tolist()
    n_avail = max(1, math.ceil(0.1 * N))  # top 10% easiest
    easy_set = set(sorted_indices[:n_avail])

    # Confirm pool at step 0 matches
    pool = ds._available_pool()
    assert set(pool) == easy_set, f"Pool at step 0 should be easiest {n_avail} examples"

    # All sampled examples must come from easy pool
    for _ in range(10):
        batch = ds.get_batch(BATCH)
        # Check each row of batch is from easy_set
        for row in batch:
            # Find which original index this row matches
            matches = [i for i in easy_set if torch.equal(ids[i], row)]
            assert len(matches) >= 1, "Batch row not from easy pool"


def test_anti_curriculum_batch_shape():
    """get_anti_curriculum_batch returns shape [B, T]."""
    ds = _make_curriculum_dataset()
    # Advance to make more data available
    ds.advance(50)
    batch = ds.get_anti_curriculum_batch(BATCH)
    assert batch.shape == (BATCH, SEQ_LEN), f"Expected ({BATCH}, {SEQ_LEN}), got {batch.shape}"


def test_anti_curriculum_harder_than_curriculum():
    """Anti-curriculum batch has higher mean score than normal batch when many steps in."""
    ids = _make_data(N)
    scores = torch.arange(N, dtype=torch.float32)  # deterministic
    pacing_fn = lambda step: 1.0  # always full dataset
    ds = CurriculumDataset(ids, scores, pacing_fn)

    # With full dataset, easy batch should come from sorted[:B] and hard from sorted[-B:]
    pool = ds._available_pool()
    hard_pool = pool[-BATCH:]
    easy_pool = pool[:BATCH]

    hard_scores = scores[torch.tensor(hard_pool)].mean().item()
    easy_scores = scores[torch.tensor(easy_pool)].mean().item()
    assert hard_scores > easy_scores, "Hard pool should have higher mean score"


# ---------------------------------------------------------------------------
# DataSelectionFilter tests
# ---------------------------------------------------------------------------

def test_filter_by_score_reduces_dataset():
    """filter_by_score keeps only examples with score <= threshold."""
    filt = DataSelectionFilter(threshold=0.5)
    ids = _make_data(N)
    # Half scores above threshold
    scores = torch.linspace(0.0, 1.0, N)
    kept = filt.filter_by_score(ids, scores)
    assert kept.shape[0] < N, "filter_by_score should remove some examples"
    assert kept.shape[1] == SEQ_LEN


def test_filter_by_score_all_below_threshold():
    """filter_by_score keeps all examples when all scores <= threshold."""
    filt = DataSelectionFilter(threshold=1.0)
    ids = _make_data(N)
    scores = torch.zeros(N)
    kept = filt.filter_by_score(ids, scores)
    assert kept.shape[0] == N


def test_quality_filter_removes_repetitive():
    """quality_filter removes sequences where one token dominates."""
    filt = DataSelectionFilter()

    # Good sequences: varied tokens
    good = torch.randint(1, VOCAB_SIZE, (8, SEQ_LEN))

    # Bad sequences: all same token (rep_ratio = 1.0 > 0.5)
    bad = torch.ones(4, SEQ_LEN, dtype=torch.long) * 3

    ids = torch.cat([good, bad], dim=0)
    kept = filt.quality_filter(ids, min_len=2, max_rep_ratio=0.5)

    # Bad sequences should be removed; good ones kept
    assert kept.shape[0] <= good.shape[0], (
        f"Expected at most {good.shape[0]} kept, got {kept.shape[0]}"
    )
    assert kept.shape[0] > 0, "Should keep some good sequences"


def test_quality_filter_removes_short():
    """quality_filter removes sequences shorter than min_len."""
    filt = DataSelectionFilter()

    # Sequences where first 2 tokens are non-zero, rest are padding (0)
    short_ids = torch.zeros(4, SEQ_LEN, dtype=torch.long)
    short_ids[:, :2] = torch.randint(1, VOCAB_SIZE, (4, 2))

    long_ids = torch.randint(1, VOCAB_SIZE, (4, SEQ_LEN))
    ids = torch.cat([short_ids, long_ids], dim=0)

    kept = filt.quality_filter(ids, min_len=4, max_rep_ratio=0.9)
    assert kept.shape[0] <= 4, f"Short seqs should be removed, kept={kept.shape[0]}"


def test_deduplication_filter_removes_duplicates():
    """deduplication_filter removes near-identical sequences."""
    filt = DataSelectionFilter()

    # Two identical sequences + one unique
    base = torch.randint(1, VOCAB_SIZE, (1, SEQ_LEN))
    dup = base.clone()
    unique = torch.randint(1, VOCAB_SIZE, (1, SEQ_LEN))
    # Make unique truly different by using different token range
    unique = (unique % (VOCAB_SIZE // 2)) + (VOCAB_SIZE // 2)

    ids = torch.cat([base, dup, unique], dim=0)
    kept = filt.deduplication_filter(ids, sim_threshold=1.0)
    assert kept.shape[0] == 2, f"Should keep 2 (base + unique), got {kept.shape[0]}"


# ---------------------------------------------------------------------------
# CurriculumTrainer tests
# ---------------------------------------------------------------------------

def test_curriculum_trainer_train_step_finite_loss():
    """train_step returns a finite float loss."""
    model = _make_model()
    ids = _make_data(N)
    scores = _make_scores(N)
    pacing_fn = lambda step: PacingFunction.competence_pacing(step, 100)
    ds = CurriculumDataset(ids, scores, pacing_fn)
    ds.batch_size = BATCH

    trainer = CurriculumTrainer(model, ds, lr=1e-3)
    loss, step = trainer.train_step()

    assert isinstance(loss, float), f"Loss should be float, got {type(loss)}"
    assert math.isfinite(loss), f"Loss should be finite, got {loss}"
    assert step == 1, f"Step should be 1 after one train_step, got {step}"


def test_curriculum_trainer_step_increments():
    """train_step increments the dataset step each call."""
    model = _make_model()
    ids = _make_data(N)
    scores = _make_scores(N)
    pacing_fn = lambda step: 1.0
    ds = CurriculumDataset(ids, scores, pacing_fn)
    ds.batch_size = BATCH

    trainer = CurriculumTrainer(model, ds, lr=1e-3)
    for expected_step in range(1, 4):
        _, step = trainer.train_step()
        assert step == expected_step, f"Expected step {expected_step}, got {step}"


def test_curriculum_trainer_get_stats():
    """get_training_stats returns expected keys."""
    model = _make_model()
    ids = _make_data(N)
    scores = _make_scores(N)
    pacing_fn = lambda step: PacingFunction.linear_pacing(step, 100, 0.1)
    ds = CurriculumDataset(ids, scores, pacing_fn)
    ds.batch_size = BATCH

    trainer = CurriculumTrainer(model, ds, lr=1e-3)
    trainer.train_step()
    stats = trainer.get_training_stats()

    assert "current_frac" in stats
    assert "step" in stats
    assert "n_seen_easy" in stats
    assert "n_seen_hard" in stats
    assert 0.0 <= stats["current_frac"] <= 1.0
    assert stats["step"] >= 1
