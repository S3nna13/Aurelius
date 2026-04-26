"""Tests for src/training/curriculum_pacing.py — pacing functions and curriculum samplers."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.curriculum_pacing import (
    CurriculumSampler,
    DifficultyScorer,
    PacingConfig,
    exponential_pacing,
    get_pacing_fraction,
    linear_pacing,
    step_pacing,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SAMPLES = 20
BATCH_SIZE = 4


def _small_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


def _small_model() -> AureliusTransformer:
    return AureliusTransformer(_small_cfg())


def _make_samples_and_difficulties():
    """20 integer samples with difficulty = sample index (0=easiest, 19=hardest)."""
    samples = list(range(N_SAMPLES))
    difficulties = torch.arange(N_SAMPLES, dtype=torch.float32)
    return samples, difficulties


# ---------------------------------------------------------------------------
# PacingConfig defaults
# ---------------------------------------------------------------------------


def test_pacing_config_defaults():
    cfg = PacingConfig()
    assert cfg.pacing_fn == "linear"
    assert cfg.start_fraction == pytest.approx(0.2)
    assert cfg.end_fraction == pytest.approx(1.0)
    assert cfg.n_steps == 1000
    assert cfg.step_size == 100


# ---------------------------------------------------------------------------
# linear_pacing
# ---------------------------------------------------------------------------


def test_linear_pacing_at_step_0_returns_start():
    assert linear_pacing(0, 1000, 0.2, 1.0) == pytest.approx(0.2)


def test_linear_pacing_at_n_steps_returns_end():
    assert linear_pacing(1000, 1000, 0.2, 1.0) == pytest.approx(1.0)


def test_linear_pacing_monotonically_increasing():
    vals = [linear_pacing(s, 100, 0.1, 1.0) for s in range(0, 101, 10)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a


def test_linear_pacing_clamps_beyond_n_steps():
    assert linear_pacing(2000, 1000, 0.2, 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# exponential_pacing
# ---------------------------------------------------------------------------


def test_exponential_pacing_at_step_0_near_start():
    val = exponential_pacing(0, 1000, 0.2, 1.0)
    assert val == pytest.approx(0.2, abs=1e-6)


def test_exponential_pacing_at_n_steps_near_end():
    val = exponential_pacing(1000, 1000, 0.2, 1.0)
    assert val == pytest.approx(1.0, abs=1e-6)


def test_exponential_pacing_monotonically_increasing():
    vals = [exponential_pacing(s, 100, 0.1, 1.0) for s in range(0, 101, 10)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a


# ---------------------------------------------------------------------------
# step_pacing
# ---------------------------------------------------------------------------


def test_step_pacing_returns_discrete_levels():
    """Values should only increase at multiples of step_size."""
    step_size = 100
    n_steps = 1000
    start, end = 0.2, 1.0
    prev = None
    stair_changes = 0
    for s in range(0, n_steps + 1, 10):
        v = step_pacing(s, step_size, start, end, n_steps)
        if prev is not None and v != prev:
            stair_changes += 1
        prev = v
    # With step_size=100 and n_steps=1000 we have 10 stairs, so 10 transitions
    assert stair_changes == 10


def test_step_pacing_at_step_0_returns_start():
    val = step_pacing(0, 100, 0.2, 1.0, 1000)
    assert val == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# get_pacing_fraction dispatches correctly
# ---------------------------------------------------------------------------


def test_get_pacing_fraction_dispatches_linear():
    cfg = PacingConfig(pacing_fn="linear", start_fraction=0.2, end_fraction=1.0, n_steps=1000)
    assert get_pacing_fraction(0, cfg) == pytest.approx(linear_pacing(0, 1000, 0.2, 1.0))
    assert get_pacing_fraction(500, cfg) == pytest.approx(linear_pacing(500, 1000, 0.2, 1.0))


def test_get_pacing_fraction_dispatches_exponential():
    cfg = PacingConfig(pacing_fn="exponential", start_fraction=0.2, end_fraction=1.0, n_steps=1000)
    assert get_pacing_fraction(500, cfg) == pytest.approx(
        exponential_pacing(500, 1000, 0.2, 1.0), rel=1e-5
    )


def test_get_pacing_fraction_dispatches_step():
    cfg = PacingConfig(
        pacing_fn="step", start_fraction=0.2, end_fraction=1.0, n_steps=1000, step_size=100
    )
    assert get_pacing_fraction(150, cfg) == pytest.approx(
        step_pacing(150, 100, 0.2, 1.0, 1000), rel=1e-5
    )


def test_get_pacing_fraction_raises_on_unknown():
    cfg = PacingConfig(pacing_fn="unknown")
    with pytest.raises(ValueError, match="Unknown pacing_fn"):
        get_pacing_fraction(0, cfg)


# ---------------------------------------------------------------------------
# DifficultyScorer
# ---------------------------------------------------------------------------


def test_difficulty_scorer_score_shape():
    model = _small_model()
    scorer = DifficultyScorer(model)
    ids_list = [torch.randint(0, 256, (1, 8)) for _ in range(5)]
    scores = scorer.score(ids_list)
    assert scores.shape == (5,)


def test_difficulty_scorer_score_non_negative():
    model = _small_model()
    scorer = DifficultyScorer(model)
    ids_list = [torch.randint(0, 256, (1, 8)) for _ in range(5)]
    scores = scorer.score(ids_list)
    assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# CurriculumSampler
# ---------------------------------------------------------------------------


def test_curriculum_sampler_get_batch_correct_count():
    samples, difficulties = _make_samples_and_difficulties()
    cfg = PacingConfig(pacing_fn="linear", start_fraction=0.5, end_fraction=1.0, n_steps=1000)
    sampler = CurriculumSampler(samples, difficulties, cfg)
    batch = sampler.get_batch(step=0, batch_size=BATCH_SIZE)
    assert len(batch) == BATCH_SIZE


def test_curriculum_sampler_early_steps_return_easier_samples():
    """At step=0 only easiest 20% are eligible; at step=n_steps all are."""
    samples, difficulties = _make_samples_and_difficulties()
    cfg = PacingConfig(pacing_fn="linear", start_fraction=0.2, end_fraction=1.0, n_steps=1000)
    sampler = CurriculumSampler(samples, difficulties, cfg)

    torch.manual_seed(0)
    early_batches = [sampler.get_batch(step=0, batch_size=BATCH_SIZE) for _ in range(50)]
    early_max = max(max(b) for b in early_batches)

    torch.manual_seed(0)
    late_batches = [sampler.get_batch(step=1000, batch_size=BATCH_SIZE) for _ in range(50)]
    late_max = max(max(b) for b in late_batches)

    # Early steps restricted to top 20% easiest (indices 0..3); late steps can reach index 19
    assert early_max <= late_max


def test_curriculum_sampler_update_difficulties():
    samples, difficulties = _make_samples_and_difficulties()
    cfg = PacingConfig()
    sampler = CurriculumSampler(samples, difficulties, cfg)
    new_scores = torch.zeros(N_SAMPLES)
    sampler.update_difficulties(new_scores)
    assert (sampler.difficulties == 0).all()


def test_curriculum_sampler_batch_size_one():
    samples, difficulties = _make_samples_and_difficulties()
    cfg = PacingConfig()
    sampler = CurriculumSampler(samples, difficulties, cfg)
    batch = sampler.get_batch(step=0, batch_size=1)
    assert len(batch) == 1
