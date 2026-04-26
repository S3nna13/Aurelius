"""Tests for src/training/curriculum.py — both TokenWeighter and epoch-based curriculum."""

from __future__ import annotations

import math

import pytest
import torch
from torch.optim import SGD

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.curriculum import (
    # Epoch-based curriculum components
    CurriculumConfig,
    CurriculumSampler,
    CurriculumTrainer,
    # Original TokenWeighter components
    TokenCurriculumConfig,
    TokenWeighter,
    WeightMode,
    difficulty_score,
    filter_by_difficulty,
    linear_curriculum,
    root_curriculum,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_cfg() -> AureliusConfig:
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


def _tiny_model() -> AureliusTransformer:
    return AureliusTransformer(_tiny_cfg())


def _token_tensor(length: int) -> torch.Tensor:
    """Return a 1-D token tensor of given length."""
    return torch.randint(0, 256, (length,))


# ---------------------------------------------------------------------------
# Original TokenWeighter tests (preserved)
# ---------------------------------------------------------------------------


def test_uniform_mode_equals_mean():
    weighter = TokenWeighter(TokenCurriculumConfig(mode=WeightMode.UNIFORM))
    loss = torch.ones(2, 8)
    result = weighter(loss)
    assert result == pytest.approx(1.0)


def test_position_weights_increase():
    weighter = TokenWeighter(TokenCurriculumConfig(mode=WeightMode.POSITION, position_exponent=1.0))
    loss = torch.zeros(1, 10)
    loss[0, -1] = 10.0
    result_pos = weighter(loss)
    result_uniform = TokenWeighter(TokenCurriculumConfig())(loss)
    assert result_pos.item() > result_uniform.item()


def test_frequency_weights_upweight_rare():
    weighter = TokenWeighter(TokenCurriculumConfig(mode=WeightMode.FREQUENCY))
    token_counts = torch.tensor([1.0, 1000.0])
    input_ids = torch.tensor([[0, 1]])
    loss2 = torch.tensor([[2.0, 1.0]])
    result2 = weighter(loss2, input_ids=input_ids, token_counts=token_counts)
    result_uniform2 = TokenWeighter()(torch.tensor([[2.0, 1.0]]))
    assert result2.item() > result_uniform2.item()


def test_custom_weights():
    weighter = TokenWeighter(TokenCurriculumConfig(mode=WeightMode.CUSTOM, normalize_weights=False))
    loss = torch.tensor([[1.0, 2.0, 3.0]])
    weights = torch.tensor([[0.0, 0.0, 1.0]])
    result = weighter(loss, custom_weights=weights)
    assert result.item() == pytest.approx(3.0)


def test_padding_mask_excludes_padded():
    weighter = TokenWeighter(TokenCurriculumConfig(mode=WeightMode.UNIFORM))
    loss = torch.tensor([[1.0, 1.0, 100.0]])
    mask = torch.tensor([[True, True, False]])
    result = weighter(loss, padding_mask=mask)
    assert result.item() == pytest.approx(1.0)


def test_returns_scalar():
    weighter = TokenWeighter()
    loss = torch.rand(4, 16)
    result = weighter(loss)
    assert result.ndim == 0


# ---------------------------------------------------------------------------
# CurriculumConfig defaults
# ---------------------------------------------------------------------------


def test_curriculum_config_defaults():
    cfg = CurriculumConfig()
    assert cfg.strategy == "linear"
    assert cfg.n_epochs == 10
    assert cfg.start_difficulty == pytest.approx(0.0)
    assert cfg.end_difficulty == pytest.approx(1.0)
    assert cfg.warmup_epochs == 2
    assert cfg.competence_threshold == pytest.approx(0.9)


def test_curriculum_config_custom():
    cfg = CurriculumConfig(
        strategy="root",
        n_epochs=20,
        start_difficulty=0.1,
        end_difficulty=0.8,
        warmup_epochs=5,
        competence_threshold=0.95,
    )
    assert cfg.strategy == "root"
    assert cfg.n_epochs == 20
    assert cfg.start_difficulty == pytest.approx(0.1)
    assert cfg.end_difficulty == pytest.approx(0.8)
    assert cfg.warmup_epochs == 5
    assert cfg.competence_threshold == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# difficulty_score
# ---------------------------------------------------------------------------


def test_difficulty_score_short_sequence():
    tokens = _token_tensor(64)  # 64 / 512 = 0.125
    score = difficulty_score(tokens, max_seq_len=512)
    assert score == pytest.approx(0.125)


def test_difficulty_score_long_sequence():
    tokens = _token_tensor(512)  # 512 / 512 = 1.0
    score = difficulty_score(tokens, max_seq_len=512)
    assert score == pytest.approx(1.0)


def test_difficulty_score_clips_at_one():
    tokens = _token_tensor(1024)  # exceeds max_seq_len
    score = difficulty_score(tokens, max_seq_len=512)
    assert score == pytest.approx(1.0)


def test_difficulty_score_range():
    for length in [1, 128, 256, 512]:
        score = difficulty_score(_token_tensor(length), max_seq_len=512)
        assert 0.0 <= score <= 1.0


def test_difficulty_score_2d_tensor():
    # 2-D tensor: shape (1, 128) -> seq len = last dim = 128
    tokens = torch.randint(0, 256, (1, 128))
    score = difficulty_score(tokens, max_seq_len=512)
    assert score == pytest.approx(128 / 512)


# ---------------------------------------------------------------------------
# linear_curriculum
# ---------------------------------------------------------------------------


def test_linear_curriculum_at_epoch_0():
    cfg = CurriculumConfig(
        strategy="linear", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0, warmup_epochs=0
    )
    assert linear_curriculum(0, cfg) == pytest.approx(0.0)


def test_linear_curriculum_at_half_epochs():
    cfg = CurriculumConfig(
        strategy="linear", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0, warmup_epochs=0
    )
    val = linear_curriculum(5, cfg)
    assert val == pytest.approx(0.5)


def test_linear_curriculum_at_n_epochs():
    cfg = CurriculumConfig(
        strategy="linear", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0, warmup_epochs=0
    )
    assert linear_curriculum(10, cfg) == pytest.approx(1.0)


def test_linear_curriculum_warmup_holds_start():
    cfg = CurriculumConfig(
        strategy="linear", n_epochs=10, start_difficulty=0.2, end_difficulty=1.0, warmup_epochs=3
    )
    for e in range(0, 4):
        assert linear_curriculum(e, cfg) == pytest.approx(0.2)


def test_linear_curriculum_monotone():
    cfg = CurriculumConfig(
        strategy="linear", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0, warmup_epochs=0
    )
    vals = [linear_curriculum(e, cfg) for e in range(11)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a


# ---------------------------------------------------------------------------
# root_curriculum
# ---------------------------------------------------------------------------


def test_root_curriculum_at_epoch_0():
    cfg = CurriculumConfig(strategy="root", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0)
    assert root_curriculum(0, cfg) == pytest.approx(0.0)


def test_root_curriculum_at_n_epochs():
    cfg = CurriculumConfig(strategy="root", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0)
    assert root_curriculum(10, cfg) == pytest.approx(1.0)


def test_root_curriculum_midpoint():
    cfg = CurriculumConfig(strategy="root", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0)
    val = root_curriculum(5, cfg)
    expected = math.sqrt(0.5)
    assert val == pytest.approx(expected, rel=1e-5)


def test_root_curriculum_faster_than_linear():
    """Root schedule should be ahead of linear at intermediate epochs."""
    cfg = CurriculumConfig(
        strategy="root", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0, warmup_epochs=0
    )
    for e in range(1, 10):
        root_val = root_curriculum(e, cfg)
        lin_val = linear_curriculum(e, cfg)
        assert root_val >= lin_val - 1e-9  # root is >= linear for 0..n_epochs


def test_root_curriculum_monotone():
    cfg = CurriculumConfig(strategy="root", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0)
    vals = [root_curriculum(e, cfg) for e in range(11)]
    for a, b in zip(vals, vals[1:]):
        assert b >= a


# ---------------------------------------------------------------------------
# filter_by_difficulty
# ---------------------------------------------------------------------------


def test_filter_by_difficulty_selects_easy():
    dataset = [_token_tensor(line) for line in [64, 128, 256, 512]]
    result = filter_by_difficulty(dataset, threshold=0.2, score_fn=lambda t: difficulty_score(t))
    # 64 -> 0.125 <= 0.2 qualifies; 128 -> 0.25 > 0.2 excluded
    assert len(result) == 1
    assert result[0].numel() == 64


def test_filter_by_difficulty_threshold_one_returns_all():
    dataset = [_token_tensor(line) for line in [64, 128, 256, 512]]
    result = filter_by_difficulty(dataset, threshold=1.0, score_fn=lambda t: difficulty_score(t))
    assert len(result) == 4


def test_filter_by_difficulty_threshold_zero_boundary():
    # With threshold=0.0, only a 0-length tensor (score=0.0) qualifies
    dataset = [torch.zeros(0, dtype=torch.long), _token_tensor(256)]
    result = filter_by_difficulty(dataset, threshold=0.0, score_fn=lambda t: difficulty_score(t))
    assert len(result) == 1
    assert result[0].numel() == 0


# ---------------------------------------------------------------------------
# CurriculumSampler
# ---------------------------------------------------------------------------


def test_curriculum_sampler_yields_increasing_data_with_epochs():
    dataset = [_token_tensor(line) for line in [64, 128, 256, 384, 512]]
    cfg = CurriculumConfig(
        strategy="linear", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0, warmup_epochs=0
    )
    sampler = CurriculumSampler(dataset, lambda t: difficulty_score(t), cfg)

    sampler.set_epoch(0)
    size_epoch0 = len(sampler)

    sampler.set_epoch(10)
    size_epoch10 = len(sampler)

    assert size_epoch10 >= size_epoch0


def test_curriculum_sampler_iter_yields_samples():
    dataset = [_token_tensor(64), _token_tensor(128)]
    cfg = CurriculumConfig(
        strategy="linear", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0, warmup_epochs=0
    )
    sampler = CurriculumSampler(dataset, lambda t: difficulty_score(t), cfg)
    sampler.set_epoch(10)
    items = list(sampler)
    assert len(items) == len(sampler)


def test_curriculum_sampler_epoch_0_restricted():
    # At epoch 0 with warmup=0, threshold=start_difficulty=0.1
    # 64-token -> 0.125 > 0.1, 512-token -> 1.0 > 0.1
    # Only a very short sequence qualifies
    dataset = [_token_tensor(line) for line in [1, 64, 512]]
    cfg = CurriculumConfig(
        strategy="linear", n_epochs=10, start_difficulty=0.1, end_difficulty=1.0, warmup_epochs=0
    )
    sampler = CurriculumSampler(dataset, lambda t: difficulty_score(t), cfg)
    sampler.set_epoch(0)
    # 1-token score = 1/512 ≈ 0.002 <= 0.1, qualifies; 64-token = 0.125 > 0.1
    assert len(sampler) == 1


def test_curriculum_sampler_len():
    dataset = [_token_tensor(64)] * 10
    cfg = CurriculumConfig()
    sampler = CurriculumSampler(dataset, lambda t: difficulty_score(t), cfg)
    sampler.set_epoch(cfg.n_epochs)
    assert len(sampler) == len(list(sampler))


# ---------------------------------------------------------------------------
# CurriculumTrainer
# ---------------------------------------------------------------------------


def test_curriculum_trainer_train_step_returns_expected_keys():
    model = _tiny_model()
    optimizer = SGD(model.parameters(), lr=1e-3)
    cfg = CurriculumConfig()
    trainer = CurriculumTrainer(model, optimizer, cfg)
    input_ids = torch.randint(0, 256, (1, 16))
    result = trainer.train_step(input_ids)
    assert "loss" in result
    assert "current_difficulty" in result


def test_curriculum_trainer_loss_is_scalar_float():
    model = _tiny_model()
    optimizer = SGD(model.parameters(), lr=1e-3)
    cfg = CurriculumConfig()
    trainer = CurriculumTrainer(model, optimizer, cfg)
    input_ids = torch.randint(0, 256, (1, 16))
    result = trainer.train_step(input_ids)
    assert isinstance(result["loss"], float)


def test_curriculum_trainer_advance_epoch_increases_difficulty():
    model = _tiny_model()
    optimizer = SGD(model.parameters(), lr=1e-3)
    cfg = CurriculumConfig(
        strategy="linear", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0, warmup_epochs=0
    )
    trainer = CurriculumTrainer(model, optimizer, cfg)
    d0 = trainer.current_difficulty
    trainer.advance_epoch()
    trainer.advance_epoch()
    trainer.advance_epoch()
    d3 = trainer.current_difficulty
    assert d3 >= d0


def test_curriculum_trainer_advance_epoch_increments_counter():
    model = _tiny_model()
    optimizer = SGD(model.parameters(), lr=1e-3)
    cfg = CurriculumConfig()
    trainer = CurriculumTrainer(model, optimizer, cfg)
    assert trainer._epoch == 0
    trainer.advance_epoch()
    assert trainer._epoch == 1
    trainer.advance_epoch()
    assert trainer._epoch == 2


def test_curriculum_trainer_difficulty_in_result():
    model = _tiny_model()
    optimizer = SGD(model.parameters(), lr=1e-3)
    cfg = CurriculumConfig(
        strategy="linear", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0, warmup_epochs=0
    )
    trainer = CurriculumTrainer(model, optimizer, cfg)
    input_ids = torch.randint(0, 256, (1, 16))
    r0 = trainer.train_step(input_ids)
    trainer.advance_epoch()
    trainer.advance_epoch()
    r2 = trainer.train_step(input_ids)
    assert r2["current_difficulty"] >= r0["current_difficulty"]


def test_curriculum_trainer_root_strategy():
    model = _tiny_model()
    optimizer = SGD(model.parameters(), lr=1e-3)
    cfg = CurriculumConfig(strategy="root", n_epochs=10, start_difficulty=0.0, end_difficulty=1.0)
    trainer = CurriculumTrainer(model, optimizer, cfg)
    trainer.advance_epoch()
    expected = root_curriculum(1, cfg)
    assert trainer.current_difficulty == pytest.approx(expected)


# ---------------------------------------------------------------------------
# competence_threshold interaction
# ---------------------------------------------------------------------------


def test_competence_threshold_default():
    cfg = CurriculumConfig()
    assert cfg.competence_threshold == pytest.approx(0.9)


def test_competence_threshold_custom():
    cfg = CurriculumConfig(competence_threshold=0.75)
    assert cfg.competence_threshold == pytest.approx(0.75)


def test_competence_threshold_filters_hard_samples():
    """With competence_threshold=0.5, samples with difficulty > 0.5 should be excluded."""
    dataset = [_token_tensor(line) for line in [64, 256, 512]]
    # Competence threshold used as filter threshold
    threshold = 0.5
    result = filter_by_difficulty(
        dataset, threshold=threshold, score_fn=lambda t: difficulty_score(t)
    )
    # 64 -> 0.125 ok, 256 -> 0.5 ok, 512 -> 1.0 excluded
    assert len(result) == 2
