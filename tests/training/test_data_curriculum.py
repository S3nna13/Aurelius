"""Tests for src/training/data_curriculum.py — multi-source adaptive curriculum."""

import torch

from src.training.data_curriculum import (
    AdaptiveCurriculumSampler,
    CurriculumConfig,
    DataSource,
    compute_difficulty_mask,
    compute_loss_adaptive_weights,
    diversity_regularization,
    normalize_weights,
)

# ---------------------------------------------------------------------------
# DataSource
# ---------------------------------------------------------------------------


def test_datasource_defaults():
    src = DataSource(name="wiki")
    assert src.name == "wiki"
    assert src.weight == 1.0
    assert src.domain == "general"
    assert src.difficulty == 0.5


def test_datasource_custom_fields():
    src = DataSource(name="code", weight=2.0, domain="programming", difficulty=0.8)
    assert src.weight == 2.0
    assert src.domain == "programming"
    assert src.difficulty == 0.8


# ---------------------------------------------------------------------------
# CurriculumConfig
# ---------------------------------------------------------------------------


def test_curriculum_config_fields():
    cfg = CurriculumConfig(n_sources=4)
    assert cfg.n_sources == 4
    assert cfg.update_interval == 100
    assert cfg.loss_ema_alpha == 0.1
    assert cfg.diversity_weight == 0.1
    assert cfg.difficulty_warmup_steps == 1000
    assert cfg.min_source_weight == 0.05


def test_curriculum_config_custom():
    cfg = CurriculumConfig(
        n_sources=3,
        update_interval=50,
        loss_ema_alpha=0.2,
        diversity_weight=0.3,
        difficulty_warmup_steps=500,
        min_source_weight=0.01,
    )
    assert cfg.update_interval == 50
    assert cfg.loss_ema_alpha == 0.2
    assert cfg.min_source_weight == 0.01


# ---------------------------------------------------------------------------
# normalize_weights
# ---------------------------------------------------------------------------


def test_normalize_weights_sums_to_one():
    w = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = normalize_weights(w)
    assert abs(result.sum().item() - 1.0) < 1e-6


def test_normalize_weights_min_floor():
    # One weight is below the floor; after clipping and normalizing it should not be negligibly small  # noqa: E501
    w = torch.tensor([0.001, 1.0, 1.0])
    result = normalize_weights(w, min_weight=0.05)
    # The floor is applied before normalization, so no raw weight < 0.05 exists
    # After normalization each is >= min_weight / (min_weight + sum_of_rest) roughly,
    # but importantly the smallest source is not effectively zero
    assert result.min().item() > 1e-4
    assert abs(result.sum().item() - 1.0) < 1e-6


def test_normalize_weights_shape_preserved():
    w = torch.tensor([0.5, 0.5])
    result = normalize_weights(w)
    assert result.shape == w.shape


# ---------------------------------------------------------------------------
# compute_loss_adaptive_weights
# ---------------------------------------------------------------------------


def test_compute_loss_adaptive_weights_shape():
    losses = torch.tensor([0.5, 1.0, 2.0])
    weights = torch.tensor([0.33, 0.33, 0.34])
    result = compute_loss_adaptive_weights(losses, weights, alpha=0.1)
    assert result.shape == losses.shape


def test_compute_loss_adaptive_weights_sums_to_one():
    losses = torch.tensor([0.5, 1.0, 2.0])
    weights = torch.tensor([0.33, 0.33, 0.34])
    result = compute_loss_adaptive_weights(losses, weights, alpha=0.1)
    assert abs(result.sum().item() - 1.0) < 1e-5


def test_compute_loss_adaptive_weights_higher_loss_higher_weight():
    # Source 2 has much higher loss — after update it should get more weight than source 0
    losses = torch.tensor([0.1, 0.5, 5.0])
    weights = torch.tensor([0.33, 0.33, 0.34])
    result = compute_loss_adaptive_weights(losses, weights, alpha=1.0)  # full update
    assert result[2].item() > result[0].item()


# ---------------------------------------------------------------------------
# compute_difficulty_mask
# ---------------------------------------------------------------------------


def test_compute_difficulty_mask_step_zero_only_easy():
    sources = [
        DataSource("easy", difficulty=0.0),
        DataSource("medium", difficulty=0.5),
        DataSource("hard", difficulty=0.9),
    ]
    mask = compute_difficulty_mask(sources, step=0, warmup_steps=1000)
    # progress=0.0, threshold=0.1
    # easy (0.0 <= 0.1): included; medium (0.5 > 0.1): excluded; hard (0.9 > 0.1): excluded
    assert mask[0].item() == 1.0
    assert mask[1].item() == 0.0
    assert mask[2].item() == 0.0


def test_compute_difficulty_mask_full_warmup_all_included():
    sources = [
        DataSource("easy", difficulty=0.0),
        DataSource("medium", difficulty=0.5),
        DataSource("hard", difficulty=0.95),
    ]
    mask = compute_difficulty_mask(sources, step=1000, warmup_steps=1000)
    # progress=1.0, threshold=1.1 — all difficulties <= 1.1
    assert mask.sum().item() == 3.0


def test_compute_difficulty_mask_shape():
    sources = [DataSource(f"s{i}", difficulty=i / 4) for i in range(4)]
    mask = compute_difficulty_mask(sources, step=500, warmup_steps=1000)
    assert mask.shape == (4,)


# ---------------------------------------------------------------------------
# diversity_regularization
# ---------------------------------------------------------------------------


def test_diversity_regularization_pushes_toward_uniform():
    # Two domains: "a" has 1 source with weight 0.9, "b" has 1 source with weight 0.1
    sources = [
        DataSource("s0", domain="a"),
        DataSource("s1", domain="b"),
    ]
    weights = torch.tensor([0.9, 0.1])
    result = diversity_regularization(weights, sources, diversity_weight=1.0)
    # Full regularization → target is [0.5, 0.5]
    assert abs(result[0].item() - 0.5) < 1e-5
    assert abs(result[1].item() - 0.5) < 1e-5


def test_diversity_regularization_sums_to_one():
    sources = [
        DataSource("s0", domain="nlp"),
        DataSource("s1", domain="nlp"),
        DataSource("s2", domain="code"),
    ]
    weights = torch.tensor([0.5, 0.3, 0.2])
    result = diversity_regularization(weights, sources, diversity_weight=0.2)
    assert abs(result.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# AdaptiveCurriculumSampler
# ---------------------------------------------------------------------------


def _make_sampler():
    sources = [
        DataSource("wiki", weight=1.0, domain="text", difficulty=0.1),
        DataSource("books", weight=1.0, domain="text", difficulty=0.4),
        DataSource("code", weight=1.0, domain="code", difficulty=0.7),
    ]
    cfg = CurriculumConfig(n_sources=3, difficulty_warmup_steps=1000)
    return AdaptiveCurriculumSampler(sources, cfg), sources


def test_sampler_update_weights_runs():
    sampler, sources = _make_sampler()
    losses = {s.name: 1.0 for s in sources}
    sampler.update_weights(losses, step=200)  # should not raise


def test_sampler_update_weights_changes_weights():
    sampler, sources = _make_sampler()
    before = dict(sampler.get_weights())
    # Give "code" a very high loss — it should get more weight
    losses = {"wiki": 0.1, "books": 0.1, "code": 10.0}
    sampler.update_weights(losses, step=1000)  # full warmup so all sources unlocked
    after = sampler.get_weights()
    # code's weight should increase relative to wiki
    assert after["code"] > before["code"] or after["code"] > after["wiki"]


def test_sampler_sample_source_returns_datasource():
    sampler, _ = _make_sampler()
    result = sampler.sample_source()
    assert isinstance(result, DataSource)


def test_sampler_sample_source_returns_valid_source():
    sampler, sources = _make_sampler()
    source_names = {s.name for s in sources}
    for _ in range(20):
        result = sampler.sample_source()
        assert result.name in source_names


def test_sampler_get_weights_keys_match_source_names():
    sampler, sources = _make_sampler()
    weights = sampler.get_weights()
    expected_keys = {s.name for s in sources}
    assert set(weights.keys()) == expected_keys


def test_sampler_get_weights_sums_to_one():
    sampler, _ = _make_sampler()
    weights = sampler.get_weights()
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-5
