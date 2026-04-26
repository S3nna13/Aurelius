"""Tests for distillation_trainer."""

from __future__ import annotations

import math

import pytest

from src.compression.distillation_trainer import (
    DISTILLATION_TRAINER_REGISTRY,
    DistillationTrainer,
    DistillConfig,
    DistillResult,
)

# --- config ---


def test_config_defaults():
    c = DistillConfig()
    assert c.temperature == 4.0
    assert c.alpha == 0.5
    assert c.hard_label_weight == 0.5
    assert c.soft_label_weight == 0.5


def test_config_custom():
    c = DistillConfig(temperature=8.0, alpha=0.3)
    assert c.temperature == 8.0
    assert c.alpha == 0.3


# --- DistillResult ---


def test_result_fields():
    r = DistillResult(hard_loss=0.1, soft_loss=0.2, total_loss=0.15, alpha=0.5)
    assert r.hard_loss == 0.1
    assert r.soft_loss == 0.2
    assert r.total_loss == 0.15
    assert r.alpha == 0.5


def test_result_frozen():
    r = DistillResult(0.1, 0.2, 0.15, 0.5)
    with pytest.raises(Exception):
        r.alpha = 0.9  # type: ignore[misc]


# --- soft_targets ---


def test_soft_targets_sum_to_one():
    t = DistillationTrainer()
    probs = t.soft_targets([1.0, 2.0, 3.0], 1.0)
    assert sum(probs) == pytest.approx(1.0)


def test_soft_targets_length():
    t = DistillationTrainer()
    probs = t.soft_targets([1.0, 2.0, 3.0, 4.0], 2.0)
    assert len(probs) == 4


def test_soft_targets_empty():
    t = DistillationTrainer()
    assert t.soft_targets([], 1.0) == []


def test_soft_targets_higher_temperature_flatter():
    t = DistillationTrainer()
    low = t.soft_targets([1.0, 2.0, 5.0], 1.0)
    high = t.soft_targets([1.0, 2.0, 5.0], 10.0)

    # entropy high > low
    def ent(p):
        return -sum(pi * math.log(pi) for pi in p if pi > 0)

    assert ent(high) > ent(low)


def test_soft_targets_uniform_logits_uniform():
    t = DistillationTrainer()
    probs = t.soft_targets([1.0, 1.0, 1.0, 1.0], 1.0)
    for p in probs:
        assert p == pytest.approx(0.25)


def test_soft_targets_non_negative():
    t = DistillationTrainer()
    probs = t.soft_targets([-5.0, 0.0, 5.0], 3.0)
    assert all(p >= 0 for p in probs)


# --- kl_divergence ---


def test_kl_self_is_zero():
    t = DistillationTrainer()
    p = [0.25, 0.25, 0.25, 0.25]
    assert t.kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_kl_non_negative():
    t = DistillationTrainer()
    p = [0.1, 0.2, 0.7]
    q = [0.3, 0.3, 0.4]
    assert t.kl_divergence(p, q) >= 0.0


def test_kl_zero_p_safe():
    t = DistillationTrainer()
    p = [0.0, 1.0]
    q = [0.5, 0.5]
    # should not raise
    val = t.kl_divergence(p, q)
    assert val >= 0.0


def test_kl_zero_q_safe():
    t = DistillationTrainer()
    p = [0.5, 0.5]
    q = [0.0, 1.0]
    # does not blow up due to epsilon
    val = t.kl_divergence(p, q)
    assert math.isfinite(val)


def test_kl_length_mismatch():
    t = DistillationTrainer()
    with pytest.raises(ValueError):
        t.kl_divergence([0.5, 0.5], [1.0])


# --- distill_step ---


def test_distill_step_returns_result():
    t = DistillationTrainer()
    r = t.distill_step([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [2])
    assert isinstance(r, DistillResult)


def test_distill_step_total_formula():
    t = DistillationTrainer(DistillConfig(alpha=0.5, temperature=2.0))
    r = t.distill_step([1.0, 2.0, 3.0], [2.0, 1.0, 0.5], [0])
    expected = 0.5 * r.hard_loss + 0.5 * r.soft_loss
    assert r.total_loss == pytest.approx(expected)


def test_distill_step_alpha_one_only_hard():
    t = DistillationTrainer(DistillConfig(alpha=1.0))
    r = t.distill_step([1.0, 2.0], [5.0, 0.0], [0])
    assert r.total_loss == pytest.approx(r.hard_loss)


def test_distill_step_alpha_zero_only_soft():
    t = DistillationTrainer(DistillConfig(alpha=0.0))
    r = t.distill_step([1.0, 2.0], [5.0, 0.0], [0])
    assert r.total_loss == pytest.approx(r.soft_loss)


def test_distill_step_matching_logits_zero_soft():
    t = DistillationTrainer()
    r = t.distill_step([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0])
    assert r.soft_loss == pytest.approx(0.0, abs=1e-9)


def test_distill_step_hard_loss_correct_label():
    t = DistillationTrainer(DistillConfig(alpha=1.0))
    # high confidence on label 2
    r = t.distill_step([0.0, 0.0, 10.0], [0.0, 0.0, 10.0], [2])
    assert r.hard_loss < 0.01


def test_distill_step_hard_loss_wrong_label():
    t = DistillationTrainer(DistillConfig(alpha=1.0))
    r = t.distill_step([0.0, 0.0, 10.0], [0.0, 0.0, 10.0], [0])
    assert r.hard_loss > 1.0


def test_distill_step_records_alpha():
    t = DistillationTrainer(DistillConfig(alpha=0.3))
    r = t.distill_step([1.0, 2.0], [1.0, 2.0], [0])
    assert r.alpha == 0.3


# --- optimal_temperature ---


def test_optimal_temperature_small_classes_clamped_low():
    t = DistillationTrainer()
    assert t.optimal_temperature(2) == 2.0


def test_optimal_temperature_large_classes_clamped_high():
    t = DistillationTrainer()
    assert t.optimal_temperature(10_000) == 10.0


def test_optimal_temperature_mid_range():
    t = DistillationTrainer()
    val = t.optimal_temperature(25)
    assert val == pytest.approx(5.0)


def test_optimal_temperature_zero_safe():
    t = DistillationTrainer()
    assert 2.0 <= t.optimal_temperature(0) <= 10.0


# --- registry ---


def test_registry_has_default():
    assert "default" in DISTILLATION_TRAINER_REGISTRY


def test_registry_constructs():
    cls = DISTILLATION_TRAINER_REGISTRY["default"]
    assert isinstance(cls(), DistillationTrainer)
