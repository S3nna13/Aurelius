"""
Tests for src/training/data_mixture.py
"""

import math
import sys
import os

import pytest

# Ensure project root is on sys.path so the src package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.training.data_mixture import (
    MixtureConfig,
    DataMixtureSampler,
    compute_curriculum_weights,
    compute_mixture_entropy,
    estimate_steps_per_epoch,
    normalize_weights,
    temperature_sample_weights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPSILON = 1e-6


def approx_equal(a: float, b: float, tol: float = EPSILON) -> bool:
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# MixtureConfig defaults  (test 1)
# ---------------------------------------------------------------------------

class TestMixtureConfigDefaults:
    def test_default_source_names(self):
        cfg = MixtureConfig()
        assert cfg.source_names == ["web", "books", "code"]

    def test_default_weights_none(self):
        cfg = MixtureConfig()
        assert cfg.weights is None

    def test_default_temperature(self):
        cfg = MixtureConfig()
        assert cfg.temperature == 1.0

    def test_default_curriculum_steps(self):
        cfg = MixtureConfig()
        assert cfg.curriculum_steps == 0

    def test_default_total_steps(self):
        cfg = MixtureConfig()
        assert cfg.total_steps == 10_000


# ---------------------------------------------------------------------------
# normalize_weights  (tests 2–3)
# ---------------------------------------------------------------------------

class TestNormalizeWeights:
    def test_sums_to_one(self):
        result = normalize_weights([1.0, 2.0, 3.0])
        assert approx_equal(sum(result), 1.0)

    def test_correct_values(self):
        result = normalize_weights([1.0, 1.0, 2.0])
        assert approx_equal(result[0], 0.25)
        assert approx_equal(result[1], 0.25)
        assert approx_equal(result[2], 0.50)

    def test_raises_on_negative_weight(self):
        with pytest.raises(ValueError):
            normalize_weights([1.0, -0.1, 2.0])

    def test_raises_on_all_zeros(self):
        with pytest.raises(ValueError):
            normalize_weights([0.0, 0.0])

    def test_single_weight_returns_one(self):
        assert approx_equal(normalize_weights([5.0])[0], 1.0)


# ---------------------------------------------------------------------------
# temperature_sample_weights  (tests 4–6)
# ---------------------------------------------------------------------------

class TestTemperatureSampleWeights:
    def test_sums_to_one(self):
        result = temperature_sample_weights([0.5, 0.3, 0.2], temperature=1.5)
        assert approx_equal(sum(result), 1.0)

    def test_t_equals_one_identity(self):
        """T=1 should give back the same (normalized) weights."""
        raw = [1.0, 2.0, 3.0]
        normalized = normalize_weights(raw)
        result = temperature_sample_weights(normalized, temperature=1.0)
        for a, b in zip(normalized, result):
            assert approx_equal(a, b, tol=1e-5)

    def test_high_temperature_approaches_uniform(self):
        """Very high T should make all weights nearly equal."""
        result = temperature_sample_weights([0.1, 0.5, 0.4], temperature=1000.0)
        n = len(result)
        for w in result:
            assert abs(w - 1.0 / n) < 0.01

    def test_raises_on_non_positive_temperature(self):
        with pytest.raises(ValueError):
            temperature_sample_weights([0.5, 0.5], temperature=0.0)

    def test_low_temperature_concentrates_mass(self):
        """Low T should concentrate mass on largest weight."""
        weights = [0.1, 0.8, 0.1]  # middle is largest
        result = temperature_sample_weights(weights, temperature=0.01)
        assert result[1] > 0.99  # nearly all mass on index 1


# ---------------------------------------------------------------------------
# compute_curriculum_weights  (tests 7–8)
# ---------------------------------------------------------------------------

class TestComputeCurriculumWeights:
    def test_step_zero_is_uniform(self):
        base = [0.6, 0.3, 0.1]
        result = compute_curriculum_weights(base, step=0, curriculum_steps=100)
        n = len(base)
        for w in result:
            assert approx_equal(w, 1.0 / n, tol=1e-5)

    def test_step_at_curriculum_equals_base(self):
        base = [0.6, 0.3, 0.1]
        result = compute_curriculum_weights(base, step=100, curriculum_steps=100)
        normalized_base = normalize_weights(base)
        for a, b in zip(result, normalized_base):
            assert approx_equal(a, b, tol=1e-5)

    def test_step_beyond_curriculum_equals_base(self):
        base = [0.5, 0.5]
        result = compute_curriculum_weights(base, step=500, curriculum_steps=100)
        normalized_base = normalize_weights(base)
        for a, b in zip(result, normalized_base):
            assert approx_equal(a, b, tol=1e-5)

    def test_midpoint_is_interpolated(self):
        base = [0.9, 0.1]
        # uniform = [0.5, 0.5]; at step=50/100 alpha=0.5 → [0.7, 0.3]
        result = compute_curriculum_weights(base, step=50, curriculum_steps=100)
        assert approx_equal(result[0], 0.7, tol=1e-5)
        assert approx_equal(result[1], 0.3, tol=1e-5)

    def test_no_curriculum_returns_base(self):
        base = [2.0, 1.0]
        result = compute_curriculum_weights(base, step=0, curriculum_steps=0)
        normalized_base = normalize_weights(base)
        for a, b in zip(result, normalized_base):
            assert approx_equal(a, b, tol=1e-5)

    def test_result_sums_to_one(self):
        base = [0.4, 0.4, 0.2]
        for step in [0, 25, 50, 100, 200]:
            result = compute_curriculum_weights(base, step=step, curriculum_steps=100)
            assert approx_equal(sum(result), 1.0)


# ---------------------------------------------------------------------------
# DataMixtureSampler  (tests 9–12)
# ---------------------------------------------------------------------------

class TestDataMixtureSampler:
    def _make_sampler(self, **kwargs):
        cfg = MixtureConfig(
            source_names=["web", "books", "code"],
            weights=[0.5, 0.3, 0.2],
            **kwargs,
        )
        return DataMixtureSampler(cfg)

    def test_sample_source_returns_valid_name(self):
        sampler = self._make_sampler()
        source = sampler.sample_source(step=0)
        assert source in ["web", "books", "code"]

    def test_sample_batch_sources_correct_length(self):
        sampler = self._make_sampler()
        batch = sampler.sample_batch_sources(batch_size=32, step=0)
        assert len(batch) == 32
        assert all(s in ["web", "books", "code"] for s in batch)

    def test_get_weights_at_step_sums_to_one(self):
        sampler = self._make_sampler(curriculum_steps=100)
        for step in [0, 50, 100, 200]:
            w = sampler.get_weights_at_step(step)
            assert approx_equal(sum(w.values()), 1.0)

    def test_get_weights_at_step_has_all_sources(self):
        sampler = self._make_sampler()
        w = sampler.get_weights_at_step(step=0)
        assert set(w.keys()) == {"web", "books", "code"}

    def test_update_weights_changes_weights(self):
        sampler = self._make_sampler()
        old_w = sampler.get_weights_at_step(step=1000)
        sampler.update_weights({"web": 0.9, "books": 0.05, "code": 0.05})
        new_w = sampler.get_weights_at_step(step=1000)
        # web should now dominate
        assert new_w["web"] > old_w["web"]

    def test_update_weights_missing_source_raises(self):
        sampler = self._make_sampler()
        with pytest.raises(ValueError):
            sampler.update_weights({"web": 0.8, "books": 0.2})  # missing "code"

    def test_uniform_default_weights(self):
        cfg = MixtureConfig(source_names=["a", "b", "c"])  # weights=None
        sampler = DataMixtureSampler(cfg)
        w = sampler.get_weights_at_step(step=1000)
        for v in w.values():
            assert approx_equal(v, 1.0 / 3, tol=1e-5)

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError):
            DataMixtureSampler(
                MixtureConfig(
                    source_names=["web", "books"],
                    weights=[0.5, 0.3, 0.2],  # length mismatch
                )
            )


# ---------------------------------------------------------------------------
# compute_mixture_entropy  (tests 13–14)
# ---------------------------------------------------------------------------

class TestComputeMixtureEntropy:
    def test_uniform_entropy_equals_log_n(self):
        n = 4
        uniform = [1.0 / n] * n
        H = compute_mixture_entropy(uniform)
        assert approx_equal(H, math.log(n), tol=1e-6)

    def test_concentrated_entropy_lower_than_uniform(self):
        n = 4
        uniform = [1.0 / n] * n
        concentrated = [0.97, 0.01, 0.01, 0.01]
        H_uniform = compute_mixture_entropy(uniform)
        H_conc = compute_mixture_entropy(concentrated)
        assert H_conc < H_uniform

    def test_zero_weight_handled_gracefully(self):
        # w=0 contributes 0 to entropy (limit 0*log(0)=0)
        weights = [0.5, 0.5, 0.0]
        H = compute_mixture_entropy(weights)
        assert approx_equal(H, math.log(2), tol=1e-6)

    def test_pure_distribution_entropy_zero(self):
        H = compute_mixture_entropy([1.0, 0.0, 0.0])
        assert approx_equal(H, 0.0)


# ---------------------------------------------------------------------------
# estimate_steps_per_epoch  (test 15)
# ---------------------------------------------------------------------------

class TestEstimateStepsPerEpoch:
    def test_returns_positive_int(self):
        sizes = {"web": 1_000_000, "books": 500_000, "code": 300_000}
        weights = normalize_weights([0.5, 0.3, 0.2])
        steps = estimate_steps_per_epoch(sizes, weights, batch_size=512)
        assert isinstance(steps, int)
        assert steps > 0

    def test_correct_value(self):
        sizes = {"web": 1000, "books": 1000}
        weights = [0.5, 0.5]
        steps = estimate_steps_per_epoch(sizes, weights, batch_size=100)
        assert steps == 20  # 2000 // 100

    def test_raises_on_zero_batch_size(self):
        with pytest.raises(ValueError):
            estimate_steps_per_epoch({"web": 1000}, [1.0], batch_size=0)

    def test_minimum_one_step(self):
        # batch_size larger than total tokens → should return 1
        sizes = {"web": 5}
        steps = estimate_steps_per_epoch(sizes, [1.0], batch_size=1000)
        assert steps >= 1
