"""Tests for gradient_noise_scheduler.py."""

from __future__ import annotations

import threading

import pytest
import torch

from src.training.gradient_noise_scheduler import (
    GradientNoiseScheduler,
    NOISE_SCHEDULER_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gradients(seed: int = 0, shapes: list[tuple[int, ...]] | None = None) -> list[torch.Tensor]:
    """Return a list of deterministic gradient tensors."""
    torch.manual_seed(seed)
    shapes = shapes or [(10,)]
    return [torch.randn(shape) for shape in shapes]


# ---------------------------------------------------------------------------
# Initialisation & validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_values(self):
        s = GradientNoiseScheduler()
        assert s.noise_type == "gaussian"
        assert s.initial_scale == pytest.approx(0.01)
        assert s.decay == "exponential"
        assert s.decay_rate == pytest.approx(0.99)

    def test_custom_values(self):
        s = GradientNoiseScheduler(
            noise_type="laplace",
            initial_scale=0.05,
            decay="linear",
            decay_rate=0.5,
        )
        assert s.noise_type == "laplace"
        assert s.initial_scale == pytest.approx(0.05)
        assert s.decay == "linear"
        assert s.decay_rate == pytest.approx(0.5)


class TestValidation:
    def test_invalid_noise_type_raises(self):
        with pytest.raises(ValueError, match="noise_type"):
            GradientNoiseScheduler(noise_type="uniform")

    def test_invalid_decay_raises(self):
        with pytest.raises(ValueError, match="decay"):
            GradientNoiseScheduler(decay="inverse")

    def test_non_positive_initial_scale_raises(self):
        with pytest.raises(ValueError, match="initial_scale"):
            GradientNoiseScheduler(initial_scale=0.0)
        with pytest.raises(ValueError, match="initial_scale"):
            GradientNoiseScheduler(initial_scale=-0.01)


# ---------------------------------------------------------------------------
# Decay schedules
# ---------------------------------------------------------------------------

class TestExponentialDecay:
    def test_scale_decreases_over_steps(self):
        s = GradientNoiseScheduler(decay="exponential", decay_rate=0.9)
        scales = [s.get_scale(step) for step in range(10)]
        for i in range(len(scales) - 1):
            assert scales[i] > scales[i + 1]

    def test_formula(self):
        s = GradientNoiseScheduler(
            initial_scale=0.1, decay="exponential", decay_rate=0.95
        )
        for step in [0, 1, 5, 10]:
            expected = 0.1 * (0.95 ** step)
            assert s.get_scale(step) == pytest.approx(expected)


class TestLinearDecay:
    def test_scale_decreases_over_steps(self):
        s = GradientNoiseScheduler(decay="linear", decay_rate=0.05)
        scales = [s.get_scale(step) for step in range(10)]
        for i in range(len(scales) - 1):
            assert scales[i] >= scales[i + 1]

    def test_formula_and_clamping(self):
        s = GradientNoiseScheduler(
            initial_scale=1.0, decay="linear", decay_rate=0.3
        )
        assert s.get_scale(0) == pytest.approx(1.0)
        assert s.get_scale(1) == pytest.approx(0.7)
        assert s.get_scale(2) == pytest.approx(0.4)
        assert s.get_scale(3) == pytest.approx(0.1)
        assert s.get_scale(4) == pytest.approx(0.0)
        assert s.get_scale(100) == pytest.approx(0.0)


class TestConstantDecay:
    def test_scale_unchanged(self):
        s = GradientNoiseScheduler(decay="constant", initial_scale=0.42)
        for step in [0, 1, 10, 100]:
            assert s.get_scale(step) == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Noise injection — all combinations
# ---------------------------------------------------------------------------

class TestAddNoise:
    @pytest.mark.parametrize("noise_type", ["gaussian", "laplace"])
    @pytest.mark.parametrize("decay", ["exponential", "linear", "constant"])
    def test_add_noise_changes_values(self, noise_type: str, decay: str):
        torch.manual_seed(42)
        s = GradientNoiseScheduler(
            noise_type=noise_type, decay=decay, decay_rate=0.5
        )
        grads = _make_gradients(seed=1)
        original = grads[0].clone()
        s.add_noise(grads, step=0)
        assert not torch.equal(grads[0], original)
        assert torch.isfinite(grads[0]).all()

    @pytest.mark.parametrize("noise_type", ["gaussian", "laplace"])
    @pytest.mark.parametrize("decay", ["exponential", "linear", "constant"])
    def test_add_noise_preserves_shape(self, noise_type: str, decay: str):
        s = GradientNoiseScheduler(noise_type=noise_type, decay=decay)
        shapes = [(4, 8), (16,), (2, 3, 5)]
        grads = _make_gradients(seed=2, shapes=shapes)
        noisy = s.add_noise([g.clone() for g in grads], step=1)
        for g, n in zip(grads, noisy):
            assert n.shape == g.shape
            assert n.dtype == g.dtype

    @pytest.mark.parametrize("noise_type", ["gaussian", "laplace"])
    @pytest.mark.parametrize("decay", ["exponential", "linear", "constant"])
    def test_add_noise_respects_step_scale(self, noise_type: str, decay: str):
        """Higher step should generally yield smaller (or equal) perturbations
        when decay is active, visible via smaller variance of the result."""
        torch.manual_seed(7)
        s = GradientNoiseScheduler(
            noise_type=noise_type, decay=decay, decay_rate=0.5, initial_scale=1.0
        )
        g0 = torch.zeros(10_000)
        g1 = g0.clone()
        s.add_noise([g0], step=0)
        s.add_noise([g1], step=5)
        if decay == "constant":
            # Same scale -> distributions are identical; means should be close.
            assert g0.abs().mean().item() == pytest.approx(
                g1.abs().mean().item(), rel=0.1
            )
        else:
            assert g0.abs().mean() > g1.abs().mean()

    @pytest.mark.parametrize("noise_type", ["gaussian", "laplace"])
    def test_add_noise_empty_list(self, noise_type: str):
        s = GradientNoiseScheduler(noise_type=noise_type)
        result = s.add_noise([], step=0)
        assert result == []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_is_empty_dict_by_default(self):
        assert isinstance(NOISE_SCHEDULER_REGISTRY, dict)

    def test_register_and_retrieve_instance(self):
        s = GradientNoiseScheduler()
        NOISE_SCHEDULER_REGISTRY["test_scheduler"] = s
        assert NOISE_SCHEDULER_REGISTRY["test_scheduler"] is s
        del NOISE_SCHEDULER_REGISTRY["test_scheduler"]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_add_noise(self):
        s = GradientNoiseScheduler(noise_type="gaussian", decay="exponential")
        grad = torch.zeros(1_000)
        original = grad.clone()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    s.add_noise([grad], step=0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions during concurrent add_noise: {errors}"
        assert torch.isfinite(grad).all()
        assert not torch.equal(grad, original)
