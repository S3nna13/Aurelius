"""Tests for advanced weight merging (TIES, DARE, SLERP)."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.training.advanced_merge import (
    MergeConfig,
    ModelMerger,
    compute_task_vector,
    dare_mask,
    dare_merge,
    slerp_merge,
    ties_elect_sign,
    ties_merge,
    ties_trim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_weights(*names: str, value: float = 1.0) -> dict[str, torch.Tensor]:
    """Create a simple state dict with the given param names."""
    return {n: torch.full((4, 4), value) for n in names}


class _TinyModel(nn.Module):
    """Minimal model for merger integration tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# MergeConfig tests
# ---------------------------------------------------------------------------


class TestMergeConfig:
    def test_defaults(self):
        cfg = MergeConfig()
        assert cfg.method == "ties"
        assert cfg.density == 0.9
        assert cfg.alpha == 0.5
        assert cfg.seed == 42

    def test_custom_values(self):
        cfg = MergeConfig(method="dare", density=0.7, alpha=0.3, seed=123)
        assert cfg.method == "dare"
        assert cfg.density == 0.7
        assert cfg.alpha == 0.3
        assert cfg.seed == 123


# ---------------------------------------------------------------------------
# compute_task_vector tests
# ---------------------------------------------------------------------------


class TestComputeTaskVector:
    def test_basic_subtraction(self):
        base = _make_weights("a", "b", value=1.0)
        ft = _make_weights("a", "b", value=3.0)
        tv = compute_task_vector(base, ft)
        assert set(tv.keys()) == {"a", "b"}
        assert torch.allclose(tv["a"], torch.full((4, 4), 2.0))

    def test_missing_key_skipped(self):
        base = _make_weights("a", "b", value=1.0)
        ft = _make_weights("a", value=2.0)  # missing "b"
        tv = compute_task_vector(base, ft)
        assert "b" not in tv
        assert "a" in tv

    def test_shape_mismatch_skipped(self):
        base = {"w": torch.zeros(4, 4)}
        ft = {"w": torch.zeros(3, 3)}
        tv = compute_task_vector(base, ft)
        assert "w" not in tv


# ---------------------------------------------------------------------------
# TIES tests
# ---------------------------------------------------------------------------


class TestTiesTrim:
    def test_density_1_keeps_all(self):
        tv = [{"w": torch.tensor([1.0, -2.0, 3.0, -0.5])}]
        trimmed = ties_trim(tv, density=1.0)
        assert torch.allclose(trimmed[0]["w"], tv[0]["w"])

    def test_density_trims_small_values(self):
        tv = [{"w": torch.tensor([10.0, 0.1, 0.2, 5.0])}]
        trimmed = ties_trim(tv, density=0.5)
        # Should keep 2 largest (10.0, 5.0) and zero out 2 smallest
        result = trimmed[0]["w"]
        assert result[0].item() == 10.0
        assert result[3].item() == 5.0
        # Smaller values should be zeroed
        assert result[1].item() == 0.0


class TestTiesElectSign:
    def test_majority_positive(self):
        tvs = [
            {"w": torch.tensor([1.0, -1.0, 1.0])},
            {"w": torch.tensor([1.0, 1.0, -1.0])},
            {"w": torch.tensor([1.0, 1.0, 1.0])},
        ]
        signs = ties_elect_sign(tvs)
        # Position 0: all positive -> +1
        assert signs["w"][0].item() == 1.0
        # Position 1: 2 positive, 1 negative -> +1
        assert signs["w"][1].item() == 1.0
        # Position 2: 2 positive, 1 negative -> +1
        assert signs["w"][2].item() == 1.0

    def test_majority_negative(self):
        tvs = [
            {"w": torch.tensor([-1.0])},
            {"w": torch.tensor([-1.0])},
            {"w": torch.tensor([1.0])},
        ]
        signs = ties_elect_sign(tvs)
        assert signs["w"][0].item() == -1.0


class TestTiesMerge:
    def test_merge_adds_to_base(self):
        base = {"w": torch.tensor([0.0, 0.0])}
        tvs = [{"w": torch.tensor([2.0, 2.0])}]
        merged = ties_merge(base, tvs, density=1.0, alpha=1.0)
        # With single task vector, density=1, alpha=1, should add full delta
        assert torch.allclose(merged["w"], torch.tensor([2.0, 2.0]))

    def test_alpha_scales_delta(self):
        base = {"w": torch.tensor([0.0, 0.0])}
        tvs = [{"w": torch.tensor([4.0, 4.0])}]
        merged = ties_merge(base, tvs, density=1.0, alpha=0.5)
        assert torch.allclose(merged["w"], torch.tensor([2.0, 2.0]))

    def test_unmatched_key_preserved(self):
        base = {"w": torch.tensor([1.0]), "bias": torch.tensor([5.0])}
        tvs = [{"w": torch.tensor([2.0])}]  # no "bias" key
        merged = ties_merge(base, tvs, density=1.0, alpha=1.0)
        assert torch.allclose(merged["bias"], torch.tensor([5.0]))


# ---------------------------------------------------------------------------
# DARE tests
# ---------------------------------------------------------------------------


class TestDareMask:
    def test_rescaling(self):
        tv = {"w": torch.ones(1000)}
        masked = dare_mask(tv, density=0.5, seed=42)
        # Non-zero values should be rescaled by 1/density = 2.0
        nonzero = masked["w"][masked["w"] != 0]
        assert torch.allclose(nonzero, torch.full_like(nonzero, 2.0))

    def test_density_1_keeps_all(self):
        tv = {"w": torch.ones(100)}
        masked = dare_mask(tv, density=1.0, seed=0)
        # Everything kept, rescaled by 1/1 = 1
        assert torch.allclose(masked["w"], torch.ones(100))

    def test_reproducibility(self):
        tv = {"w": torch.randn(100)}
        m1 = dare_mask(tv, density=0.5, seed=42)
        m2 = dare_mask(tv, density=0.5, seed=42)
        assert torch.allclose(m1["w"], m2["w"])


class TestDareMerge:
    def test_single_tv(self):
        base = {"w": torch.zeros(1000)}
        tvs = [{"w": torch.ones(1000)}]
        merged = dare_merge(base, tvs, density=1.0, alpha=1.0, seed=42)
        # density=1 keeps all, alpha=1 adds full delta
        assert torch.allclose(merged["w"], torch.ones(1000))

    def test_alpha_scaling(self):
        base = {"w": torch.zeros(10)}
        tvs = [{"w": torch.ones(10)}]
        merged = dare_merge(base, tvs, density=1.0, alpha=0.5, seed=42)
        assert torch.allclose(merged["w"], torch.full((10,), 0.5))


# ---------------------------------------------------------------------------
# SLERP tests
# ---------------------------------------------------------------------------


class TestSlerpMerge:
    def test_alpha_0_returns_a(self):
        wa = {"w": torch.randn(4, 4)}
        wb = {"w": torch.randn(4, 4)}
        merged = slerp_merge(wa, wb, alpha=0.0)
        assert torch.allclose(merged["w"], wa["w"], atol=1e-5)

    def test_alpha_1_returns_b(self):
        wa = {"w": torch.randn(4, 4)}
        wb = {"w": torch.randn(4, 4)}
        merged = slerp_merge(wa, wb, alpha=1.0)
        assert torch.allclose(merged["w"], wb["w"], atol=1e-5)

    def test_alpha_half_midpoint(self):
        # For orthogonal vectors, slerp(0.5) should be at 45 degrees to each
        wa = {"w": torch.tensor([1.0, 0.0])}
        wb = {"w": torch.tensor([0.0, 1.0])}
        merged = slerp_merge(wa, wb, alpha=0.5)
        expected = torch.tensor([math.sqrt(2) / 2, math.sqrt(2) / 2])
        assert torch.allclose(merged["w"], expected, atol=1e-5)

    def test_missing_key_in_b_uses_a(self):
        wa = {"w": torch.ones(3), "extra": torch.full((3,), 9.0)}
        wb = {"w": torch.ones(3) * 2}
        merged = slerp_merge(wa, wb, alpha=0.5)
        assert torch.allclose(merged["extra"], torch.full((3,), 9.0))


# ---------------------------------------------------------------------------
# ModelMerger tests
# ---------------------------------------------------------------------------


class TestModelMerger:
    def test_ties_dispatch(self):
        cfg = MergeConfig(method="ties", density=1.0, alpha=0.5)
        merger = ModelMerger(cfg)
        base = _TinyModel()
        ft1 = _TinyModel()
        # Perturb ft1
        with torch.no_grad():
            ft1.linear.weight.add_(1.0)
        result = merger.merge(base, [ft1])
        assert "linear.weight" in result

    def test_dare_dispatch(self):
        cfg = MergeConfig(method="dare", density=1.0, alpha=1.0, seed=0)
        merger = ModelMerger(cfg)
        base = _TinyModel()
        ft1 = _TinyModel()
        with torch.no_grad():
            ft1.linear.weight.add_(2.0)
        result = merger.merge(base, [ft1])
        # With density=1 and alpha=1, should be close to ft1 weights
        expected = ft1.linear.weight
        assert torch.allclose(result["linear.weight"], expected, atol=1e-5)

    def test_slerp_dispatch(self):
        cfg = MergeConfig(method="slerp", alpha=0.0)
        merger = ModelMerger(cfg)
        base = _TinyModel()
        m1 = _TinyModel()
        m2 = _TinyModel()
        with torch.no_grad():
            m1.linear.weight.fill_(1.0)
            m2.linear.weight.fill_(2.0)
        result = merger.merge(base, [m1, m2])
        assert torch.allclose(result["linear.weight"], m1.linear.weight, atol=1e-5)

    def test_slerp_requires_two_models(self):
        cfg = MergeConfig(method="slerp")
        merger = ModelMerger(cfg)
        base = _TinyModel()
        with pytest.raises(ValueError, match="exactly 2"):
            merger.merge(base, [_TinyModel()])

    def test_unknown_method_raises(self):
        cfg = MergeConfig(method="unknown")
        merger = ModelMerger(cfg)
        with pytest.raises(ValueError, match="Unknown merge method"):
            merger.merge(_TinyModel(), [_TinyModel()])
