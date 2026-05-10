"""Tests for aurelius.training.model_merging_v2."""

from __future__ import annotations

import math

import pytest
import torch
from aurelius.training.model_merging_v2 import (
    DAREMerge,
    LinearMerge,
    MergeConfig,
    ModelMerger,
    SLERPMerge,
    TIESMerge,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sd(seed: int, shapes=None):
    """Return a simple state dict with deterministic values."""
    if shapes is None:
        shapes = {"w": (4, 4), "b": (4,)}
    torch.manual_seed(seed)
    return {k: torch.randn(*v) for k, v in shapes.items()}


def _sd_close(sd_a, sd_b, atol=1e-5):
    """Return True if all tensors in two state dicts are close."""
    for key in sd_a:
        if not torch.allclose(sd_a[key].float(), sd_b[key].float(), atol=atol):
            return False
    return True


# ---------------------------------------------------------------------------
# MergeConfig tests
# ---------------------------------------------------------------------------


class TestMergeConfig:
    def test_defaults(self):
        cfg = MergeConfig()
        assert cfg.merge_method == "linear"
        assert cfg.alpha == 0.5
        assert cfg.dare_density == 0.9
        assert cfg.ties_k == 0.2

    def test_custom(self):
        cfg = MergeConfig(merge_method="slerp", alpha=0.3, dare_density=0.7, ties_k=0.5)
        assert cfg.merge_method == "slerp"
        assert cfg.alpha == 0.3
        assert cfg.dare_density == 0.7
        assert cfg.ties_k == 0.5


# ---------------------------------------------------------------------------
# LinearMerge tests
# ---------------------------------------------------------------------------


class TestLinearMerge:
    def test_merge_two_alpha_zero_returns_sd_a(self):
        lm = LinearMerge()
        sd_a = _make_sd(0)
        sd_b = _make_sd(1)
        result = lm.merge_two(sd_a, sd_b, alpha=0.0)
        assert _sd_close(result, sd_a)

    def test_merge_two_alpha_one_returns_sd_b(self):
        lm = LinearMerge()
        sd_a = _make_sd(0)
        sd_b = _make_sd(1)
        result = lm.merge_two(sd_a, sd_b, alpha=1.0)
        assert _sd_close(result, sd_b)

    def test_merge_uniform_average(self):
        lm = LinearMerge()
        sd_a = _make_sd(0)
        sd_b = _make_sd(1)
        result = lm.merge([sd_a, sd_b])
        expected = {k: ((sd_a[k].float() + sd_b[k].float()) / 2.0).to(sd_a[k].dtype) for k in sd_a}
        assert _sd_close(result, expected)

    def test_merge_two_midpoint(self):
        lm = LinearMerge()
        sd_a = _make_sd(0)
        sd_b = _make_sd(1)
        result = lm.merge_two(sd_a, sd_b, alpha=0.5)
        expected = {k: ((sd_a[k].float() + sd_b[k].float()) / 2.0).to(sd_a[k].dtype) for k in sd_a}
        assert _sd_close(result, expected)

    def test_identical_models_return_same(self):
        lm = LinearMerge()
        sd = _make_sd(42)
        result = lm.merge([sd, sd])
        assert _sd_close(result, sd)


# ---------------------------------------------------------------------------
# SLERPMerge tests
# ---------------------------------------------------------------------------


class TestSLERPMerge:
    def test_slerp_at_t0_returns_v0(self):
        sm = SLERPMerge()
        v0 = torch.randn(8)
        v1 = torch.randn(8)
        result = sm.slerp(v0, v1, t=0.0)
        assert torch.allclose(result.float(), v0.float(), atol=1e-5)

    def test_slerp_at_t1_returns_v1(self):
        sm = SLERPMerge()
        v0 = torch.randn(8)
        v1 = torch.randn(8)
        result = sm.slerp(v0, v1, t=1.0)
        assert torch.allclose(result.float(), v1.float(), atol=1e-5)

    def test_slerp_norm_preserved(self):
        """The norm of the SLERPed vector should be between ‖v0‖ and ‖v1‖
        (exactly their geometric mean when t=0.5 for equal-norm inputs)."""
        sm = SLERPMerge()
        # Use unit vectors for clean norm preservation
        v0 = torch.randn(16)
        v0 = v0 / v0.norm()
        v1 = torch.randn(16)
        v1 = v1 / v1.norm()
        result = sm.slerp(v0, v1, t=0.5)
        result_norm = result.float().norm().item()
        # For unit input vectors, output norm should be ≈ 1
        assert abs(result_norm - 1.0) < 0.01

    def test_slerp_parallel_fallback(self):
        """SLERP of nearly-identical vectors should return approximately v0 at t=0."""
        sm = SLERPMerge()
        v0 = torch.tensor([1.0, 0.0, 0.0])
        v1 = v0.clone()
        result = sm.slerp(v0, v1, t=0.5)
        assert torch.allclose(result.float(), v0.float(), atol=1e-5)

    def test_slerp_merge_state_dict(self):
        sm = SLERPMerge()
        sd_a = _make_sd(0)
        sd_b = _make_sd(1)
        result = sm.merge(sd_a, sd_b, t=0.5)
        assert set(result.keys()) == set(sd_a.keys())
        for key in sd_a:
            assert result[key].shape == sd_a[key].shape


# ---------------------------------------------------------------------------
# TIESMerge tests
# ---------------------------------------------------------------------------


class TestTIESMerge:
    def test_trim_delta_keeps_top_k_fraction(self):
        tm = TIESMerge()
        torch.manual_seed(7)
        delta = torch.randn(100)
        k = 0.2
        trimmed = tm.trim_delta(delta, k)
        n_nonzero = (trimmed != 0).sum().item()
        expected = max(1, math.ceil(k * 100))
        assert n_nonzero <= expected + 2  # small tolerance for ties in topk

    def test_trim_delta_zeros_rest(self):
        tm = TIESMerge()
        delta = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        trimmed = tm.trim_delta(delta, k=0.4)  # keep top 2
        n_nonzero = (trimmed != 0).sum().item()
        assert n_nonzero == 2
        # The kept values should be the top-2
        assert trimmed[4].item() != 0  # 5.0 kept
        assert trimmed[3].item() != 0  # 4.0 kept

    def test_resolve_signs_returns_signs(self):
        tm = TIESMerge()
        d0 = torch.tensor([1.0, -1.0, 2.0])
        d1 = torch.tensor([1.0, 1.0, -3.0])
        d2 = torch.tensor([1.0, -1.0, 2.0])
        majority = tm.resolve_signs([d0, d1, d2])
        # col 0: +1+1+1 → +1
        # col 1: -1+1-1 → -1
        # col 2: +1-1+1 → +1
        expected = torch.tensor([1.0, -1.0, 1.0])
        assert torch.allclose(majority.float(), expected)

    def test_ties_merge_valid_state_dict(self):
        tm = TIESMerge()
        base = _make_sd(0)
        ft1 = _make_sd(1)
        ft2 = _make_sd(2)
        result = tm.merge(base, [ft1, ft2], k=0.2)
        assert set(result.keys()) == set(base.keys())
        for key in base:
            assert result[key].shape == base[key].shape
            assert not torch.isnan(result[key]).any()


# ---------------------------------------------------------------------------
# DAREMerge tests
# ---------------------------------------------------------------------------


class TestDAREMerge:
    def test_dare_prune_expected_density(self):
        dm = DAREMerge()
        torch.manual_seed(0)
        delta = torch.ones(10_000)
        density = 0.3
        pruned = dm.dare_prune(delta, density)
        fraction_nonzero = (pruned != 0).float().mean().item()
        # Allow ±5 % tolerance around target density
        assert abs(fraction_nonzero - density) < 0.05

    def test_dare_prune_rescaling(self):
        """With density=1.0 the output should equal the input (no pruning)."""
        dm = DAREMerge()
        delta = torch.randn(50)
        pruned = dm.dare_prune(delta, density=1.0)
        assert torch.allclose(pruned.float(), delta.float())

    def test_dare_merge_valid_state_dict(self):
        dm = DAREMerge()
        base = _make_sd(0)
        ft = _make_sd(1)
        result = dm.merge(base, ft, density=0.9, alpha=0.5)
        assert set(result.keys()) == set(base.keys())
        for key in base:
            assert result[key].shape == base[key].shape
            assert not torch.isnan(result[key]).any()

    def test_dare_merge_base_only_when_alpha_zero(self):
        """alpha=0 means no delta is applied, so result should equal base."""
        dm = DAREMerge()
        base = _make_sd(0)
        ft = _make_sd(1)
        result = dm.merge(base, ft, density=0.9, alpha=0.0)
        assert _sd_close(result, base)


# ---------------------------------------------------------------------------
# ModelMerger facade tests
# ---------------------------------------------------------------------------


class TestModelMerger:
    def test_dispatches_to_linear(self):
        cfg = MergeConfig(merge_method="linear")
        merger = ModelMerger(cfg)
        sd_a = _make_sd(0)
        sd_b = _make_sd(1)
        result = merger.merge(sd_a, sd_b)
        expected = LinearMerge().merge([sd_a, sd_b])
        assert _sd_close(result, expected)

    def test_dispatches_to_slerp(self):
        cfg = MergeConfig(merge_method="slerp", alpha=0.5)
        merger = ModelMerger(cfg)
        sd_a = _make_sd(0)
        sd_b = _make_sd(1)
        result = merger.merge(sd_a, sd_b)
        assert set(result.keys()) == set(sd_a.keys())

    def test_dispatches_to_ties(self):
        cfg = MergeConfig(merge_method="ties", ties_k=0.2)
        merger = ModelMerger(cfg)
        base = _make_sd(0)
        ft = _make_sd(1)
        result = merger.merge(base, ft)
        assert set(result.keys()) == set(base.keys())

    def test_dispatches_to_dare(self):
        cfg = MergeConfig(merge_method="dare", dare_density=0.9, alpha=0.5)
        merger = ModelMerger(cfg)
        base = _make_sd(0)
        ft = _make_sd(1)
        result = merger.merge(base, ft)
        assert set(result.keys()) == set(base.keys())

    def test_dispatches_to_task_arithmetic(self):
        cfg = MergeConfig(merge_method="task_arithmetic", alpha=0.5)
        merger = ModelMerger(cfg)
        base = _make_sd(0)
        ft = _make_sd(1)
        result = merger.merge(base, ft)
        assert set(result.keys()) == set(base.keys())

    def test_unknown_method_raises(self):
        cfg = MergeConfig(merge_method="unknown_method")
        merger = ModelMerger(cfg)
        sd = _make_sd(0)
        with pytest.raises(ValueError, match="unknown_method"):
            merger.merge(sd, sd)

    def test_linear_merge_identical_models(self):
        """Merging two copies of the same model should return that model."""
        cfg = MergeConfig(merge_method="linear")
        merger = ModelMerger(cfg)
        sd = _make_sd(42)
        result = merger.merge(sd, sd)
        assert _sd_close(result, sd)
