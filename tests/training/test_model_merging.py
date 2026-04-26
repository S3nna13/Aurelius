"""
Tests for src/training/model_merging.py.

All models use tiny configs (D=8) to keep tests fast.
Pure PyTorch — no HuggingFace, scipy, or sklearn.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.model_merging import (
    MergeConfig,
    ModelMerger,
    apply_task_vector,
    compute_task_vector,
    dare_mask,
    linear_merge,
    slerp_merge,
    ties_merge,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 8  # tiny hidden dimension


def tiny_linear() -> nn.Linear:
    """Return a tiny nn.Linear(D, D) with deterministic weights."""
    return nn.Linear(D, D)


def make_state_dict(seed: int) -> dict[str, torch.Tensor]:
    """Create a reproducible state dict for a tiny Linear(D, D)."""
    torch.manual_seed(seed)
    m = tiny_linear()
    return m.state_dict()


def make_model(seed: int) -> nn.Linear:
    torch.manual_seed(seed)
    m = tiny_linear()
    return m


# ---------------------------------------------------------------------------
# 1. MergeConfig defaults
# ---------------------------------------------------------------------------


class TestMergeConfigDefaults:
    def test_default_method(self):
        cfg = MergeConfig()
        assert cfg.method == "linear"

    def test_default_weights_is_none(self):
        cfg = MergeConfig()
        assert cfg.weights is None

    def test_default_density(self):
        cfg = MergeConfig()
        assert cfg.density == 1.0

    def test_default_lambda_coeff(self):
        cfg = MergeConfig()
        assert cfg.lambda_coeff == 1.0


# ---------------------------------------------------------------------------
# 2. linear_merge — correctness
# ---------------------------------------------------------------------------


class TestLinearMerge:
    def test_equal_weights_is_simple_average(self):
        sd1 = make_state_dict(0)
        sd2 = make_state_dict(1)
        merged = linear_merge([sd1, sd2], [0.5, 0.5])

        for k in sd1:
            expected = (sd1[k].float() + sd2[k].float()) / 2.0
            assert torch.allclose(merged[k].float(), expected, atol=1e-5), (
                f"Key '{k}': merged value differs from simple average."
            )

    def test_output_has_all_keys(self):
        sd1 = make_state_dict(0)
        sd2 = make_state_dict(1)
        merged = linear_merge([sd1, sd2], [0.5, 0.5])
        assert set(merged.keys()) == set(sd1.keys())

    def test_weight_one_on_first_returns_first(self):
        sd1 = make_state_dict(0)
        sd2 = make_state_dict(1)
        merged = linear_merge([sd1, sd2], [1.0, 0.0])
        for k in sd1:
            assert torch.allclose(merged[k].float(), sd1[k].float(), atol=1e-5)

    def test_weights_must_sum_to_one(self):
        sd1 = make_state_dict(0)
        sd2 = make_state_dict(1)
        with pytest.raises(ValueError, match="sum to 1"):
            linear_merge([sd1, sd2], [0.6, 0.6])

    def test_three_models_equal_weights(self):
        sds = [make_state_dict(i) for i in range(3)]
        w = [1.0 / 3, 1.0 / 3, 1.0 / 3]
        merged = linear_merge(sds, w)
        assert set(merged.keys()) == set(sds[0].keys())
        for k in sds[0]:
            expected = sum(sd[k].float() for sd in sds) / 3.0
            assert torch.allclose(merged[k].float(), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. slerp_merge — boundary conditions
# ---------------------------------------------------------------------------


class TestSlerpMerge:
    def test_t0_returns_first_model(self):
        sd1 = make_state_dict(0)
        sd2 = make_state_dict(1)
        merged = slerp_merge(sd1, sd2, t=0.0)
        for k in sd1:
            assert torch.allclose(merged[k].float(), sd1[k].float(), atol=1e-5), (
                f"Key '{k}': slerp(t=0) should equal sd1."
            )

    def test_t1_returns_second_model(self):
        sd1 = make_state_dict(0)
        sd2 = make_state_dict(1)
        merged = slerp_merge(sd1, sd2, t=1.0)
        for k in sd1:
            assert torch.allclose(merged[k].float(), sd2[k].float(), atol=1e-5), (
                f"Key '{k}': slerp(t=1) should equal sd2."
            )

    def test_t05_is_between_models(self):
        """Merged parameter norms should be between the two endpoint norms."""
        sd1 = make_state_dict(0)
        sd2 = make_state_dict(1)
        merged = slerp_merge(sd1, sd2, t=0.5)
        for k in sd1:
            p1 = sd1[k].float()
            p2 = sd2[k].float()
            pm = merged[k].float()
            if p1.ndim <= 1:
                continue  # skip scalar / bias — linear interp, still valid
            n1 = torch.norm(p1).item()
            n2 = torch.norm(p2).item()
            nm = torch.norm(pm).item()
            lo, hi = min(n1, n2), max(n1, n2)
            assert lo - 1e-4 <= nm <= hi + 1e-4, (
                f"Key '{k}': slerp(t=0.5) norm {nm:.4f} not in [{lo:.4f}, {hi:.4f}]."
            )

    def test_slerp_output_has_all_keys(self):
        sd1 = make_state_dict(0)
        sd2 = make_state_dict(1)
        merged = slerp_merge(sd1, sd2, t=0.5)
        assert set(merged.keys()) == set(sd1.keys())


# ---------------------------------------------------------------------------
# 4. compute_task_vector / apply_task_vector
# ---------------------------------------------------------------------------


class TestTaskVector:
    def test_compute_task_vector_difference(self):
        base = make_state_dict(0)
        ft = make_state_dict(1)
        tv = compute_task_vector(base, ft)
        for k in base:
            expected = ft[k].float() - base[k].float()
            assert torch.allclose(tv[k].float(), expected, atol=1e-5), (
                f"Key '{k}': task vector mismatch."
            )

    def test_apply_task_vector_scale0_returns_base(self):
        base = make_state_dict(0)
        ft = make_state_dict(1)
        tv = compute_task_vector(base, ft)
        result = apply_task_vector(base, tv, scale=0.0)
        for k in base:
            assert torch.allclose(result[k].float(), base[k].float(), atol=1e-5), (
                f"Key '{k}': scale=0 should return base."
            )

    def test_apply_task_vector_scale1_returns_finetuned(self):
        base = make_state_dict(0)
        ft = make_state_dict(1)
        tv = compute_task_vector(base, ft)
        result = apply_task_vector(base, tv, scale=1.0)
        for k in base:
            assert torch.allclose(result[k].float(), ft[k].float(), atol=1e-5), (
                f"Key '{k}': scale=1 should return fine-tuned weights."
            )


# ---------------------------------------------------------------------------
# 5. TIES merging
# ---------------------------------------------------------------------------


class TestTiesMerge:
    def test_ties_output_has_same_keys_as_base(self):
        base = make_state_dict(0)
        tv1 = compute_task_vector(base, make_state_dict(1))
        tv2 = compute_task_vector(base, make_state_dict(2))
        merged = ties_merge(base, [tv1, tv2])
        assert set(merged.keys()) == set(base.keys())

    def test_ties_output_is_finite(self):
        base = make_state_dict(0)
        tvs = [compute_task_vector(base, make_state_dict(i)) for i in range(1, 4)]
        merged = ties_merge(base, tvs, lambda_coeff=0.5)
        for k, v in merged.items():
            assert torch.isfinite(v).all(), f"Key '{k}': non-finite values after TIES."

    def test_ties_lambda0_returns_base(self):
        """With lambda_coeff=0 the merged result should equal the base."""
        base = make_state_dict(0)
        tv1 = compute_task_vector(base, make_state_dict(1))
        merged = ties_merge(base, [tv1], lambda_coeff=0.0)
        for k in base:
            assert torch.allclose(merged[k].float(), base[k].float(), atol=1e-5)


# ---------------------------------------------------------------------------
# 6. DARE masking
# ---------------------------------------------------------------------------


class TestDareMask:
    def test_dare_density1_close_to_identity(self):
        """density=1.0 keeps all elements — masked vector should equal original."""
        tv = compute_task_vector(make_state_dict(0), make_state_dict(1))
        masked = dare_mask(tv, density=1.0, seed=0)
        for k in tv:
            assert torch.allclose(masked[k].float(), tv[k].float(), atol=1e-5), (
                f"Key '{k}': density=1 should not alter the task vector."
            )

    def test_dare_reduces_magnitude_when_density_lt_1(self):
        """With density < 1, the expected L1 norm should be preserved but the
        raw (unscaled) sum of absolute values after *dropping* entries (before
        the 1/density rescaling) is lower. We verify that at least some entries
        are zeroed out."""
        tv = compute_task_vector(make_state_dict(0), make_state_dict(1))
        # Use density=0.5 with a fixed seed so the test is deterministic
        masked = dare_mask(tv, density=0.5, seed=7)
        for k in tv:
            orig = tv[k].float()
            mask_v = masked[k].float()
            # After rescaling by 1/density=2, some entries must be 0 (dropped)
            # and others must be ~2x the original; check at least one zero exists
            if orig.numel() > 4:
                zero_count = (mask_v == 0).sum().item()
                assert zero_count > 0, f"Key '{k}': expected some zeros for density=0.5, got none."

    def test_dare_output_has_all_keys(self):
        tv = compute_task_vector(make_state_dict(0), make_state_dict(1))
        masked = dare_mask(tv, density=0.7, seed=99)
        assert set(masked.keys()) == set(tv.keys())

    def test_dare_density0_zeros_everything(self):
        tv = compute_task_vector(make_state_dict(0), make_state_dict(1))
        masked = dare_mask(tv, density=0.0, seed=0)
        for k in masked:
            assert (masked[k] == 0).all(), f"Key '{k}': density=0 should zero all entries."


# ---------------------------------------------------------------------------
# 7. ModelMerger integration
# ---------------------------------------------------------------------------


class TestModelMerger:
    def test_merge_returns_dict(self):
        cfg = MergeConfig(method="linear")
        merger = ModelMerger(cfg)
        m1, m2 = make_model(0), make_model(1)
        result = merger.merge([m1, m2])
        assert isinstance(result, dict)

    def test_load_merged_loads_without_error(self):
        cfg = MergeConfig(method="linear")
        merger = ModelMerger(cfg)
        m1, m2 = make_model(0), make_model(1)
        merged_sd = merger.merge([m1, m2])
        target = tiny_linear()
        loaded = merger.load_merged(target, merged_sd)
        assert loaded is target

    def test_merger_slerp_two_models(self):
        cfg = MergeConfig(method="slerp", weights=[0.5, 0.5])
        merger = ModelMerger(cfg)
        m1, m2 = make_model(0), make_model(1)
        result = merger.merge([m1, m2])
        assert isinstance(result, dict)
        assert set(result.keys()) == set(m1.state_dict().keys())

    def test_merger_ties(self):
        cfg = MergeConfig(method="ties", lambda_coeff=1.0)
        merger = ModelMerger(cfg)
        models = [make_model(i) for i in range(3)]
        result = merger.merge(models)
        assert isinstance(result, dict)
        for v in result.values():
            assert torch.isfinite(v).all()

    def test_merger_dare(self):
        cfg = MergeConfig(method="dare", density=0.5)
        merger = ModelMerger(cfg)
        models = [make_model(i) for i in range(3)]
        result = merger.merge(models)
        assert isinstance(result, dict)
        for v in result.values():
            assert torch.isfinite(v).all()

    def test_merger_requires_at_least_two_models(self):
        cfg = MergeConfig(method="linear")
        merger = ModelMerger(cfg)
        with pytest.raises(ValueError, match="[Aa]t least two"):
            merger.merge([make_model(0)])

    def test_merger_unknown_method_raises(self):
        cfg = MergeConfig(method="unknown_method")
        merger = ModelMerger(cfg)
        with pytest.raises(ValueError, match="Unknown merge method"):
            merger.merge([make_model(0), make_model(1)])

    def test_loaded_model_produces_correct_output(self):
        """After loading, the model's forward pass should differ from both originals."""
        cfg = MergeConfig(method="linear")
        merger = ModelMerger(cfg)
        m1, m2 = make_model(0), make_model(1)
        merged_sd = merger.merge([m1, m2])
        target = tiny_linear()
        merger.load_merged(target, merged_sd)

        torch.manual_seed(42)
        x = torch.randn(4, D)
        with torch.no_grad():
            y_merged = target(x)
            y1 = m1(x)
            y2 = m2(x)

        # Merged output should be the average of the two models' outputs
        expected = (y1 + y2) / 2.0
        assert torch.allclose(y_merged, expected, atol=1e-5)
