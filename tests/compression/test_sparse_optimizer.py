"""Tests for src/compression/sparse_optimizer.py (≥28 tests)."""
from __future__ import annotations

import dataclasses
import pytest

from src.compression.sparse_optimizer import (
    SparseOptimizer,
    SparseOptimizerConfig,
    SparseUpdate,
    SPARSE_OPTIMIZER_REGISTRY,
)


# ---------------------------------------------------------------------------
# SparseOptimizerConfig
# ---------------------------------------------------------------------------

class TestSparseOptimizerConfig:
    def test_defaults(self):
        cfg = SparseOptimizerConfig()
        assert cfg.lr == pytest.approx(0.01)
        assert cfg.sparsity_threshold == pytest.approx(0.0)
        assert cfg.skip_sparse_ratio == pytest.approx(0.0)

    def test_custom(self):
        cfg = SparseOptimizerConfig(lr=0.001, sparsity_threshold=1e-4, skip_sparse_ratio=0.5)
        assert cfg.lr == pytest.approx(0.001)

    def test_frozen(self):
        cfg = SparseOptimizerConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.lr = 1.0  # type: ignore[misc]

    def test_frozen_skip_ratio(self):
        cfg = SparseOptimizerConfig(skip_sparse_ratio=0.3)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.skip_sparse_ratio = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SparseUpdate
# ---------------------------------------------------------------------------

class TestSparseUpdate:
    def test_fields(self):
        u = SparseUpdate(param_id="p0", grad_nnz=8, grad_total=10, applied=True)
        assert u.param_id == "p0"
        assert u.grad_nnz == 8
        assert u.grad_total == 10
        assert u.applied is True

    def test_sparsity_property(self):
        u = SparseUpdate(param_id="p", grad_nnz=3, grad_total=10, applied=True)
        assert u.sparsity == pytest.approx(0.7)

    def test_sparsity_all_zero(self):
        u = SparseUpdate(param_id="p", grad_nnz=0, grad_total=10, applied=False)
        assert u.sparsity == pytest.approx(1.0)

    def test_sparsity_all_nonzero(self):
        u = SparseUpdate(param_id="p", grad_nnz=10, grad_total=10, applied=True)
        assert u.sparsity == pytest.approx(0.0)

    def test_sparsity_zero_total(self):
        u = SparseUpdate(param_id="p", grad_nnz=0, grad_total=0, applied=False)
        assert u.sparsity == pytest.approx(1.0)

    def test_frozen(self):
        u = SparseUpdate(param_id="p", grad_nnz=1, grad_total=10, applied=True)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            u.applied = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SparseOptimizer.step — dense gradients
# ---------------------------------------------------------------------------

class TestStepDense:
    def _opt(self, lr: float = 0.1, skip: float = 0.0) -> SparseOptimizer:
        return SparseOptimizer(SparseOptimizerConfig(lr=lr, skip_sparse_ratio=skip))

    def test_sgd_applied(self):
        opt = self._opt(lr=0.1)
        params = [1.0, 2.0, 3.0]
        grads = [0.1, 0.2, 0.3]
        new_params, upd = opt.step("p", params, grads)
        assert new_params[0] == pytest.approx(1.0 - 0.1 * 0.1)
        assert new_params[1] == pytest.approx(2.0 - 0.1 * 0.2)
        assert new_params[2] == pytest.approx(3.0 - 0.1 * 0.3)

    def test_applied_flag_true_for_dense(self):
        opt = self._opt()
        _, upd = opt.step("p", [1.0], [0.5])
        assert upd.applied is True

    def test_nnz_count_correct(self):
        opt = self._opt()
        _, upd = opt.step("p", [0.0, 0.0, 0.0], [1.0, 0.0, 2.0])
        assert upd.grad_nnz == 2
        assert upd.grad_total == 3

    def test_lr_scales_update(self):
        params = [5.0]
        grads = [1.0]
        opt_small = self._opt(lr=0.01)
        opt_large = self._opt(lr=1.0)
        new_small, _ = opt_small.step("p", params, grads)
        new_large, _ = opt_large.step("p", params, grads)
        assert abs(5.0 - new_large[0]) > abs(5.0 - new_small[0])

    def test_original_params_not_mutated(self):
        opt = self._opt()
        params = [1.0, 2.0]
        grads = [0.1, 0.1]
        opt.step("p", params, grads)
        assert params == [1.0, 2.0]

    def test_zero_grads_not_updated(self):
        # Use skip_sparse_ratio=1.0 so the update is never skipped (sparsity <= 1.0 always).
        opt = SparseOptimizer(SparseOptimizerConfig(lr=0.1, skip_sparse_ratio=1.0))
        params = [1.0, 2.0, 3.0]
        grads = [0.0, 0.5, 0.0]
        new_params, _ = opt.step("p", params, grads)
        assert new_params[0] == pytest.approx(1.0)  # zero grad → unchanged
        assert new_params[2] == pytest.approx(3.0)
        assert new_params[1] == pytest.approx(2.0 - 0.1 * 0.5)


# ---------------------------------------------------------------------------
# SparseOptimizer.step — skip when sparse
# ---------------------------------------------------------------------------

class TestStepSkip:
    def _opt(self, skip: float = 0.5, threshold: float = 0.0) -> SparseOptimizer:
        return SparseOptimizer(
            SparseOptimizerConfig(lr=0.1, sparsity_threshold=threshold, skip_sparse_ratio=skip)
        )

    def test_skip_when_all_zero(self):
        opt = self._opt(skip=0.0)  # any sparsity > 0 triggers skip
        params = [1.0, 2.0]
        grads = [0.0, 0.0]
        new_params, upd = opt.step("p", params, grads)
        # sparsity=1.0 > skip_sparse_ratio=0.0 → skip
        assert upd.applied is False
        assert new_params == [1.0, 2.0]

    def test_skip_preserves_params(self):
        opt = self._opt(skip=0.3)
        params = [10.0, 20.0, 30.0, 40.0]
        grads = [0.0, 0.0, 0.0, 1.0]  # sparsity = 0.75 > 0.3 → skip
        new_params, upd = opt.step("p", params, grads)
        assert upd.applied is False
        assert new_params == [10.0, 20.0, 30.0, 40.0]

    def test_applied_when_dense_enough(self):
        opt = self._opt(skip=0.9)  # allow up to 90% sparsity
        params = [1.0]
        grads = [0.5]  # sparsity=0.0 ≤ 0.9 → apply
        _, upd = opt.step("p", params, grads)
        assert upd.applied is True

    def test_threshold_affects_nnz(self):
        cfg = SparseOptimizerConfig(lr=0.1, sparsity_threshold=0.5, skip_sparse_ratio=0.0)
        opt = SparseOptimizer(cfg)
        # grads below threshold treated as zero
        params = [1.0, 2.0, 3.0]
        grads = [0.3, 0.3, 0.3]  # all below threshold 0.5 → nnz=0, sparsity=1.0 > 0.0 → skip
        _, upd = opt.step("p", params, grads)
        assert upd.grad_nnz == 0
        assert upd.applied is False


# ---------------------------------------------------------------------------
# SparseOptimizer.stats and reset_stats
# ---------------------------------------------------------------------------

class TestStats:
    def _opt(self) -> SparseOptimizer:
        return SparseOptimizer(SparseOptimizerConfig(lr=0.1, skip_sparse_ratio=0.0))

    def test_initial_stats_zero(self):
        opt = self._opt()
        s = opt.stats()
        assert s["total_updates"] == 0
        assert s["skipped"] == 0
        assert s["applied"] == 0

    def test_total_updates_increments(self):
        opt = self._opt()
        opt.step("p", [1.0], [0.5])
        opt.step("p", [1.0], [0.5])
        assert opt.stats()["total_updates"] == 2

    def test_applied_increments(self):
        opt = self._opt()
        opt.step("p", [1.0], [0.5])
        assert opt.stats()["applied"] == 1
        assert opt.stats()["skipped"] == 0

    def test_skipped_increments(self):
        cfg = SparseOptimizerConfig(lr=0.1, skip_sparse_ratio=0.0)
        opt = SparseOptimizer(cfg)
        opt.step("p", [1.0], [0.0])  # sparsity=1.0 > 0.0 → skip
        assert opt.stats()["skipped"] == 1
        assert opt.stats()["applied"] == 0

    def test_stats_applied_plus_skipped_equals_total(self):
        cfg = SparseOptimizerConfig(lr=0.1, skip_sparse_ratio=0.5)
        opt = SparseOptimizer(cfg)
        opt.step("p", [1.0, 2.0], [1.0, 1.0])  # dense → applied
        opt.step("p", [1.0, 2.0], [0.0, 0.0])  # all zero → skipped
        s = opt.stats()
        assert s["applied"] + s["skipped"] == s["total_updates"]

    def test_reset_stats(self):
        opt = self._opt()
        opt.step("p", [1.0], [0.5])
        opt.step("p", [1.0], [0.5])
        opt.reset_stats()
        s = opt.stats()
        assert s["total_updates"] == 0
        assert s["skipped"] == 0
        assert s["applied"] == 0

    def test_reset_then_continue(self):
        opt = self._opt()
        opt.step("p", [1.0], [0.5])
        opt.reset_stats()
        opt.step("p", [1.0], [0.5])
        assert opt.stats()["total_updates"] == 1


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in SPARSE_OPTIMIZER_REGISTRY

    def test_registry_default_is_class(self):
        assert SPARSE_OPTIMIZER_REGISTRY["default"] is SparseOptimizer

    def test_registry_instantiable(self):
        cls = SPARSE_OPTIMIZER_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, SparseOptimizer)
