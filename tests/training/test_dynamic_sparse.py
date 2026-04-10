"""Tests for dynamic_sparse: RigL-style dynamic sparse training."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.dynamic_sparse import (
    DynamicSparseConfig,
    compute_sparsity,
    create_random_mask,
    apply_mask,
    compute_magnitude_mask,
    compute_gradient_scores,
    rigl_update,
    DynamicSparseTrainer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    """Minimal AureliusTransformer that runs fast on CPU."""
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


@pytest.fixture
def tiny_input():
    torch.manual_seed(7)
    return torch.randint(0, 256, (2, 8))


# ---------------------------------------------------------------------------
# 1. DynamicSparseConfig defaults
# ---------------------------------------------------------------------------

class TestDynamicSparseConfig:
    def test_default_sparsity(self):
        cfg = DynamicSparseConfig()
        assert cfg.sparsity == 0.9

    def test_default_update_interval(self):
        cfg = DynamicSparseConfig()
        assert cfg.update_interval == 100

    def test_default_init_method(self):
        cfg = DynamicSparseConfig()
        assert cfg.init_method == "random"

    def test_default_grow_method(self):
        cfg = DynamicSparseConfig()
        assert cfg.grow_method == "gradient"

    def test_custom_values(self):
        cfg = DynamicSparseConfig(sparsity=0.5, update_interval=50,
                                  init_method="magnitude", grow_method="random")
        assert cfg.sparsity == 0.5
        assert cfg.update_interval == 50
        assert cfg.init_method == "magnitude"
        assert cfg.grow_method == "random"


# ---------------------------------------------------------------------------
# 2. compute_sparsity
# ---------------------------------------------------------------------------

class TestComputeSparsity:
    def test_all_zeros(self):
        t = torch.zeros(100)
        assert compute_sparsity(t) == 1.0

    def test_no_zeros(self):
        t = torch.ones(100)
        assert compute_sparsity(t) == 0.0

    def test_half_zeros(self):
        t = torch.cat([torch.zeros(50), torch.ones(50)])
        assert abs(compute_sparsity(t) - 0.5) < 1e-6

    def test_returns_float(self):
        t = torch.zeros(10)
        assert isinstance(compute_sparsity(t), float)

    def test_2d_tensor(self):
        t = torch.zeros(4, 4)
        t[0, 0] = 1.0
        expected = 15 / 16
        assert abs(compute_sparsity(t) - expected) < 1e-6


# ---------------------------------------------------------------------------
# 3. create_random_mask
# ---------------------------------------------------------------------------

class TestCreateRandomMask:
    def test_shape(self):
        mask = create_random_mask((10, 20), sparsity=0.8)
        assert mask.shape == (10, 20)

    def test_sparsity_accuracy(self):
        torch.manual_seed(42)
        mask = create_random_mask((1000,), sparsity=0.8)
        actual_sparsity = compute_sparsity(mask)
        # Allow ±2 % tolerance for small rounding
        assert abs(actual_sparsity - 0.8) < 0.02

    def test_binary_values(self):
        mask = create_random_mask((50, 50), sparsity=0.5)
        unique = mask.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_at_least_one_active(self):
        mask = create_random_mask((5,), sparsity=0.99)
        assert mask.sum().item() >= 1


# ---------------------------------------------------------------------------
# 4. apply_mask
# ---------------------------------------------------------------------------

class TestApplyMask:
    def test_zeroes_pruned_weights(self):
        model = nn.Linear(8, 8, bias=False)
        nn.init.constant_(model.weight, 1.0)

        # Mask that keeps only top-left quadrant
        mask = torch.zeros(8, 8)
        mask[:4, :4] = 1.0
        masks = {"weight": mask}

        apply_mask(model, masks)
        # Bottom-right quadrant should be zero
        assert model.weight[4:, 4:].sum().item() == 0.0
        # Top-left quadrant should still be 1
        assert model.weight[:4, :4].sum().item() == 16.0

    def test_full_mask_preserves_weights(self):
        model = nn.Linear(4, 4, bias=False)
        nn.init.constant_(model.weight, 2.0)
        mask = torch.ones(4, 4)
        apply_mask(model, {"weight": mask})
        assert (model.weight == 2.0).all()

    def test_zero_mask_zeros_all(self):
        model = nn.Linear(4, 4, bias=False)
        nn.init.constant_(model.weight, 5.0)
        mask = torch.zeros(4, 4)
        apply_mask(model, {"weight": mask})
        assert (model.weight == 0.0).all()


# ---------------------------------------------------------------------------
# 5. compute_magnitude_mask
# ---------------------------------------------------------------------------

class TestComputeMagnitudeMask:
    def test_keeps_largest_values(self):
        # Param with known ordering
        param = nn.Parameter(torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0]))
        mask = compute_magnitude_mask(param, sparsity=0.6)
        # 5 elements * 0.4 active = 2 active; should keep indices 1 (5.0) and 3 (8.0)
        assert mask[1].item() == 1.0
        assert mask[3].item() == 1.0

    def test_correct_active_count(self):
        torch.manual_seed(0)
        param = nn.Parameter(torch.randn(100))
        sparsity = 0.7
        mask = compute_magnitude_mask(param, sparsity=sparsity)
        n_active = int(mask.sum().item())
        expected = round(100 * (1 - sparsity))
        assert n_active == expected

    def test_output_is_binary(self):
        param = nn.Parameter(torch.randn(50))
        mask = compute_magnitude_mask(param, sparsity=0.5)
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_2d_param(self):
        param = nn.Parameter(torch.randn(10, 10))
        mask = compute_magnitude_mask(param, sparsity=0.8)
        assert mask.shape == (10, 10)
        n_active = int(mask.sum().item())
        expected = round(100 * 0.2)
        assert n_active == expected


# ---------------------------------------------------------------------------
# 6. compute_gradient_scores
# ---------------------------------------------------------------------------

class TestComputeGradientScores:
    def test_shape_matches_param(self):
        param = nn.Parameter(torch.randn(8, 16))
        param.grad = torch.randn(8, 16)
        scores = compute_gradient_scores(param)
        assert scores.shape == param.shape

    def test_no_grad_returns_zeros(self):
        param = nn.Parameter(torch.randn(5, 5))
        # grad is None by default
        scores = compute_gradient_scores(param)
        assert (scores == 0).all()

    def test_scores_are_nonnegative(self):
        param = nn.Parameter(torch.randn(10, 10))
        param.grad = torch.randn(10, 10)
        scores = compute_gradient_scores(param)
        assert (scores >= 0).all()

    def test_scores_formula(self):
        # |grad * weight| by element
        param = nn.Parameter(torch.tensor([2.0, -3.0]))
        param.grad = torch.tensor([-1.0, 4.0])
        scores = compute_gradient_scores(param)
        expected = torch.tensor([2.0, 12.0])
        assert torch.allclose(scores, expected)


# ---------------------------------------------------------------------------
# 7. rigl_update — returns dict with same keys
# ---------------------------------------------------------------------------

class TestRigLUpdate:
    @pytest.fixture
    def simple_model_and_masks(self):
        torch.manual_seed(42)
        model = nn.Linear(16, 16, bias=False)
        # Manually set up gradients
        loss = model.weight.sum()
        loss.backward()
        mask = create_random_mask((16, 16), sparsity=0.8)
        masks = {"weight": mask}
        return model, masks

    def test_returns_same_keys(self, simple_model_and_masks):
        model, masks = simple_model_and_masks
        new_masks = rigl_update(model, masks, sparsity=0.8, grow_method="gradient")
        assert set(new_masks.keys()) == set(masks.keys())

    def test_preserves_sparsity_level(self, simple_model_and_masks):
        model, masks = simple_model_and_masks
        sparsity = 0.8
        new_masks = rigl_update(model, masks, sparsity=sparsity, grow_method="gradient")
        for name, mask in new_masks.items():
            actual = compute_sparsity(mask)
            # Allow ±2 % due to edge rounding
            assert abs(actual - sparsity) < 0.02, (
                f"Mask '{name}' sparsity {actual:.3f} deviates from target {sparsity}"
            )

    def test_random_grow_same_keys(self, simple_model_and_masks):
        model, masks = simple_model_and_masks
        new_masks = rigl_update(model, masks, sparsity=0.8, grow_method="random")
        assert set(new_masks.keys()) == set(masks.keys())

    def test_mask_is_binary(self, simple_model_and_masks):
        model, masks = simple_model_and_masks
        new_masks = rigl_update(model, masks, sparsity=0.8, grow_method="gradient")
        for mask in new_masks.values():
            assert set(mask.unique().tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# 8. DynamicSparseTrainer — train_step
# ---------------------------------------------------------------------------

class TestDynamicSparseTrainer:
    @pytest.fixture
    def trainer_and_input(self, tiny_model):
        torch.manual_seed(1)
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=1e-4)
        cfg = DynamicSparseConfig(sparsity=0.5, update_interval=5)
        trainer = DynamicSparseTrainer(tiny_model, cfg, optimizer)
        input_ids = torch.randint(0, 256, (2, 8))
        return trainer, input_ids

    def test_train_step_returns_loss(self, trainer_and_input):
        trainer, input_ids = trainer_and_input
        result = trainer.train_step(input_ids)
        assert "loss" in result
        assert isinstance(result["loss"], float)
        assert result["loss"] > 0

    def test_train_step_returns_sparsity(self, trainer_and_input):
        trainer, input_ids = trainer_and_input
        result = trainer.train_step(input_ids)
        assert "sparsity" in result
        assert 0.0 <= result["sparsity"] <= 1.0

    def test_train_step_returns_step(self, trainer_and_input):
        trainer, input_ids = trainer_and_input
        result = trainer.train_step(input_ids)
        assert "step" in result
        assert isinstance(result["step"], int)

    def test_step_counter_increments(self, trainer_and_input):
        trainer, input_ids = trainer_and_input
        r1 = trainer.train_step(input_ids)
        r2 = trainer.train_step(input_ids)
        assert r2["step"] == r1["step"] + 1

    def test_masks_initialized(self, tiny_model):
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=1e-4)
        cfg = DynamicSparseConfig(sparsity=0.5)
        trainer = DynamicSparseTrainer(tiny_model, cfg, optimizer)
        assert len(trainer.masks) > 0

    def test_rigl_update_fires_at_interval(self, tiny_model):
        """After update_interval steps, masks dict must still be valid."""
        torch.manual_seed(99)
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=1e-4)
        cfg = DynamicSparseConfig(sparsity=0.5, update_interval=3)
        trainer = DynamicSparseTrainer(tiny_model, cfg, optimizer)
        input_ids = torch.randint(0, 256, (2, 8))

        for _ in range(4):
            trainer.train_step(input_ids)

        # Masks should still cover the same parameters
        assert len(trainer.masks) > 0
        for mask in trainer.masks.values():
            assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_magnitude_init_method(self, tiny_model):
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=1e-4)
        cfg = DynamicSparseConfig(sparsity=0.5, init_method="magnitude")
        trainer = DynamicSparseTrainer(tiny_model, cfg, optimizer)
        assert len(trainer.masks) > 0

    def test_random_grow_method(self, tiny_model):
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=1e-4)
        cfg = DynamicSparseConfig(sparsity=0.5, update_interval=2, grow_method="random")
        trainer = DynamicSparseTrainer(tiny_model, cfg, optimizer)
        input_ids = torch.randint(0, 256, (2, 8))
        for _ in range(3):
            result = trainer.train_step(input_ids)
        assert isinstance(result["loss"], float)
