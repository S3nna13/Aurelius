"""Tests for the multi-task learning framework."""

import math

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.multitask import (
    DynamicTemperatureBalancing,
    MultitaskConfig,
    MultitaskModel,
    MultitaskTrainer,
    TaskHead,
    UncertaintyWeighting,
    project_conflicting_gradients,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
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


@pytest.fixture
def backbone(small_config):
    torch.manual_seed(42)
    return AureliusTransformer(small_config)


@pytest.fixture
def task_heads(small_config):
    """Two simple task heads: classification (10 classes) and regression (scalar)."""
    cls_head = TaskHead(
        name="classification",
        head=nn.Linear(small_config.vocab_size, 10),
        loss_fn=nn.CrossEntropyLoss(),
        weight=1.0,
    )
    reg_head = TaskHead(
        name="regression",
        head=nn.Linear(small_config.vocab_size, 1),
        loss_fn=nn.MSELoss(),
        weight=1.0,
    )
    return [cls_head, reg_head]


@pytest.fixture
def multitask_model(backbone, task_heads):
    config = MultitaskConfig()
    return MultitaskModel(backbone, task_heads, config)


@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, 256, (2, 16))


@pytest.fixture
def targets():
    torch.manual_seed(0)
    return {
        "classification": torch.randint(0, 10, (2,)),
        "regression": torch.randn(2, 1),
    }


# ---------------------------------------------------------------------------
# 1. MultitaskConfig defaults
# ---------------------------------------------------------------------------


def test_multitask_config_defaults():
    cfg = MultitaskConfig()
    assert cfg.balancing_strategy == "static"
    assert cfg.gradient_surgery is False
    assert cfg.temperature_lr == 0.01
    assert cfg.max_grad_norm == 1.0


# ---------------------------------------------------------------------------
# 2. project_conflicting_gradients — aligned gradients unchanged
# ---------------------------------------------------------------------------


def test_pcgrad_aligned_gradients():
    g1 = torch.tensor([1.0, 2.0, 3.0])
    g2 = torch.tensor([2.0, 4.0, 6.0])  # parallel to g1
    result = project_conflicting_gradients([g1, g2])
    # Aligned gradients should not be modified
    assert torch.allclose(result[0], g1, atol=1e-5)
    assert torch.allclose(result[1], g2, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. project_conflicting_gradients — opposing gradients reduce conflict
# ---------------------------------------------------------------------------


def test_pcgrad_opposing_gradients():
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([-1.0, 0.0])  # directly opposing
    result = project_conflicting_gradients([g1, g2])
    # After projection, g1 should have the conflicting component removed
    # g1 - (g1.g2 / ||g2||^2) * g2 = [1,0] - (-1/1)*[-1,0] = [1,0]+[-1,0] = [0,0]
    assert torch.allclose(result[0], torch.tensor([0.0, 0.0]), atol=1e-5)
    assert torch.allclose(result[1], torch.tensor([0.0, 0.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# 4. UncertaintyWeighting output is scalar
# ---------------------------------------------------------------------------


def test_uncertainty_weighting_scalar():
    uw = UncertaintyWeighting(n_tasks=3)
    losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.5)]
    total = uw(losses)
    assert total.dim() == 0  # scalar


# ---------------------------------------------------------------------------
# 5. UncertaintyWeighting log_vars are learnable
# ---------------------------------------------------------------------------


def test_uncertainty_weighting_learnable():
    uw = UncertaintyWeighting(n_tasks=2)
    assert uw.log_vars.requires_grad is True
    assert uw.log_vars.shape == (2,)


# ---------------------------------------------------------------------------
# 6. DynamicTemperatureBalancing output is scalar
# ---------------------------------------------------------------------------


def test_dynamic_temperature_scalar():
    dtb = DynamicTemperatureBalancing(n_tasks=3)
    losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.5)]
    total = dtb(losses)
    assert total.dim() == 0  # scalar


# ---------------------------------------------------------------------------
# 7. DynamicTemperatureBalancing temperatures are learnable
# ---------------------------------------------------------------------------


def test_dynamic_temperature_learnable():
    dtb = DynamicTemperatureBalancing(n_tasks=2)
    assert dtb.temperatures.requires_grad is True
    assert dtb.temperatures.shape == (2,)


# ---------------------------------------------------------------------------
# 8. MultitaskModel forward returns tensor for each task
# ---------------------------------------------------------------------------


def test_multitask_model_forward(multitask_model, input_ids):
    cls_out = multitask_model(input_ids, "classification")
    reg_out = multitask_model(input_ids, "regression")
    assert cls_out.shape == (2, 10)  # batch=2, num_classes=10
    assert reg_out.shape == (2, 1)  # batch=2, scalar output


# ---------------------------------------------------------------------------
# 9. MultitaskModel raises on unknown task name
# ---------------------------------------------------------------------------


def test_multitask_model_unknown_task(multitask_model, input_ids):
    with pytest.raises(ValueError, match="Unknown task"):
        multitask_model(input_ids, "nonexistent_task")


# ---------------------------------------------------------------------------
# 10. MultitaskTrainer compute_all_losses returns loss per task
# ---------------------------------------------------------------------------


def test_trainer_compute_all_losses(multitask_model, input_ids, targets):
    config = MultitaskConfig()
    optimizer = torch.optim.Adam(multitask_model.parameters(), lr=1e-3)
    trainer = MultitaskTrainer(multitask_model, optimizer, config)

    losses = trainer.compute_all_losses(input_ids, targets)
    assert "classification" in losses
    assert "regression" in losses
    assert losses["classification"].dim() == 0
    assert losses["regression"].dim() == 0


# ---------------------------------------------------------------------------
# 11. MultitaskTrainer train_step returns correct keys
# ---------------------------------------------------------------------------


def test_trainer_train_step_keys(multitask_model, input_ids, targets):
    config = MultitaskConfig()
    optimizer = torch.optim.Adam(multitask_model.parameters(), lr=1e-3)
    trainer = MultitaskTrainer(multitask_model, optimizer, config)

    result = trainer.train_step(input_ids, targets)
    assert "total_loss" in result
    assert "classification_loss" in result
    assert "regression_loss" in result


# ---------------------------------------------------------------------------
# 12. MultitaskTrainer train_step loss is finite
# ---------------------------------------------------------------------------


def test_trainer_train_step_finite(multitask_model, input_ids, targets):
    config = MultitaskConfig()
    optimizer = torch.optim.Adam(multitask_model.parameters(), lr=1e-3)
    trainer = MultitaskTrainer(multitask_model, optimizer, config)

    result = trainer.train_step(input_ids, targets)
    assert math.isfinite(result["total_loss"])
    assert math.isfinite(result["classification_loss"])
    assert math.isfinite(result["regression_loss"])


# ---------------------------------------------------------------------------
# 13. MultitaskTrainer with uncertainty balancing includes uncertainty params
# ---------------------------------------------------------------------------


def test_trainer_uncertainty_balancing(backbone, task_heads, input_ids, targets):
    config = MultitaskConfig(balancing_strategy="uncertainty")
    model = MultitaskModel(backbone, task_heads, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = MultitaskTrainer(model, optimizer, config)

    # Balancer should exist and have learnable log_vars
    assert trainer.balancer is not None
    assert isinstance(trainer.balancer, UncertaintyWeighting)
    assert trainer.balancer.log_vars.requires_grad is True

    # Train step should still work
    result = trainer.train_step(input_ids, targets)
    assert math.isfinite(result["total_loss"])
