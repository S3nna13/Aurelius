"""Tests for Multi-Task Learning with Task Embeddings (multi_task_embed.py)."""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.multi_task_embed import (
    GradNormWeighting,
    MultiTaskConfig,
    MultiTaskTrainer,
    TaskEmbedding,
    UncertaintyWeighting,
)

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg():
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
def model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


def make_loader(
    vocab_size: int = 256, seq_len: int = 8, n_samples: int = 4, batch_size: int = 2
) -> DataLoader:
    """Return a DataLoader that yields random integer input_ids."""
    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (n_samples, seq_len))
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# 1. MultiTaskConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = MultiTaskConfig()
    assert cfg.n_tasks == 4
    assert cfg.task_embed_dim == 16
    assert cfg.loss_weighting == "uniform"
    assert cfg.lr == 1e-4


def test_config_custom():
    cfg = MultiTaskConfig(n_tasks=2, task_embed_dim=32, loss_weighting="uncertainty", lr=3e-4)
    assert cfg.n_tasks == 2
    assert cfg.task_embed_dim == 32
    assert cfg.loss_weighting == "uncertainty"
    assert cfg.lr == 3e-4


# ---------------------------------------------------------------------------
# 2. TaskEmbedding — forward shape
# ---------------------------------------------------------------------------


def test_task_embedding_shape_int():
    te = TaskEmbedding(n_tasks=4, embed_dim=16)
    emb = te(0)
    assert emb.shape == (16,), f"Expected (16,), got {emb.shape}"


def test_task_embedding_shape_varies_with_embed_dim():
    te = TaskEmbedding(n_tasks=4, embed_dim=32)
    emb = te(1)
    assert emb.shape == (32,)


# ---------------------------------------------------------------------------
# 3. TaskEmbedding — different tasks → different embeddings
# ---------------------------------------------------------------------------


def test_task_embedding_different_tasks_differ():
    torch.manual_seed(7)
    te = TaskEmbedding(n_tasks=4, embed_dim=16)
    emb0 = te(0)
    emb1 = te(1)
    # Embeddings for distinct tasks should not be identical
    assert not torch.allclose(emb0, emb1), "Embeddings for task 0 and task 1 should differ"


def test_task_embedding_same_task_consistent():
    te = TaskEmbedding(n_tasks=4, embed_dim=16)
    emb_a = te(2)
    emb_b = te(2)
    assert torch.allclose(emb_a, emb_b), "Same task should return same embedding"


# ---------------------------------------------------------------------------
# 4. TaskEmbedding — tensor task_id
# ---------------------------------------------------------------------------


def test_task_embedding_tensor_task_id_scalar():
    te = TaskEmbedding(n_tasks=4, embed_dim=16)
    tid = torch.tensor(2, dtype=torch.long)
    emb = te(tid)
    assert emb.shape == (16,)


def test_task_embedding_tensor_task_id_1d():
    """1-D tensor: trainer may pass batch indices; we take the first element."""
    te = TaskEmbedding(n_tasks=4, embed_dim=16)
    tid = torch.tensor([3], dtype=torch.long)
    emb = te(tid)
    assert emb.shape == (16,)


def test_task_embedding_int_vs_tensor_match():
    """Embedding from int task_id and equivalent long tensor should match."""
    te = TaskEmbedding(n_tasks=4, embed_dim=16)
    emb_int = te(1)
    emb_tensor = te(torch.tensor(1, dtype=torch.long))
    assert torch.allclose(emb_int, emb_tensor)


# ---------------------------------------------------------------------------
# 5. UncertaintyWeighting — output is scalar
# ---------------------------------------------------------------------------


def test_uncertainty_weighting_scalar():
    uw = UncertaintyWeighting(n_tasks=3)
    losses = [torch.tensor(1.0), torch.tensor(0.5), torch.tensor(2.0)]
    total = uw(losses)
    assert total.dim() == 0, f"Expected scalar (dim 0), got dim {total.dim()}"


def test_uncertainty_weighting_scalar_single_task():
    uw = UncertaintyWeighting(n_tasks=1)
    losses = [torch.tensor(0.8)]
    total = uw(losses)
    assert total.dim() == 0


# ---------------------------------------------------------------------------
# 6. UncertaintyWeighting — gradients flow through sigma params
# ---------------------------------------------------------------------------


def test_uncertainty_weighting_grad_flows():
    uw = UncertaintyWeighting(n_tasks=2)
    losses = [torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)]
    total = uw(losses)
    total.backward()
    # log_sigma must have a gradient after backward
    assert uw.log_sigma.grad is not None, "log_sigma should have gradients"
    assert uw.log_sigma.grad.shape == (2,)


def test_uncertainty_weighting_log_sigma_learnable():
    uw = UncertaintyWeighting(n_tasks=4)
    assert uw.log_sigma.requires_grad is True
    assert uw.log_sigma.shape == (4,)


# ---------------------------------------------------------------------------
# 7. GradNormWeighting — update returns positive weights
# ---------------------------------------------------------------------------


def test_gradnorm_update_positive():
    gn = GradNormWeighting(n_tasks=3, alpha=1.5)
    current = [torch.tensor(0.8), torch.tensor(1.2), torch.tensor(0.5)]
    initial = [torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)]
    w = gn.update(current, initial)
    assert (w > 0).all(), "All weights must be positive"


def test_gradnorm_update_shape():
    gn = GradNormWeighting(n_tasks=4, alpha=1.5)
    current = [torch.tensor(float(i + 1)) for i in range(4)]
    initial = [torch.tensor(2.0) for _ in range(4)]
    w = gn.update(current, initial)
    assert w.shape == (4,)


# ---------------------------------------------------------------------------
# 8. GradNormWeighting — weights sum approximately to n_tasks
# ---------------------------------------------------------------------------


def test_gradnorm_weights_sum_to_n_tasks():
    n = 4
    gn = GradNormWeighting(n_tasks=n, alpha=1.5)
    current = [torch.tensor(float(i + 1)) for i in range(n)]
    initial = [torch.tensor(2.0) for _ in range(n)]
    w = gn.update(current, initial)
    assert abs(w.sum().item() - n) < 1.0, f"Weights should sum ≈ {n}, got {w.sum().item():.4f}"


def test_gradnorm_uniform_losses_roughly_uniform():
    """When all tasks decrease at the same rate, weights should be ≈ uniform."""
    n = 3
    gn = GradNormWeighting(n_tasks=n, alpha=1.5)
    current = [torch.tensor(0.5)] * n
    initial = [torch.tensor(1.0)] * n
    w = gn.update(current, initial)
    # Each weight should be ≈ 1.0 (n/n)
    for wi in w:
        assert abs(wi.item() - 1.0) < 0.5, f"Expected ≈1.0 for uniform losses, got {wi.item():.4f}"


# ---------------------------------------------------------------------------
# 9. MultiTaskTrainer.train_step returns per_task_losses dict
# ---------------------------------------------------------------------------


def test_trainer_train_step_per_task_losses(model):
    cfg = MultiTaskConfig(n_tasks=2, loss_weighting="uniform", lr=1e-4)
    loaders = {0: make_loader(), 1: make_loader()}
    trainer = MultiTaskTrainer(model, cfg, loaders)

    result = trainer.train_step()
    assert "per_task_losses" in result
    ptl = result["per_task_losses"]
    assert isinstance(ptl, dict)
    assert 0 in ptl and 1 in ptl


def test_trainer_per_task_losses_are_floats(model):
    cfg = MultiTaskConfig(n_tasks=2, loss_weighting="uniform", lr=1e-4)
    loaders = {0: make_loader(), 1: make_loader()}
    trainer = MultiTaskTrainer(model, cfg, loaders)

    result = trainer.train_step()
    for tid, v in result["per_task_losses"].items():
        assert isinstance(v, float), f"task {tid} loss should be float, got {type(v)}"


# ---------------------------------------------------------------------------
# 10. MultiTaskTrainer — total_loss is scalar
# ---------------------------------------------------------------------------


def test_trainer_total_loss_is_scalar(model):
    cfg = MultiTaskConfig(n_tasks=2, loss_weighting="uniform", lr=1e-4)
    loaders = {0: make_loader(), 1: make_loader()}
    trainer = MultiTaskTrainer(model, cfg, loaders)

    result = trainer.train_step()
    assert "total_loss" in result
    assert isinstance(result["total_loss"], float)


def test_trainer_total_loss_finite(model):
    cfg = MultiTaskConfig(n_tasks=2, loss_weighting="uniform", lr=1e-4)
    loaders = {0: make_loader(), 1: make_loader()}
    trainer = MultiTaskTrainer(model, cfg, loaders)

    result = trainer.train_step()
    assert math.isfinite(result["total_loss"])


def test_trainer_uncertainty_weighting(model):
    cfg = MultiTaskConfig(n_tasks=2, loss_weighting="uncertainty", lr=1e-4)
    loaders = {0: make_loader(), 1: make_loader()}
    trainer = MultiTaskTrainer(model, cfg, loaders)

    assert trainer._uncertainty is not None
    result = trainer.train_step()
    assert math.isfinite(result["total_loss"])


def test_trainer_gradnorm_weighting(model):
    cfg = MultiTaskConfig(n_tasks=2, loss_weighting="gradnorm", lr=1e-4)
    loaders = {0: make_loader(), 1: make_loader()}
    trainer = MultiTaskTrainer(model, cfg, loaders)

    result = trainer.train_step()
    assert math.isfinite(result["total_loss"])


def test_trainer_multiple_steps(model):
    """Trainer should be callable for multiple sequential steps."""
    cfg = MultiTaskConfig(n_tasks=2, loss_weighting="uniform", lr=1e-4)
    loaders = {0: make_loader(), 1: make_loader()}
    trainer = MultiTaskTrainer(model, cfg, loaders)

    for _ in range(3):
        result = trainer.train_step()
        assert math.isfinite(result["total_loss"])
