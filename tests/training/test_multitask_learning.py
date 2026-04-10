"""Tests for src/training/multitask_learning.py — task routing and heads."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.multitask_learning import (
    TaskConfig,
    TaskHead,
    MultiTaskRouter,
    MultiTaskTrainer,
    compute_multitask_loss,
)

# ---------------------------------------------------------------------------
# Constants / fixtures
# ---------------------------------------------------------------------------

D_MODEL = 64
VOCAB_SIZE = 256
N_CLASSES = 4
BATCH = 2
SEQ_LEN = 8


@pytest.fixture(scope="module")
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=D_MODEL,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def backbone(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def input_ids():
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


@pytest.fixture
def cls_config():
    return TaskConfig(task_name="cls", task_type="classification", n_classes=N_CLASSES, weight=1.0)


@pytest.fixture
def gen_config():
    return TaskConfig(task_name="gen", task_type="generation", n_classes=D_MODEL, weight=1.0)


@pytest.fixture
def reg_config():
    return TaskConfig(task_name="reg", task_type="regression", n_classes=1, weight=1.0)


@pytest.fixture
def trainer(backbone, cls_config, gen_config):
    params = list(backbone.parameters())
    opt = optim.SGD(params, lr=1e-3)
    return MultiTaskTrainer(backbone, [cls_config, gen_config], opt)


# ---------------------------------------------------------------------------
# 1. TaskConfig fields accessible
# ---------------------------------------------------------------------------

def test_task_config_fields(cls_config):
    assert cls_config.task_name == "cls"
    assert cls_config.task_type == "classification"
    assert cls_config.n_classes == N_CLASSES
    assert cls_config.weight == 1.0


# ---------------------------------------------------------------------------
# 2. TaskHead output shape for classification (B, T, n_classes)
# ---------------------------------------------------------------------------

def test_task_head_classification_shape():
    tc = TaskConfig("cls", "classification", n_classes=N_CLASSES)
    head = TaskHead(D_MODEL, tc)
    h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = head(h)
    assert out.shape == (BATCH, SEQ_LEN, N_CLASSES)


# ---------------------------------------------------------------------------
# 3. TaskHead output shape for generation (B, T, d_model)
# ---------------------------------------------------------------------------

def test_task_head_generation_shape():
    tc = TaskConfig("gen", "generation", n_classes=D_MODEL)
    head = TaskHead(D_MODEL, tc)
    h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = head(h)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# 4. TaskHead differentiable (gradient flows through classification head)
# ---------------------------------------------------------------------------

def test_task_head_differentiable():
    tc = TaskConfig("cls", "classification", n_classes=N_CLASSES)
    head = TaskHead(D_MODEL, tc)
    h = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    out = head(h)
    loss = out.sum()
    loss.backward()
    assert h.grad is not None
    assert h.grad.shape == h.shape


# ---------------------------------------------------------------------------
# 5. MultiTaskRouter output shape (B, n_tasks)
# ---------------------------------------------------------------------------

def test_router_output_shape():
    n_tasks = 3
    router = MultiTaskRouter(D_MODEL, n_tasks)
    h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    weights = router(h)
    assert weights.shape == (BATCH, n_tasks)


# ---------------------------------------------------------------------------
# 6. MultiTaskRouter weights sum to 1
# ---------------------------------------------------------------------------

def test_router_weights_sum_to_one():
    router = MultiTaskRouter(D_MODEL, 4)
    h = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    weights = router(h)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)


# ---------------------------------------------------------------------------
# 7. compute_multitask_loss weighted sum correct
# ---------------------------------------------------------------------------

def test_compute_multitask_loss_weighted_sum():
    l1 = torch.tensor(2.0)
    l2 = torch.tensor(3.0)
    losses = {"a": l1, "b": l2}
    weights = {"a": 0.5, "b": 2.0}
    result = compute_multitask_loss(losses, weights)
    expected = 0.5 * 2.0 + 2.0 * 3.0
    assert abs(result.item() - expected) < 1e-5


# ---------------------------------------------------------------------------
# 8. compute_multitask_loss single task equals that task's loss
# ---------------------------------------------------------------------------

def test_compute_multitask_loss_single_task():
    l = torch.tensor(5.0)
    result = compute_multitask_loss({"only": l}, {"only": 1.0})
    assert abs(result.item() - 5.0) < 1e-5


# ---------------------------------------------------------------------------
# 9. MultiTaskTrainer.train_step returns required keys
# ---------------------------------------------------------------------------

def test_train_step_returns_required_keys(trainer, input_ids):
    result = trainer.train_step(input_ids, "cls")
    assert "loss" in result
    assert "task" in result


# ---------------------------------------------------------------------------
# 10. MultiTaskTrainer.train_step task name in result
# ---------------------------------------------------------------------------

def test_train_step_task_name(trainer, input_ids):
    result = trainer.train_step(input_ids, "cls")
    assert result["task"] == "cls"


# ---------------------------------------------------------------------------
# 11. MultiTaskTrainer.get_task_losses returns all task names as keys
# ---------------------------------------------------------------------------

def test_get_task_losses_keys(trainer, input_ids):
    losses = trainer.get_task_losses(input_ids, ["cls", "gen"])
    assert "cls" in losses
    assert "gen" in losses


# ---------------------------------------------------------------------------
# 12. MultiTaskTrainer.get_task_losses values are floats
# ---------------------------------------------------------------------------

def test_get_task_losses_values_are_floats(trainer, input_ids):
    losses = trainer.get_task_losses(input_ids, ["cls", "gen"])
    for v in losses.values():
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# 13. add_task_head increases number of registered tasks
# ---------------------------------------------------------------------------

def test_add_task_head_increases_count(backbone):
    cfg1 = TaskConfig("t1", "classification", n_classes=2)
    cfg2 = TaskConfig("t2", "regression", n_classes=1)
    params = list(backbone.parameters())
    opt = optim.SGD(params, lr=1e-3)
    trainer = MultiTaskTrainer(backbone, [cfg1], opt)
    assert len(trainer.task_heads) == 1
    trainer.add_task_head(cfg2)
    assert len(trainer.task_heads) == 2


# ---------------------------------------------------------------------------
# 14. Different tasks produce different losses
# ---------------------------------------------------------------------------

def test_different_tasks_produce_different_losses(backbone, input_ids):
    cfg_cls = TaskConfig("cls2", "classification", n_classes=N_CLASSES)
    cfg_reg = TaskConfig("reg2", "regression", n_classes=1)
    params = list(backbone.parameters())
    opt = optim.SGD(params, lr=1e-3)
    trainer = MultiTaskTrainer(backbone, [cfg_cls, cfg_reg], opt)
    losses = trainer.get_task_losses(input_ids, ["cls2", "reg2"])
    assert losses["cls2"] != losses["reg2"]
