"""Tests for knowledge_distillation.py — at least 16 tests covering all components."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.knowledge_distillation import (
    DistillationConfig,
    FeatureDistillationLoss,
    DistillationTrainer,
    LayerWiseDistillationTrainer,
    soft_labels,
    kl_distillation_loss,
    mse_distillation_loss,
    distillation_loss,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model():
    torch.manual_seed(42)
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


def _make_optimizer(model):
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )


# ---------------------------------------------------------------------------
# 1. DistillationConfig defaults
# ---------------------------------------------------------------------------

def test_distillation_config_defaults():
    cfg = DistillationConfig()
    assert cfg.temperature == 4.0
    assert cfg.alpha == 0.5
    assert cfg.kd_loss_type == "kl"


# ---------------------------------------------------------------------------
# 2. soft_labels output shape
# ---------------------------------------------------------------------------

def test_soft_labels_shape():
    logits = torch.randn(2, 8, 256)
    out = soft_labels(logits, temperature=4.0)
    assert out.shape == logits.shape


# ---------------------------------------------------------------------------
# 3. soft_labels sums to ~1 along vocab dim
# ---------------------------------------------------------------------------

def test_soft_labels_sums_to_one():
    logits = torch.randn(2, 8, 256)
    out = soft_labels(logits, temperature=4.0)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# 4. soft_labels higher temperature -> softer (lower max prob)
# ---------------------------------------------------------------------------

def test_soft_labels_higher_temp_softer():
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 256) * 5.0
    low_temp = soft_labels(logits, temperature=1.0)
    high_temp = soft_labels(logits, temperature=10.0)
    # Higher temperature means lower maximum probability (flatter distribution)
    assert low_temp.max().item() > high_temp.max().item()


# ---------------------------------------------------------------------------
# 5. kl_distillation_loss returns scalar
# ---------------------------------------------------------------------------

def test_kl_distillation_loss_scalar():
    student = torch.randn(2, 8, 256)
    teacher = torch.randn(2, 8, 256)
    loss = kl_distillation_loss(student, teacher, temperature=4.0)
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# 6. kl_distillation_loss >= 0
# ---------------------------------------------------------------------------

def test_kl_distillation_loss_nonnegative():
    student = torch.randn(2, 8, 256)
    teacher = torch.randn(2, 8, 256)
    loss = kl_distillation_loss(student, teacher, temperature=4.0)
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# 7. kl_distillation_loss = 0 when student == teacher
# ---------------------------------------------------------------------------

def test_kl_distillation_loss_zero_when_equal():
    logits = torch.randn(2, 8, 256)
    loss = kl_distillation_loss(logits, logits.clone(), temperature=4.0)
    assert loss.item() < 1e-5


# ---------------------------------------------------------------------------
# 8. mse_distillation_loss returns scalar
# ---------------------------------------------------------------------------

def test_mse_distillation_loss_scalar():
    student = torch.randn(2, 8, 256)
    teacher = torch.randn(2, 8, 256)
    loss = mse_distillation_loss(student, teacher)
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# 9. mse_distillation_loss = 0 when student == teacher
# ---------------------------------------------------------------------------

def test_mse_distillation_loss_zero_when_equal():
    logits = torch.randn(2, 8, 256)
    loss = mse_distillation_loss(logits, logits.clone())
    assert loss.item() < 1e-6


# ---------------------------------------------------------------------------
# 10. distillation_loss returns (Tensor, dict)
# ---------------------------------------------------------------------------

def test_distillation_loss_return_types():
    B, T, V = 2, 8, 256
    student = torch.randn(B, T, V)
    teacher = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    cfg = DistillationConfig()

    result = distillation_loss(student, teacher, labels, cfg)
    assert isinstance(result, tuple) and len(result) == 2
    total_loss, metrics = result
    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# 11. distillation_loss dict has correct keys
# ---------------------------------------------------------------------------

def test_distillation_loss_dict_keys():
    B, T, V = 2, 8, 256
    student = torch.randn(B, T, V)
    teacher = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    cfg = DistillationConfig()

    _, metrics = distillation_loss(student, teacher, labels, cfg)
    assert "kd_loss" in metrics
    assert "ce_loss" in metrics
    assert "total_loss" in metrics


# ---------------------------------------------------------------------------
# 12. FeatureDistillationLoss with same dims: no projection
# ---------------------------------------------------------------------------

def test_feature_distillation_loss_same_dim():
    fdl = FeatureDistillationLoss(student_dim=64, teacher_dim=64)
    assert fdl.proj is None
    s = torch.randn(2, 8, 64)
    t = torch.randn(2, 8, 64)
    loss = fdl(s, t)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# 13. FeatureDistillationLoss with different dims: projects and returns scalar
# ---------------------------------------------------------------------------

def test_feature_distillation_loss_diff_dim():
    fdl = FeatureDistillationLoss(student_dim=32, teacher_dim=64)
    assert fdl.proj is not None
    s = torch.randn(2, 8, 32)
    t = torch.randn(2, 8, 64)
    loss = fdl(s, t)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# 14. DistillationTrainer.train_step returns dict with correct keys
# ---------------------------------------------------------------------------

def test_distillation_trainer_train_step_keys():
    student = _make_model()
    teacher = _make_model()
    cfg = DistillationConfig()
    optimizer = _make_optimizer(student)

    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        config=cfg,
        optimizer=optimizer,
    )

    input_ids = torch.randint(0, 256, (2, 16))
    metrics = trainer.train_step(input_ids)

    assert "kd_loss" in metrics
    assert "ce_loss" in metrics
    assert "total_loss" in metrics


# ---------------------------------------------------------------------------
# 15. DistillationTrainer teacher params are frozen
# ---------------------------------------------------------------------------

def test_distillation_trainer_teacher_frozen():
    student = _make_model()
    teacher = _make_model()
    cfg = DistillationConfig()
    optimizer = _make_optimizer(student)

    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        config=cfg,
        optimizer=optimizer,
    )

    # All teacher params must not require grad
    for p in teacher.parameters():
        assert not p.requires_grad, "Teacher parameter should not require grad"

    # Run a step and confirm no gradients accumulated on teacher
    input_ids = torch.randint(0, 256, (2, 8))
    trainer.train_step(input_ids)

    for p in teacher.parameters():
        assert p.grad is None, "Teacher should not have gradients after train_step"


# ---------------------------------------------------------------------------
# 16. LayerWiseDistillationTrainer.train_step includes 'feature_loss' key
# ---------------------------------------------------------------------------

def test_layer_wise_trainer_feature_loss_key():
    student = _make_model()
    teacher = _make_model()
    cfg = DistillationConfig()
    optimizer = _make_optimizer(student)

    trainer = LayerWiseDistillationTrainer(
        student=student,
        teacher=teacher,
        config=cfg,
        optimizer=optimizer,
        feature_loss_weight=0.1,
    )

    input_ids = torch.randint(0, 256, (2, 16))
    metrics = trainer.train_step(input_ids)

    assert "feature_loss" in metrics
    assert "kd_loss" in metrics
    assert "ce_loss" in metrics
    assert "total_loss" in metrics


# ---------------------------------------------------------------------------
# 17. distillation_loss mse kd_loss_type works
# ---------------------------------------------------------------------------

def test_distillation_loss_mse_type():
    B, T, V = 2, 8, 256
    student = torch.randn(B, T, V)
    teacher = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    cfg = DistillationConfig(kd_loss_type="mse")

    total_loss, metrics = distillation_loss(student, teacher, labels, cfg)
    assert total_loss.ndim == 0
    assert metrics["kd_loss"] >= 0.0
