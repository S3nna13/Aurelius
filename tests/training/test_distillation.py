"""Tests for knowledge distillation."""

import math

import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.distillation import (
    DistillationConfig,
    DistillationTrainer,
    distillation_loss,
)


def _make_model(n_layers=2):
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=n_layers,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    return AureliusTransformer(cfg)


def test_distillation_loss_shapes():
    """distillation_loss must return 3 scalar tensors."""
    B, S, V = 2, 16, 256
    student_logits = torch.randn(B, S, V)
    teacher_logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))

    total, hard, soft = distillation_loss(student_logits, teacher_logits, labels)

    assert total.ndim == 0
    assert hard.ndim == 0
    assert soft.ndim == 0
    assert torch.isfinite(total)


def test_distillation_loss_alpha_0():
    """With alpha=0, total_loss should equal T^2 * KL divergence (soft only)."""
    B, S, V = 2, 8, 256
    student_logits = torch.randn(B, S, V)
    teacher_logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))

    total, hard, soft = distillation_loss(
        student_logits, teacher_logits, labels, temperature=4.0, alpha=0.0
    )
    assert abs(total.item() - soft.item()) < 1e-5


def test_distillation_loss_alpha_1():
    """With alpha=1, total_loss should equal hard CE loss."""
    B, S, V = 2, 8, 256
    student_logits = torch.randn(B, S, V)
    teacher_logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))

    total, hard, soft = distillation_loss(
        student_logits, teacher_logits, labels, temperature=4.0, alpha=1.0
    )
    assert abs(total.item() - hard.item()) < 1e-5


def test_distillation_loss_identical_teacher():
    """When student == teacher, soft loss should be near 0."""
    B, S, V = 2, 8, 256
    logits = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))

    total, hard, soft = distillation_loss(
        logits, logits.clone(), labels, temperature=4.0, alpha=0.5
    )
    assert soft.item() < 0.01  # soft loss near 0 when student matches teacher


def test_trainer_step_updates_student():
    """DistillationTrainer.step must update student weights."""
    teacher = _make_model(n_layers=2)
    student = _make_model(n_layers=2)

    cfg = DistillationConfig(temperature=4.0, alpha=0.5, learning_rate=1e-3)
    trainer = DistillationTrainer(teacher, student, cfg)

    before = {n: p.clone() for n, p in student.named_parameters()}

    ids = torch.randint(0, 256, (2, 16))
    metrics = trainer.step(ids, ids)

    assert math.isfinite(metrics["total_loss"])
    changed = any(
        not torch.equal(before[n], p) for n, p in student.named_parameters() if p.requires_grad
    )
    assert changed, "Student weights did not update"


def test_teacher_stays_frozen():
    """Teacher parameters must not accumulate gradients during distillation."""
    teacher = _make_model()
    student = _make_model()

    cfg = DistillationConfig()
    trainer = DistillationTrainer(teacher, student, cfg)

    ids = torch.randint(0, 256, (2, 8))
    trainer.step(ids, ids)

    for p in teacher.parameters():
        assert p.grad is None, "Teacher should not have gradients"
