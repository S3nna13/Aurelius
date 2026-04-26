"""Tests for layer-wise knowledge distillation (layer_distillation.py)."""

from __future__ import annotations

import torch
import torch.optim as optim

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.layer_distillation import (
    HiddenStateProjector,
    LayerDistillConfig,
    LayerDistillTrainer,
    extract_layer_hidden_states,
    hidden_state_alignment_loss,
    patient_kd_loss,
    relation_based_loss,
    soft_label_loss,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


def _make_model() -> AureliusTransformer:
    torch.manual_seed(42)
    return AureliusTransformer(TINY_CFG)


def _make_trainer(same_model: bool = True):
    """Return (trainer, teacher, student) with shared or different weights."""
    torch.manual_seed(0)
    teacher = _make_model()
    student = _make_model()
    optimizer = optim.AdamW(student.parameters(), lr=1e-4)
    cfg = LayerDistillConfig(
        teacher_layers=[0, 1],
        student_layers=[0, 1],
        hidden_dim_teacher=64,
        hidden_dim_student=64,
    )
    trainer = LayerDistillTrainer(teacher, student, optimizer, cfg)
    return trainer, teacher, student


# ---------------------------------------------------------------------------
# 1. test_config_loss_weights
# ---------------------------------------------------------------------------


def test_config_loss_weights():
    cfg = LayerDistillConfig(teacher_layers=[0], student_layers=[0])
    assert "task" in cfg.loss_weights
    assert "hidden" in cfg.loss_weights
    assert isinstance(cfg.loss_weights["task"], float)
    assert isinstance(cfg.loss_weights["hidden"], float)


# ---------------------------------------------------------------------------
# 2. test_extract_hidden_states_count
# ---------------------------------------------------------------------------


def test_extract_hidden_states_count():
    model = _make_model()
    input_ids = torch.randint(0, 256, (2, 8))
    layer_indices = [0, 1]
    hiddens = extract_layer_hidden_states(model, input_ids, layer_indices)
    assert len(hiddens) == len(layer_indices)


# ---------------------------------------------------------------------------
# 3. test_extract_hidden_states_shape
# ---------------------------------------------------------------------------


def test_extract_hidden_states_shape():
    model = _make_model()
    B, T = 2, 8
    input_ids = torch.randint(0, 256, (B, T))
    layer_indices = [0, 1]
    hiddens = extract_layer_hidden_states(model, input_ids, layer_indices)
    for h in hiddens:
        assert h.ndim == 3
        assert h.shape[0] == B
        assert h.shape[1] == T
        assert h.shape[2] == 64  # d_model


# ---------------------------------------------------------------------------
# 4. test_relation_based_loss_scalar
# ---------------------------------------------------------------------------


def test_relation_based_loss_scalar():
    t_h = torch.randn(4, 8, 64)
    s_h = torch.randn(4, 8, 64)
    loss = relation_based_loss(t_h, s_h)
    assert loss.ndim == 0  # scalar


# ---------------------------------------------------------------------------
# 5. test_relation_based_loss_identical
# ---------------------------------------------------------------------------


def test_relation_based_loss_identical():
    x = torch.randn(4, 8, 64)
    loss = relation_based_loss(x, x)
    assert loss.item() < 1e-5


# ---------------------------------------------------------------------------
# 6. test_hidden_alignment_loss_scalar
# ---------------------------------------------------------------------------


def test_hidden_alignment_loss_scalar():
    t_h = torch.randn(2, 8, 64)
    s_h = torch.randn(2, 8, 64)
    loss = hidden_state_alignment_loss(t_h, s_h)
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# 7. test_hidden_alignment_loss_identical
# ---------------------------------------------------------------------------


def test_hidden_alignment_loss_identical():
    x = torch.randn(2, 8, 64)
    loss = hidden_state_alignment_loss(x, x)
    assert loss.item() < 1e-6


# ---------------------------------------------------------------------------
# 8. test_soft_label_loss_scalar
# ---------------------------------------------------------------------------


def test_soft_label_loss_scalar():
    student_logits = torch.randn(16, 256)
    teacher_logits = torch.randn(16, 256)
    loss = soft_label_loss(student_logits, teacher_logits, temperature=4.0)
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# 9. test_soft_label_loss_identical
# ---------------------------------------------------------------------------


def test_soft_label_loss_identical():
    logits = torch.randn(16, 256)
    loss = soft_label_loss(logits, logits, temperature=4.0)
    assert loss.item() < 1e-5


# ---------------------------------------------------------------------------
# 10. test_patient_kd_loss_scalar
# ---------------------------------------------------------------------------


def test_patient_kd_loss_scalar():
    teacher_hiddens = [torch.randn(2, 8, 64) for _ in range(2)]
    student_hiddens = [torch.randn(2, 8, 64) for _ in range(2)]
    loss = patient_kd_loss(teacher_hiddens, student_hiddens)
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# 11. test_hidden_state_projector_shape
# ---------------------------------------------------------------------------


def test_hidden_state_projector_shape():
    B, T = 2, 8
    d_student, d_teacher = 32, 64
    proj = HiddenStateProjector(d_student, d_teacher)
    x = torch.randn(B, T, d_student)
    out = proj(x)
    assert out.shape == (B, T, d_teacher)


# ---------------------------------------------------------------------------
# 12. test_layer_distill_trainer_step_keys
# ---------------------------------------------------------------------------


def test_layer_distill_trainer_step_keys():
    trainer, _, _ = _make_trainer()
    input_ids = torch.randint(0, 256, (2, 8))
    labels = torch.randint(0, 256, (2, 8))
    result = trainer.train_step(input_ids, labels)
    assert "total_loss" in result
    assert "task_loss" in result
    assert "hidden_loss" in result
    assert "relation_loss" in result


# ---------------------------------------------------------------------------
# 13. test_layer_distill_trainer_loss_positive
# ---------------------------------------------------------------------------


def test_layer_distill_trainer_loss_positive():
    trainer, _, _ = _make_trainer()
    input_ids = torch.randint(0, 256, (2, 8))
    labels = torch.randint(0, 256, (2, 8))
    result = trainer.train_step(input_ids, labels)
    assert result["total_loss"] > 0.0


# ---------------------------------------------------------------------------
# 14. test_evaluate_alignment_keys
# ---------------------------------------------------------------------------


def test_evaluate_alignment_keys():
    trainer, _, _ = _make_trainer()
    input_ids = torch.randint(0, 256, (2, 8))
    result = trainer.evaluate_layer_alignment(input_ids)
    assert "mean_alignment" in result


# ---------------------------------------------------------------------------
# 15. test_evaluate_alignment_range
# ---------------------------------------------------------------------------


def test_evaluate_alignment_range():
    trainer, _, _ = _make_trainer()
    input_ids = torch.randint(0, 256, (2, 8))
    result = trainer.evaluate_layer_alignment(input_ids)
    mean_align = result["mean_alignment"]
    assert 0.0 <= mean_align <= 1.0, f"mean_alignment={mean_align} out of [0, 1]"
