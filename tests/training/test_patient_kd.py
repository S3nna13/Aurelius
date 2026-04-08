"""Tests for Patient Knowledge Distillation (patient_kd.py)."""
import torch
import torch.nn as nn
import torch.optim as optim
import pytest

from src.training.patient_kd import (
    PKDConfig,
    get_patient_layers,
    pkd_hidden_loss,
    attention_transfer_loss,
    PKDLoss,
    PatientKDTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(n_layers: int = 2) -> AureliusTransformer:
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=n_layers,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


def _make_hiddens(n: int, B: int = 1, T: int = 8, D: int = 64) -> list[torch.Tensor]:
    return [torch.randn(B, T, D) for _ in range(n)]


def _make_attns(n: int, B: int = 1, H: int = 2, T: int = 8) -> list[torch.Tensor]:
    return [torch.randn(B, H, T, T) for _ in range(n)]


# ---------------------------------------------------------------------------
# 1. test_pkd_config_defaults
# ---------------------------------------------------------------------------

def test_pkd_config_defaults():
    cfg = PKDConfig()
    assert cfg.n_student_layers == 2
    assert cfg.n_teacher_layers == 4
    assert cfg.patience_strategy == "last"
    assert cfg.beta == 500.0
    assert cfg.temperature == 4.0
    assert cfg.normalize_hidden is True


# ---------------------------------------------------------------------------
# 2. test_get_patient_layers_last
# ---------------------------------------------------------------------------

def test_get_patient_layers_last():
    # n_teacher=4, n_student=2, strategy="last" -> [2, 3]
    result = get_patient_layers(4, 2, "last")
    assert result == [2, 3], f"Expected [2, 3], got {result}"
    assert len(result) == 2


# ---------------------------------------------------------------------------
# 3. test_get_patient_layers_skip
# ---------------------------------------------------------------------------

def test_get_patient_layers_skip():
    n_student = 3
    result = get_patient_layers(6, n_student, "skip")
    assert len(result) == n_student
    # All indices must be valid (< n_teacher)
    for idx in result:
        assert 0 <= idx < 6


# ---------------------------------------------------------------------------
# 4. test_pkd_hidden_loss_same
# ---------------------------------------------------------------------------

def test_pkd_hidden_loss_same():
    hiddens = _make_hiddens(2)
    # Same tensors: MSE should be 0
    loss = pkd_hidden_loss(hiddens, hiddens, [0, 1], normalize=False)
    assert loss.item() < 1e-6


# ---------------------------------------------------------------------------
# 5. test_pkd_hidden_loss_different
# ---------------------------------------------------------------------------

def test_pkd_hidden_loss_different():
    s = _make_hiddens(2)
    t = _make_hiddens(4)
    loss = pkd_hidden_loss(s, t, [0, 1], normalize=False)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# 6. test_pkd_hidden_loss_normalized
# ---------------------------------------------------------------------------

def test_pkd_hidden_loss_normalized():
    s = _make_hiddens(2)
    t = _make_hiddens(4)
    loss = pkd_hidden_loss(s, t, [0, 1], normalize=True)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# 7. test_attention_transfer_loss_scalar
# ---------------------------------------------------------------------------

def test_attention_transfer_loss_scalar():
    s_attns = _make_attns(2)
    t_attns = _make_attns(4)
    loss = attention_transfer_loss(s_attns, t_attns, [0, 1])
    assert loss.ndim == 1 or loss.ndim == 0  # scalar or 1-element
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# 8. test_pkd_loss_forward_keys
# ---------------------------------------------------------------------------

def test_pkd_loss_forward_keys():
    cfg = PKDConfig(n_student_layers=2, n_teacher_layers=4)
    loss_fn = PKDLoss(cfg, student_d_model=64, teacher_d_model=64)

    B, T, V = 1, 8, 256
    s_hiddens = _make_hiddens(2, B=B, T=T, D=64)
    t_hiddens = _make_hiddens(4, B=B, T=T, D=64)
    s_logits = torch.randn(B, T, V)
    t_logits = torch.randn(B, T, V)
    labels = torch.zeros(B, T, dtype=torch.long)

    total, metrics = loss_fn(s_hiddens, t_hiddens, s_logits, t_logits, labels)
    assert "soft_kl" in metrics
    assert "hidden_loss" in metrics
    assert "total" in metrics


# ---------------------------------------------------------------------------
# 9. test_pkd_loss_forward_positive
# ---------------------------------------------------------------------------

def test_pkd_loss_forward_positive():
    cfg = PKDConfig(n_student_layers=2, n_teacher_layers=4)
    loss_fn = PKDLoss(cfg, student_d_model=64, teacher_d_model=64)

    B, T, V = 1, 8, 256
    s_hiddens = _make_hiddens(2, B=B, T=T, D=64)
    t_hiddens = _make_hiddens(4, B=B, T=T, D=64)
    s_logits = torch.randn(B, T, V)
    t_logits = torch.randn(B, T, V)
    labels = torch.zeros(B, T, dtype=torch.long)

    total, metrics = loss_fn(s_hiddens, t_hiddens, s_logits, t_logits, labels)
    assert total.item() > 0.0
    assert torch.isfinite(total)


# ---------------------------------------------------------------------------
# 10. test_patient_kd_trainer_step_keys
# ---------------------------------------------------------------------------

def test_patient_kd_trainer_step_keys():
    torch.manual_seed(0)
    student = _make_model(n_layers=2)
    teacher = _make_model(n_layers=2)
    optimizer = optim.AdamW(student.parameters(), lr=1e-4)
    cfg = PKDConfig(n_student_layers=2, n_teacher_layers=2, patience_strategy="last")

    trainer = PatientKDTrainer(student, teacher, optimizer, cfg)
    input_ids = torch.randint(0, 256, (1, 8))
    labels = torch.randint(0, 256, (1, 8))

    result = trainer.train_step(input_ids, labels)
    assert "loss" in result
    assert "soft_kl" in result
    assert "hidden_loss" in result


# ---------------------------------------------------------------------------
# 11. test_patient_kd_teacher_frozen
# ---------------------------------------------------------------------------

def test_patient_kd_teacher_frozen():
    torch.manual_seed(1)
    student = _make_model(n_layers=2)
    teacher = _make_model(n_layers=2)
    optimizer = optim.AdamW(student.parameters(), lr=1e-4)
    cfg = PKDConfig(n_student_layers=2, n_teacher_layers=2, patience_strategy="last")

    trainer = PatientKDTrainer(student, teacher, optimizer, cfg)
    input_ids = torch.randint(0, 256, (1, 8))
    labels = torch.randint(0, 256, (1, 8))

    trainer.train_step(input_ids, labels)

    for p in teacher.parameters():
        assert p.grad is None, "Teacher parameter should not have gradients"
        assert not p.requires_grad, "Teacher parameter should be frozen"
