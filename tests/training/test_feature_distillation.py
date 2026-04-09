"""Tests for feature-level knowledge distillation module."""
import torch
import torch.nn as nn
import pytest

from src.training.feature_distillation import (
    FeatureDistillConfig,
    hidden_state_mse_loss,
    attention_transfer_loss,
    kd_logit_loss,
    FeatureProjector,
    extract_hidden_states,
    FeatureDistillTrainer,
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


# ---------------------------------------------------------------------------
# Tests: FeatureDistillConfig
# ---------------------------------------------------------------------------

def test_feature_distill_config_defaults():
    cfg = FeatureDistillConfig()
    assert cfg.alpha == 0.5
    assert cfg.temperature == 4.0
    assert cfg.align_layers is None
    assert cfg.attention_transfer is False
    assert cfg.hint_layer_student == 1
    assert cfg.hint_layer_teacher == 2


# ---------------------------------------------------------------------------
# Tests: hidden_state_mse_loss
# ---------------------------------------------------------------------------

def test_hidden_state_mse_loss_scalar_output():
    B, T, D = 2, 8, 64
    student_h = torch.randn(B, T, D)
    teacher_h = torch.randn(B, T, D)
    loss = hidden_state_mse_loss(student_h, teacher_h)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_hidden_state_mse_loss_with_projector_different_dims():
    B, T, D_s, D_t = 2, 8, 32, 64
    student_h = torch.randn(B, T, D_s)
    teacher_h = torch.randn(B, T, D_t)
    projector = nn.Linear(D_s, D_t, bias=False)
    loss = hidden_state_mse_loss(student_h, teacher_h, projector=projector)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_hidden_state_mse_loss_identical_tensors_zero():
    B, T, D = 2, 8, 64
    h = torch.randn(B, T, D)
    loss = hidden_state_mse_loss(h, h)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: attention_transfer_loss
# ---------------------------------------------------------------------------

def test_attention_transfer_loss_scalar_output():
    B, H, T = 2, 4, 8
    student_attn = torch.rand(B, H, T, T)
    teacher_attn = torch.rand(B, H, T, T)
    loss = attention_transfer_loss(student_attn, teacher_attn)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_attention_transfer_loss_identical_zero():
    B, H, T = 2, 4, 8
    attn = torch.rand(B, H, T, T) + 0.1  # avoid all-zero map
    loss = attention_transfer_loss(attn, attn)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: kd_logit_loss
# ---------------------------------------------------------------------------

def test_kd_logit_loss_scalar_output():
    B, S, V = 2, 8, 256
    student_logits = torch.randn(B, S, V)
    teacher_logits = torch.randn(B, S, V)
    loss = kd_logit_loss(student_logits, teacher_logits, temperature=4.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_kd_logit_loss_same_logits_near_zero():
    B, S, V = 2, 8, 256
    logits = torch.randn(B, S, V)
    loss = kd_logit_loss(logits, logits, temperature=4.0)
    # KL divergence of identical distributions is 0
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Tests: FeatureProjector
# ---------------------------------------------------------------------------

def test_feature_projector_output_shape():
    B, T, D_s, D_t = 2, 8, 32, 64
    projector = FeatureProjector(student_dim=D_s, teacher_dim=D_t)
    x = torch.randn(B, T, D_s)
    out = projector(x)
    assert out.shape == (B, T, D_t)


# ---------------------------------------------------------------------------
# Tests: extract_hidden_states
# ---------------------------------------------------------------------------

def test_extract_hidden_states_returns_correct_keys():
    model = _make_model()
    input_ids = torch.randint(0, 256, (1, 16))
    layer_indices = [0, 1]
    with torch.no_grad():
        hidden = extract_hidden_states(model, input_ids, layer_indices)
    assert set(hidden.keys()) == {0, 1}
    for idx, h in hidden.items():
        assert h.ndim == 3  # (B, T, D)
        assert h.shape[0] == 1  # batch size
        assert h.shape[1] == 16  # seq len


# ---------------------------------------------------------------------------
# Tests: FeatureDistillTrainer.train_step
# ---------------------------------------------------------------------------

def test_feature_distill_trainer_train_step_returns_correct_keys():
    student = _make_model()
    teacher = _make_model()
    cfg = FeatureDistillConfig(
        alpha=0.5,
        align_layers=[(0, 0), (1, 1)],
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    trainer = FeatureDistillTrainer(student, teacher, cfg, optimizer)

    input_ids = torch.randint(0, 256, (2, 16))
    labels = torch.randint(0, 256, (2, 16))
    result = trainer.train_step(input_ids, labels)

    assert "loss" in result
    assert "task_loss" in result
    assert "feature_loss" in result


def test_feature_distill_trainer_train_step_loss_is_finite():
    student = _make_model()
    teacher = _make_model()
    cfg = FeatureDistillConfig(
        alpha=0.5,
        align_layers=[(0, 0), (1, 1)],
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    trainer = FeatureDistillTrainer(student, teacher, cfg, optimizer)

    input_ids = torch.randint(0, 256, (2, 16))
    labels = torch.randint(0, 256, (2, 16))
    result = trainer.train_step(input_ids, labels)

    assert torch.isfinite(torch.tensor(result["loss"]))
    assert torch.isfinite(torch.tensor(result["task_loss"]))
    assert torch.isfinite(torch.tensor(result["feature_loss"]))
