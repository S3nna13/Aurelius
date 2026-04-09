"""Tests for layer-wise knowledge distillation (layer_distillation.py)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import pytest

from src.training.layer_distillation import (
    LayerDistillConfig,
    LinearProjector,
    hidden_state_loss,
    attention_map_loss,
    feature_distribution_loss,
    ActivationHook,
    LayerDistillationLoss,
    LayerDistillTrainer,
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


# ---------------------------------------------------------------------------
# 1. test_layer_distill_config_defaults
# ---------------------------------------------------------------------------

def test_layer_distill_config_defaults():
    cfg = LayerDistillConfig()
    assert cfg.teacher_layers == []
    assert cfg.student_layers == []
    assert cfg.hidden_loss_weight == 1.0
    assert cfg.attention_loss_weight == 0.5
    assert cfg.feature_loss_weight == 0.5
    assert cfg.temperature == 4.0
    assert cfg.loss_type == "mse"
    assert cfg.use_projector is True


# ---------------------------------------------------------------------------
# 2. test_linear_projector_shape
# ---------------------------------------------------------------------------

def test_linear_projector_shape():
    B, T = 2, 8
    student_dim, teacher_dim = 32, 64
    proj = LinearProjector(student_dim, teacher_dim)
    x = torch.randn(B, T, student_dim)
    out = proj(x)
    assert out.shape == (B, T, teacher_dim)


# ---------------------------------------------------------------------------
# 3. test_hidden_state_loss_mse_zero
# ---------------------------------------------------------------------------

def test_hidden_state_loss_mse_zero():
    x = torch.randn(2, 8, 64)
    loss = hidden_state_loss(x, x, loss_type="mse")
    assert loss.item() < 1e-6


# ---------------------------------------------------------------------------
# 4. test_hidden_state_loss_cosine_zero
# ---------------------------------------------------------------------------

def test_hidden_state_loss_cosine_zero():
    x = torch.randn(2, 8, 64)
    loss = hidden_state_loss(x, x, loss_type="cosine")
    assert abs(loss.item()) < 1e-5


# ---------------------------------------------------------------------------
# 5. test_hidden_state_loss_kl_zero
# ---------------------------------------------------------------------------

def test_hidden_state_loss_kl_zero():
    x = torch.randn(2, 8, 64)
    loss = hidden_state_loss(x, x, loss_type="kl")
    assert abs(loss.item()) < 1e-5


# ---------------------------------------------------------------------------
# 6. test_attention_map_loss_identical
# ---------------------------------------------------------------------------

def test_attention_map_loss_identical():
    x = torch.randn(2, 4, 8, 8)
    loss = attention_map_loss(x, x)
    assert abs(loss.item()) < 1e-5


# ---------------------------------------------------------------------------
# 7. test_feature_distribution_loss_identical
# ---------------------------------------------------------------------------

def test_feature_distribution_loss_identical():
    x = torch.randn(2, 8, 64)
    loss = feature_distribution_loss(x, x, temperature=4.0)
    assert abs(loss.item()) < 1e-5


# ---------------------------------------------------------------------------
# 8. test_activation_hook_captures
# ---------------------------------------------------------------------------

def test_activation_hook_captures():
    hook = ActivationHook()
    linear = nn.Linear(16, 16)
    handle = hook.register(linear)

    x = torch.randn(4, 16)
    _ = linear(x)

    assert len(hook.activations) == 1
    assert hook.activations[0].shape == (4, 16)

    hook.clear()
    assert len(hook.activations) == 0

    handle.remove()


# ---------------------------------------------------------------------------
# 9. test_layer_distill_loss_forward_scalar
# ---------------------------------------------------------------------------

def test_layer_distill_loss_forward_scalar():
    cfg = LayerDistillConfig(use_projector=True)
    loss_fn = LayerDistillationLoss(cfg, student_dim=64, teacher_dim=64)

    s_hiddens = [torch.randn(1, 8, 64) for _ in range(2)]
    t_hiddens = [torch.randn(1, 8, 64) for _ in range(2)]

    total_loss, info = loss_fn(s_hiddens, t_hiddens)
    assert total_loss.ndim == 0  # scalar


# ---------------------------------------------------------------------------
# 10. test_layer_distill_loss_forward_info_keys
# ---------------------------------------------------------------------------

def test_layer_distill_loss_forward_info_keys():
    cfg = LayerDistillConfig(use_projector=False)
    loss_fn = LayerDistillationLoss(cfg, student_dim=64, teacher_dim=64)

    s_hiddens = [torch.randn(1, 8, 64) for _ in range(2)]
    t_hiddens = [torch.randn(1, 8, 64) for _ in range(2)]

    _, info = loss_fn(s_hiddens, t_hiddens)
    assert "hidden_loss" in info
    assert "n_pairs" in info
    assert info["n_pairs"] == 2


# ---------------------------------------------------------------------------
# 11. test_layer_distill_trainer_step_keys
# ---------------------------------------------------------------------------

def test_layer_distill_trainer_step_keys():
    torch.manual_seed(0)
    student = _make_model(n_layers=2)
    teacher = _make_model(n_layers=2)
    optimizer = optim.AdamW(student.parameters(), lr=1e-4)
    cfg = LayerDistillConfig()

    trainer = LayerDistillTrainer(student, teacher, cfg, optimizer)
    input_ids = torch.randint(0, 256, (1, 8))

    result = trainer.train_step(input_ids)
    assert "loss" in result
    assert "distill_loss" in result
    assert "ce_loss" in result


# ---------------------------------------------------------------------------
# 12. test_layer_distill_trainer_teacher_frozen
# ---------------------------------------------------------------------------

def test_layer_distill_trainer_teacher_frozen():
    torch.manual_seed(1)
    student = _make_model(n_layers=2)
    teacher = _make_model(n_layers=2)
    optimizer = optim.AdamW(student.parameters(), lr=1e-4)
    cfg = LayerDistillConfig()

    trainer = LayerDistillTrainer(student, teacher, cfg, optimizer)
    input_ids = torch.randint(0, 256, (1, 8))

    trainer.train_step(input_ids)

    for p in teacher.parameters():
        assert p.grad is None, "Teacher parameter should not have gradients"
        assert not p.requires_grad, "Teacher parameter should be frozen"
