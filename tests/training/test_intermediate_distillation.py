"""Tests for intermediate layer knowledge distillation (TinyBERT style)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.intermediate_distillation import (
    IntermDistillConfig,
    LayerMapper,
    AlignmentProjection,
    IntermediateDistillationLoss,
    IntermDistillTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config(**kwargs) -> AureliusConfig:
    """Return a small AureliusConfig for fast tests."""
    defaults = dict(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    defaults.update(kwargs)
    return AureliusConfig(**defaults)


def _make_model(**kwargs) -> AureliusTransformer:
    torch.manual_seed(0)
    return AureliusTransformer(_small_config(**kwargs))


# ---------------------------------------------------------------------------
# 1. LayerMapper — uniform
# ---------------------------------------------------------------------------

def test_layer_mapper_uniform():
    """Uniform mapping for 2 student, 4 teacher layers."""
    mapper = LayerMapper(n_student=2, n_teacher=4, method="uniform")
    mapping = mapper.get_mapping()

    assert len(mapping) == 2, "Should have 2 pairs (one per student layer)"

    student_indices = [s for s, t in mapping]
    teacher_indices = [t for s, t in mapping]

    # Student indices must be 0 and 1
    assert student_indices == [0, 1]

    # All teacher indices must be valid
    for t in teacher_indices:
        assert 0 <= t < 4, f"Teacher index {t} out of range"

    # Uniform: i=0 -> round(0 * 4 / 2) = 0; i=1 -> round(1 * 4 / 2) = 2
    assert mapping[0] == (0, 0)
    assert mapping[1] == (1, 2)


# ---------------------------------------------------------------------------
# 2. LayerMapper — last_n
# ---------------------------------------------------------------------------

def test_layer_mapper_last_n():
    """last_n maps to last n_student teacher layers."""
    mapper = LayerMapper(n_student=2, n_teacher=4, method="last_n")
    mapping = mapper.get_mapping()

    assert len(mapping) == 2
    # last 2 teacher layers are indices 2 and 3
    assert mapping[0] == (0, 2)
    assert mapping[1] == (1, 3)


# ---------------------------------------------------------------------------
# 3. AlignmentProjection — same dim
# ---------------------------------------------------------------------------

def test_alignment_projection_same_dim():
    """d_student == d_teacher -> Identity (no extra params)."""
    proj = AlignmentProjection(d_student=64, d_teacher=64)

    # Should have no learnable parameters
    params = list(proj.parameters())
    assert len(params) == 0, "Same-dim projection should have no parameters"

    x = torch.randn(2, 10, 64)
    out = proj(x)
    assert out.shape == (2, 10, 64)
    assert torch.allclose(out, x)


# ---------------------------------------------------------------------------
# 4. AlignmentProjection — different dim
# ---------------------------------------------------------------------------

def test_alignment_projection_diff_dim():
    """d_student != d_teacher -> has Linear params."""
    proj = AlignmentProjection(d_student=64, d_teacher=128)

    params = list(proj.parameters())
    assert len(params) > 0, "Different-dim projection must have parameters"

    x = torch.randn(2, 10, 64)
    out = proj(x)
    assert out.shape == (2, 10, 128)


# ---------------------------------------------------------------------------
# 5. hidden_loss — same dim
# ---------------------------------------------------------------------------

def test_hidden_loss_same_dim():
    """hidden_loss returns a scalar differentiable tensor."""
    B, T, D = 2, 8, 64
    config = IntermDistillConfig()
    loss_fn = IntermediateDistillationLoss(
        student_d_model=D,
        teacher_d_model=D,
        n_student_layers=2,
        n_teacher_layers=4,
        config=config,
    )

    student_hiddens = [torch.randn(B, T, D, requires_grad=True) for _ in range(2)]
    teacher_hiddens = [torch.randn(B, T, D) for _ in range(4)]

    loss = loss_fn.hidden_loss(student_hiddens, teacher_hiddens)

    assert loss.ndim == 0, "hidden_loss must return a scalar tensor"
    assert loss.requires_grad or loss.is_leaf, "Must be differentiable"

    # Verify backward does not raise
    loss.backward()


# ---------------------------------------------------------------------------
# 6. attention_loss — same heads
# ---------------------------------------------------------------------------

def test_attention_loss_same_heads():
    """attention_loss works with matching head counts and returns scalar tensor."""
    B, H, T = 2, 4, 8
    config = IntermDistillConfig()
    loss_fn = IntermediateDistillationLoss(
        student_d_model=64,
        teacher_d_model=64,
        n_student_layers=2,
        n_teacher_layers=4,
        config=config,
    )

    student_attn = [torch.randn(B, H, T, T, requires_grad=True) for _ in range(2)]
    teacher_attn = [torch.randn(B, H, T, T) for _ in range(4)]

    loss = loss_fn.attention_loss(student_attn, teacher_attn)

    assert loss.ndim == 0, "attention_loss must return a scalar tensor"
    loss.backward()


# ---------------------------------------------------------------------------
# 7. prediction_loss — same vocab
# ---------------------------------------------------------------------------

def test_prediction_loss_same_vocab():
    """KL div works with same vocab size and returns scalar tensor."""
    B, T, V = 2, 8, 256
    config = IntermDistillConfig(temperature=4.0)
    loss_fn = IntermediateDistillationLoss(
        student_d_model=64,
        teacher_d_model=64,
        n_student_layers=2,
        n_teacher_layers=4,
        config=config,
    )

    student_logits = torch.randn(B, T, V, requires_grad=True)
    teacher_logits = torch.randn(B, T, V)

    loss = loss_fn.prediction_loss(student_logits, teacher_logits)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


# ---------------------------------------------------------------------------
# 8. prediction_loss — different vocab
# ---------------------------------------------------------------------------

def test_prediction_loss_diff_vocab():
    """KL div works when student and teacher have different vocab sizes (truncation)."""
    B, T = 2, 8
    V_student, V_teacher = 200, 300
    config = IntermDistillConfig(temperature=2.0)
    loss_fn = IntermediateDistillationLoss(
        student_d_model=64,
        teacher_d_model=64,
        n_student_layers=2,
        n_teacher_layers=4,
        config=config,
    )

    student_logits = torch.randn(B, T, V_student, requires_grad=True)
    teacher_logits = torch.randn(B, T, V_teacher)

    # Should not raise; uses min(V_student, V_teacher) = 200 positions
    loss = loss_fn.prediction_loss(student_logits, teacher_logits)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


# ---------------------------------------------------------------------------
# 9. forward() returns required keys
# ---------------------------------------------------------------------------

def test_distillation_loss_forward_returns_dict():
    """forward() returns dict with 'total', 'pred_loss', 'hidden_loss', 'attn_loss'."""
    B, T, D, V = 2, 8, 64, 256
    config = IntermDistillConfig()
    loss_fn = IntermediateDistillationLoss(
        student_d_model=D,
        teacher_d_model=D,
        n_student_layers=2,
        n_teacher_layers=4,
        config=config,
    )

    student_logits = torch.randn(B, T, V)
    teacher_logits = torch.randn(B, T, V)
    student_hiddens = [torch.randn(B, T, D) for _ in range(2)]
    teacher_hiddens = [torch.randn(B, T, D) for _ in range(4)]
    student_attn = [torch.randn(B, 4, T, T) for _ in range(2)]
    teacher_attn = [torch.randn(B, 4, T, T) for _ in range(4)]

    result = loss_fn(
        student_logits, teacher_logits,
        student_hiddens, teacher_hiddens,
        student_attn, teacher_attn,
    )

    required_keys = {"total", "pred_loss", "hidden_loss", "attn_loss"}
    assert required_keys == set(result.keys()), f"Missing keys: {required_keys - set(result.keys())}"

    for key, val in result.items():
        assert isinstance(val, torch.Tensor), f"'{key}' should be a tensor"
        assert torch.isfinite(val), f"'{key}' should be finite"


# ---------------------------------------------------------------------------
# 10. Total loss = weighted sum
# ---------------------------------------------------------------------------

def test_total_loss_is_weighted_sum():
    """total ~ alpha_pred * pred + alpha_hidden * hidden (attn=0 when not provided)."""
    B, T, D, V = 2, 8, 64, 256
    config = IntermDistillConfig(alpha_pred=0.5, alpha_hidden=0.33, alpha_attn=0.17)
    loss_fn = IntermediateDistillationLoss(
        student_d_model=D,
        teacher_d_model=D,
        n_student_layers=2,
        n_teacher_layers=4,
        config=config,
    )

    student_logits = torch.randn(B, T, V)
    teacher_logits = torch.randn(B, T, V)
    student_hiddens = [torch.randn(B, T, D) for _ in range(2)]
    teacher_hiddens = [torch.randn(B, T, D) for _ in range(4)]

    result = loss_fn(student_logits, teacher_logits, student_hiddens, teacher_hiddens)

    expected = (
        config.alpha_pred * result["pred_loss"]
        + config.alpha_hidden * result["hidden_loss"]
        + config.alpha_attn * result["attn_loss"]
    )

    assert torch.allclose(result["total"], expected, atol=1e-5), (
        f"total={result['total'].item():.6f} != expected={expected.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 11. IntermDistillTrainer.train_step returns metrics
# ---------------------------------------------------------------------------

def test_distill_trainer_step_returns_metrics():
    """train_step returns dict containing the 'loss' key."""
    torch.manual_seed(42)
    student = _make_model()
    teacher = _make_model()

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    config = IntermDistillConfig()

    trainer = IntermDistillTrainer(
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        config=config,
    )

    input_ids = torch.randint(0, 256, (2, 16))
    metrics = trainer.train_step(input_ids)

    assert "loss" in metrics, "train_step must return a dict with 'loss' key"
    assert isinstance(metrics["loss"], float), "'loss' must be a Python float"
    assert torch.isfinite(torch.tensor(metrics["loss"])), "'loss' must be finite"
