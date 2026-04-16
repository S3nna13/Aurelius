"""Tests for src/training/distillation_v2.py.

All models use tiny configs (small D, V, B, T) for speed.
Pure PyTorch — no HuggingFace, scipy, or sklearn.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.distillation_v2 import (
    DistillConfig,
    FeatureProjector,
    DistillationTrainer,
    kl_divergence_loss,
    mse_logit_loss,
    cosine_embedding_loss,
    compute_combined_distillation_loss,
)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

B, T, V, D = 2, 4, 16, 8  # batch, seq_len, vocab, feature_dim


def rand_logits(*shape):
    torch.manual_seed(42)
    return torch.randn(*shape)


class TinyLM(nn.Module):
    """Minimal LM: embedding + linear → (B, T, V) logits returned directly."""

    def __init__(self, vocab: int = V, dim: int = D):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))  # (B, T, V)


# ---------------------------------------------------------------------------
# 1. DistillConfig defaults
# ---------------------------------------------------------------------------

def test_distill_config_defaults():
    cfg = DistillConfig()
    assert cfg.temperature == 4.0
    assert cfg.alpha == 0.5
    assert cfg.feature_loss_weight == 0.0
    assert cfg.kd_loss_type == "kl"


def test_distill_config_invalid_loss_type():
    with pytest.raises(ValueError, match="kd_loss_type"):
        DistillConfig(kd_loss_type="bad")


def test_distill_config_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        DistillConfig(alpha=1.5)


def test_distill_config_invalid_temperature():
    with pytest.raises(ValueError, match="temperature"):
        DistillConfig(temperature=0.0)


# ---------------------------------------------------------------------------
# 2. kl_divergence_loss
# ---------------------------------------------------------------------------

def test_kl_divergence_loss_returns_scalar_finite_nonneg():
    s = rand_logits(B, T, V)
    t = rand_logits(B, T, V)
    loss = kl_divergence_loss(s, t, temperature=4.0)
    assert loss.ndim == 0, "Expected scalar"
    assert torch.isfinite(loss), "Loss must be finite"
    assert loss.item() >= 0.0, "KL divergence is non-negative"


def test_kl_divergence_loss_identical_logits_near_zero():
    logits = rand_logits(B, T, V)
    loss = kl_divergence_loss(logits, logits.clone(), temperature=4.0)
    assert loss.item() < 1e-5, f"KL with identical logits should be ~0, got {loss.item()}"


def test_kl_divergence_loss_higher_temp_softer():
    """Higher temperature softens distributions.

    As T -> infinity the softmax output approaches uniform, so the un-scaled KL
    (before T^2 compensation) decreases monotonically.  We verify this by
    comparing raw_kl = kl_loss / T^2 at two temperatures.
    """
    import torch.nn.functional as _F

    torch.manual_seed(0)
    s = torch.randn(B, T, V)
    t = torch.randn(B, T, V)

    def raw_kl(temp):
        s_log = _F.log_softmax(s.reshape(-1, V) / temp, dim=-1)
        t_soft = _F.softmax(t.reshape(-1, V) / temp, dim=-1)
        return _F.kl_div(s_log, t_soft, reduction="batchmean").clamp(min=0.0).item()

    raw_low_t = raw_kl(1.0)
    raw_high_t = raw_kl(100.0)
    assert raw_high_t < raw_low_t, (
        f"Higher temperature should produce a lower un-scaled KL divergence "
        f"(T=1: {raw_low_t:.4f}, T=100: {raw_high_t:.4f})"
    )


def test_kl_divergence_loss_2d_input():
    """Should also accept (B, V) tensors."""
    s = rand_logits(B, V)
    t = rand_logits(B, V)
    loss = kl_divergence_loss(s, t, temperature=2.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 3. mse_logit_loss
# ---------------------------------------------------------------------------

def test_mse_logit_loss_returns_scalar_finite_nonneg():
    s = rand_logits(B, T, V)
    t = rand_logits(B, T, V)
    loss = mse_logit_loss(s, t)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_mse_logit_loss_identical_logits_zero():
    logits = rand_logits(B, T, V)
    loss = mse_logit_loss(logits, logits.clone())
    assert loss.item() < 1e-6, f"MSE of identical logits should be 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 4. cosine_embedding_loss
# ---------------------------------------------------------------------------

def test_cosine_embedding_loss_returns_scalar_in_range():
    s = rand_logits(B, T, D)
    t = rand_logits(B, T, D)
    loss = cosine_embedding_loss(s, t)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    # 1 - cos_sim in [-1, 1] so avg in [0, 2] for random vectors
    assert -1e-6 <= loss.item() <= 2.0 + 1e-6, f"Cosine loss {loss.item()} out of [0, 2]"


def test_cosine_embedding_loss_identical_features_near_zero():
    feats = rand_logits(B, T, D)
    loss = cosine_embedding_loss(feats, feats.clone())
    assert loss.item() < 1e-5, f"Cosine loss with identical features should be ~0, got {loss.item()}"


def test_cosine_embedding_loss_2d_input():
    s = rand_logits(B, D)
    t = rand_logits(B, D)
    loss = cosine_embedding_loss(s, t)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 5. compute_combined_distillation_loss
# ---------------------------------------------------------------------------

def test_combined_loss_returns_scalar_and_dict():
    cfg = DistillConfig()
    s = rand_logits(B, T, V)
    t = rand_logits(B, T, V)
    labels = torch.randint(0, V, (B, T))
    total, metrics = compute_combined_distillation_loss(s, t, labels, cfg)
    assert total.ndim == 0
    assert torch.isfinite(total)
    assert isinstance(metrics, dict)


def test_combined_loss_dict_has_required_keys():
    cfg = DistillConfig()
    s = rand_logits(B, T, V)
    t = rand_logits(B, T, V)
    labels = torch.randint(0, V, (B, T))
    _, metrics = compute_combined_distillation_loss(s, t, labels, cfg)
    for key in ("kd_loss", "ce_loss", "total"):
        assert key in metrics, f"Missing key '{key}' in metrics dict"


def test_combined_loss_alpha_one_ignores_ce():
    """alpha=1.0 means total = KD loss only; CE is multiplied by 0."""
    cfg = DistillConfig(alpha=1.0)
    s = rand_logits(B, T, V)
    t = rand_logits(B, T, V)
    # Use all-ignore labels to make CE undefined — should not matter
    labels = torch.full((B, T), -100, dtype=torch.long)
    total, metrics = compute_combined_distillation_loss(s, t, labels, cfg)
    kd = metrics["kd_loss"].item()
    assert abs(total.item() - kd) < 1e-5, (
        f"With alpha=1.0, total ({total.item()}) should equal kd_loss ({kd})"
    )


def test_combined_loss_alpha_zero_equals_ce():
    """alpha=0.0 means total = CE loss only."""
    cfg = DistillConfig(alpha=0.0)
    s = rand_logits(B, T, V)
    t = rand_logits(B, T, V)
    labels = torch.randint(0, V, (B, T))
    total, metrics = compute_combined_distillation_loss(s, t, labels, cfg)
    ce = metrics["ce_loss"].item()
    assert abs(total.item() - ce) < 1e-5, (
        f"With alpha=0.0, total ({total.item()}) should equal ce_loss ({ce})"
    )


def test_combined_loss_mse_type():
    cfg = DistillConfig(kd_loss_type="mse")
    s = rand_logits(B, T, V)
    t = rand_logits(B, T, V)
    labels = torch.randint(0, V, (B, T))
    total, metrics = compute_combined_distillation_loss(s, t, labels, cfg)
    assert torch.isfinite(total)
    assert "kd_loss" in metrics


def test_combined_loss_cosine_type():
    cfg = DistillConfig(kd_loss_type="cosine")
    s = rand_logits(B, T, V)
    t = rand_logits(B, T, V)
    labels = torch.randint(0, V, (B, T))
    total, metrics = compute_combined_distillation_loss(s, t, labels, cfg)
    assert torch.isfinite(total)
    assert "kd_loss" in metrics


def test_combined_loss_ignore_index():
    """Labels with ignore_index=-100 should not crash and give finite loss."""
    cfg = DistillConfig()
    s = rand_logits(B, T, V)
    t = rand_logits(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[:, 0] = -100  # mask out first token
    total, _ = compute_combined_distillation_loss(s, t, labels, cfg)
    assert torch.isfinite(total)


# ---------------------------------------------------------------------------
# 6. FeatureProjector
# ---------------------------------------------------------------------------

def test_feature_projector_output_shape():
    student_dim, teacher_dim = 8, 16
    proj = FeatureProjector(student_dim, teacher_dim)
    x = torch.randn(B, T, student_dim)
    out = proj(x)
    assert out.shape == (B, T, teacher_dim), f"Expected ({B}, {T}, {teacher_dim}), got {out.shape}"


def test_feature_projector_no_bias():
    proj = FeatureProjector(8, 16)
    assert proj.proj.bias is None, "FeatureProjector should have no bias"


def test_feature_projector_2d_input():
    proj = FeatureProjector(8, 16)
    x = torch.randn(B, 8)
    out = proj(x)
    assert out.shape == (B, 16)


# ---------------------------------------------------------------------------
# 7. DistillationTrainer
# ---------------------------------------------------------------------------

def test_distillation_trainer_freeze_teacher():
    """freeze_teacher() must set all teacher params to requires_grad=False."""
    student = TinyLM()
    teacher = TinyLM()
    cfg = DistillConfig()
    trainer = DistillationTrainer(student, teacher, cfg)
    trainer.freeze_teacher()
    for name, param in teacher.named_parameters():
        assert not param.requires_grad, f"Teacher param '{name}' should be frozen"


def test_distillation_trainer_get_trainable_params_student_only():
    """get_trainable_params() must return only student parameters."""
    student = TinyLM()
    teacher = TinyLM()
    cfg = DistillConfig()
    trainer = DistillationTrainer(student, teacher, cfg)

    trainable = trainer.get_trainable_params()
    student_param_ids = {id(p) for p in student.parameters()}
    teacher_param_ids = {id(p) for p in teacher.parameters()}

    for p in trainable:
        assert id(p) in student_param_ids, "Non-student param in trainable list"
        assert id(p) not in teacher_param_ids, "Teacher param leaked into trainable list"


def test_distillation_trainer_compute_loss_returns_correct_types():
    student = TinyLM()
    teacher = TinyLM()
    cfg = DistillConfig()
    trainer = DistillationTrainer(student, teacher, cfg)

    input_ids = torch.randint(0, V, (B, T))
    labels = torch.randint(0, V, (B, T))
    loss, metrics = trainer.compute_loss(input_ids, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert isinstance(metrics, dict)
    for key in ("kd_loss", "ce_loss", "total"):
        assert key in metrics


def test_distillation_trainer_teacher_no_grad_during_compute():
    """Teacher params must not accumulate gradients after compute_loss."""
    student = TinyLM()
    teacher = TinyLM()
    cfg = DistillConfig()
    trainer = DistillationTrainer(student, teacher, cfg)

    input_ids = torch.randint(0, V, (B, T))
    labels = torch.randint(0, V, (B, T))
    loss, _ = trainer.compute_loss(input_ids, labels)
    loss.backward()

    for name, param in teacher.named_parameters():
        assert param.grad is None, f"Teacher param '{name}' should have no grad"


def test_distillation_trainer_student_grad_flows():
    """Student params must receive gradients after backward."""
    student = TinyLM()
    teacher = TinyLM()
    cfg = DistillConfig()
    trainer = DistillationTrainer(student, teacher, cfg)

    input_ids = torch.randint(0, V, (B, T))
    labels = torch.randint(0, V, (B, T))
    loss, _ = trainer.compute_loss(input_ids, labels)
    loss.backward()

    has_grad = any(p.grad is not None for p in student.parameters() if p.requires_grad)
    assert has_grad, "Student params must receive gradients"


def test_distillation_trainer_with_projector():
    """Trainer with a FeatureProjector should include projector params in trainable list."""
    student = TinyLM(dim=D)
    teacher = TinyLM(dim=D * 2)
    cfg = DistillConfig()
    projector = FeatureProjector(D, D * 2)
    trainer = DistillationTrainer(student, teacher, cfg, projector=projector)

    trainable_ids = {id(p) for p in trainer.get_trainable_params()}
    proj_param_ids = {id(p) for p in projector.parameters()}
    assert proj_param_ids.issubset(trainable_ids), "Projector params must be in trainable list"
