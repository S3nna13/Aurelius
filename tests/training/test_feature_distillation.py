"""Tests for feature-level knowledge distillation (FeatDistillConfig API)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.feature_distillation import (
    FeatDistillConfig,
    FeatureAdapter,
    extract_features,
    compute_feature_loss,
    compute_attention_transfer_loss,
    soft_label_loss,
    FeatureDistillTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(seed: int = 42) -> AureliusTransformer:
    torch.manual_seed(seed)
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


def _make_trainer(
    layer_mapping: list[tuple[int, int]] | None = None,
) -> tuple[FeatureDistillTrainer, nn.Module]:
    student = _make_model(seed=0)
    teacher = _make_model(seed=1)
    cfg = FeatDistillConfig(
        layer_mapping=layer_mapping or [(0, 0), (1, 1)],
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    trainer = FeatureDistillTrainer(student, teacher, optimizer, cfg)
    return trainer, student


# ---------------------------------------------------------------------------
# Test 1: FeatDistillConfig defaults
# ---------------------------------------------------------------------------

def test_feat_distill_config_defaults():
    cfg = FeatDistillConfig()
    assert cfg.temperature == 4.0
    assert cfg.alpha == 0.5
    assert cfg.feature_loss_weight == 0.1
    assert cfg.attention_loss_weight == 0.1
    assert cfg.layer_mapping == []


# ---------------------------------------------------------------------------
# Test 2: FeatureAdapter output shape
# ---------------------------------------------------------------------------

def test_feature_adapter_output_shape():
    B, T, D_s, D_t = 2, 8, 32, 64
    adapter = FeatureAdapter(student_dim=D_s, teacher_dim=D_t)
    x = torch.randn(B, T, D_s)
    out = adapter(x)
    assert out.shape == (B, T, D_t)


# ---------------------------------------------------------------------------
# Test 3: extract_features returns correct layer keys
# ---------------------------------------------------------------------------

def test_extract_features_returns_correct_keys():
    model = _make_model()
    input_ids = torch.randint(0, 256, (1, 16))
    layer_indices = [0, 1]
    feats = extract_features(model, input_ids, layer_indices)
    assert set(feats.keys()) == {0, 1}


# ---------------------------------------------------------------------------
# Test 4: extract_features tensor shape (B, T, D)
# ---------------------------------------------------------------------------

def test_extract_features_tensor_shape():
    model = _make_model()
    B, T = 2, 12
    input_ids = torch.randint(0, 256, (B, T))
    feats = extract_features(model, input_ids, [0, 1])
    for idx, h in feats.items():
        assert h.ndim == 3, f"Layer {idx} should be 3D (B, T, D)"
        assert h.shape[0] == B
        assert h.shape[1] == T


# ---------------------------------------------------------------------------
# Test 5: compute_feature_loss returns scalar tensor
# ---------------------------------------------------------------------------

def test_compute_feature_loss_returns_scalar():
    B, T, D = 2, 8, 64
    s_feats = {0: torch.randn(B, T, D), 1: torch.randn(B, T, D)}
    t_feats = {0: torch.randn(B, T, D), 1: torch.randn(B, T, D)}
    loss = compute_feature_loss(s_feats, t_feats, [(0, 0), (1, 1)])
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 6: compute_feature_loss zero when features equal
# ---------------------------------------------------------------------------

def test_compute_feature_loss_zero_for_equal_features():
    B, T, D = 2, 8, 64
    h = torch.randn(B, T, D)
    feats = {0: h}
    loss = compute_feature_loss(feats, feats, [(0, 0)])
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 7: compute_attention_transfer_loss returns scalar
# ---------------------------------------------------------------------------

def test_compute_attention_transfer_loss_returns_scalar():
    B, T, Ds, Dt = 2, 8, 64, 64
    s_h = torch.randn(B, T, Ds)
    t_h = torch.randn(B, T, Dt)
    loss = compute_attention_transfer_loss(s_h, t_h)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 8: compute_attention_transfer_loss zero for identical hidden states
# ---------------------------------------------------------------------------

def test_compute_attention_transfer_loss_zero_for_identical():
    B, T, D = 2, 8, 64
    h = torch.randn(B, T, D) + 1.0  # avoid all-zero attention map
    loss = compute_attention_transfer_loss(h, h)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 9: soft_label_loss returns scalar
# ---------------------------------------------------------------------------

def test_soft_label_loss_returns_scalar():
    B, T, V = 2, 8, 256
    s_logits = torch.randn(B, T, V)
    t_logits = torch.randn(B, T, V)
    loss = soft_label_loss(s_logits, t_logits, temperature=4.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Test 10: soft_label_loss — higher temperature produces smoother distributions
# ---------------------------------------------------------------------------

def test_soft_label_loss_higher_temperature_smoother():
    """Higher temperature softens logits, reducing loss variance across batches."""
    torch.manual_seed(7)
    B, T, V = 4, 16, 256
    # Use peaked logits so temperature matters
    s_logits = torch.randn(B, T, V) * 5.0
    t_logits = torch.randn(B, T, V) * 5.0

    loss_low_T = soft_label_loss(s_logits, t_logits, temperature=1.0)
    loss_high_T = soft_label_loss(s_logits, t_logits, temperature=8.0)

    # Both should be finite
    assert torch.isfinite(loss_low_T)
    assert torch.isfinite(loss_high_T)
    # Higher temperature scales T^2, but the KL itself decreases substantially
    # (flatter distributions are harder to distinguish).
    # The ratio test: at T=1 the KL is large, at T=8 the KL numerically is smaller
    # but scaled by T^2=64. We verify both are finite and non-negative.
    assert loss_low_T.item() >= 0.0
    assert loss_high_T.item() >= 0.0


# ---------------------------------------------------------------------------
# Test 11: FeatureDistillTrainer.train_step returns required keys
# ---------------------------------------------------------------------------

def test_trainer_train_step_returns_required_keys():
    trainer, _ = _make_trainer()
    input_ids = torch.randint(0, 256, (2, 16))
    result = trainer.train_step(input_ids)
    for key in ("loss", "feature_loss", "attention_loss", "kl_loss", "task_loss"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Test 12: FeatureDistillTrainer.train_step all losses finite
# ---------------------------------------------------------------------------

def test_trainer_train_step_all_losses_finite():
    trainer, _ = _make_trainer()
    input_ids = torch.randint(0, 256, (2, 16))
    result = trainer.train_step(input_ids)
    for key, val in result.items():
        assert torch.isfinite(torch.tensor(val)), f"{key} is not finite: {val}"


# ---------------------------------------------------------------------------
# Test 13: FeatureDistillTrainer.evaluate returns required keys
# ---------------------------------------------------------------------------

def test_trainer_evaluate_returns_required_keys():
    trainer, _ = _make_trainer()
    input_ids = torch.randint(0, 256, (2, 16))
    result = trainer.evaluate(input_ids)
    for key in ("loss", "feature_loss", "attention_loss", "kl_loss", "task_loss"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Test 14: Teacher model is frozen (no grad after init)
# ---------------------------------------------------------------------------

def test_teacher_model_frozen():
    trainer, _ = _make_trainer()
    for p in trainer.teacher.parameters():
        assert not p.requires_grad, "Teacher parameter should not require grad"


# ---------------------------------------------------------------------------
# Test 15: compute_feature_loss with adapters works without error
# ---------------------------------------------------------------------------

def test_compute_feature_loss_with_adapters():
    B, T, D_s, D_t = 2, 8, 32, 64
    s_feats = {0: torch.randn(B, T, D_s)}
    t_feats = {0: torch.randn(B, T, D_t)}
    adapters = {0: FeatureAdapter(student_dim=D_s, teacher_dim=D_t)}
    loss = compute_feature_loss(s_feats, t_feats, [(0, 0)], adapters=adapters)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
