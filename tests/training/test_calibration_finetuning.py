"""Tests for src/training/calibration_finetuning.py

Tiny configs:
    vocab_size / n_classes = 16
    seq_len  = 8
    batch    = 4
    N (val)  = 32
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from src.training.calibration_finetuning import (
    CalibrationBenchmark,
    CalibrationConfig,
    FocalLoss,
    LabelSmoothingLoss,
    MixupTrainer,
    TemperatureScalingTrainer,
    VectorScalingTrainer,
)

# ---------------------------------------------------------------------------
# Shared tiny constants
# ---------------------------------------------------------------------------
V = 16      # vocab / n_classes
T = 8       # seq_len
B = 4       # batch
N = 32      # validation samples


# ---------------------------------------------------------------------------
# Minimal language-model stub used by MixupTrainer
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    """Minimal LM that accepts inputs_embeds and returns [B, T, V] logits."""

    def __init__(self, vocab_size: int = V, d_model: int = 32, seq_len: int = T):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed(input_ids)
        return self.proj(x)  # [B, T, V]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_logits(b: int = B, t: int = T, v: int = V) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(b, t, v)


def make_targets(b: int = B, t: int = T, v: int = V) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, v, (b, t))


def make_val_logits(n: int = N, c: int = V) -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(n, c)


def make_val_labels(n: int = N, c: int = V) -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randint(0, c, (n,))


def make_probs(n: int = N, c: int = V) -> torch.Tensor:
    torch.manual_seed(4)
    return F.softmax(torch.randn(n, c), dim=-1)


# ===========================================================================
# 1. LabelSmoothingLoss — forward produces finite positive scalar
# ===========================================================================

def test_label_smoothing_loss_finite_positive():
    loss_fn = LabelSmoothingLoss(vocab_size=V, smoothing=0.1)
    loss = loss_fn(make_logits(), make_targets())
    assert loss.ndim == 0, "Expected scalar"
    assert math.isfinite(loss.item()), "Loss must be finite"
    assert loss.item() > 0.0, "Loss must be positive"


# ===========================================================================
# 2. LabelSmoothingLoss with smoothing=0 matches standard cross-entropy
# ===========================================================================

def test_label_smoothing_loss_zero_smoothing_matches_ce():
    logits = make_logits()
    targets = make_targets()

    loss_smooth = LabelSmoothingLoss(vocab_size=V, smoothing=0.0)
    smooth_val = loss_smooth(logits, targets).item()

    ce_val = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1)).item()
    assert abs(smooth_val - ce_val) < 1e-4, (
        f"smoothing=0 should match CE: {smooth_val:.6f} vs {ce_val:.6f}"
    )


# ===========================================================================
# 3. LabelSmoothingLoss — ignore_index positions are excluded
# ===========================================================================

def test_label_smoothing_loss_ignore_index():
    logits = make_logits()
    targets = make_targets()

    # Mask out first half of the sequence
    targets_masked = targets.clone()
    targets_masked[:, : T // 2] = -100

    loss_fn = LabelSmoothingLoss(vocab_size=V, smoothing=0.1, ignore_index=-100)
    loss_full = loss_fn(logits, targets)
    loss_masked = loss_fn(logits, targets_masked)

    # Losses should differ when tokens are masked
    assert abs(loss_full.item() - loss_masked.item()) > 1e-6, (
        "Masking half the tokens should change the loss"
    )


# ===========================================================================
# 4. TemperatureScalingTrainer — calibrate returns positive float
# ===========================================================================

def test_temperature_scaling_calibrate_positive():
    model = TinyLM()
    trainer = TemperatureScalingTrainer(model)
    temp = trainer.calibrate(make_val_logits(), make_val_labels())
    assert isinstance(temp, float), "calibrate should return float"
    assert temp > 0.0, f"Temperature must be positive, got {temp}"


# ===========================================================================
# 5. TemperatureScalingTrainer — scaled_logits changes values
# ===========================================================================

def test_temperature_scaling_scaled_logits_differs():
    model = TinyLM()
    trainer = TemperatureScalingTrainer(model)
    # Force a non-unity temperature
    trainer.temperature = nn.Parameter(torch.tensor([2.0]))
    logits = make_val_logits()
    scaled = trainer.scaled_logits(logits)
    assert not torch.allclose(logits, scaled), (
        "scaled_logits with T!=1 should change values"
    )
    assert logits.shape == scaled.shape, "Shape must be preserved"


# ===========================================================================
# 6. TemperatureScalingTrainer — ECE in [0, 1]
# ===========================================================================

def test_temperature_scaling_ece_range():
    probs = make_probs()
    labels = make_val_labels()
    ece = TemperatureScalingTrainer.ece(probs, labels, n_bins=15)
    assert isinstance(ece, float), "ECE should be a float"
    assert 0.0 <= ece <= 1.0, f"ECE must be in [0,1], got {ece}"


# ===========================================================================
# 7. VectorScalingTrainer — calibrate runs without error
# ===========================================================================

def test_vector_scaling_calibrate_no_error():
    trainer = VectorScalingTrainer(n_classes=V)
    # Should complete without raising
    trainer.calibrate(make_val_logits(), make_val_labels(), n_steps=10)


# ===========================================================================
# 8. VectorScalingTrainer — scaled_logits shape unchanged
# ===========================================================================

def test_vector_scaling_scaled_logits_shape():
    trainer = VectorScalingTrainer(n_classes=V)
    trainer.calibrate(make_val_logits(), make_val_labels(), n_steps=5)
    logits = make_val_logits()
    scaled = trainer.scaled_logits(logits)
    assert scaled.shape == logits.shape, (
        f"Shape mismatch: {scaled.shape} vs {logits.shape}"
    )


# ===========================================================================
# 9. FocalLoss — forward produces finite positive scalar
# ===========================================================================

def test_focal_loss_finite_positive():
    loss_fn = FocalLoss(gamma=2.0)
    loss = loss_fn(make_logits(), make_targets())
    assert loss.ndim == 0, "Expected scalar"
    assert math.isfinite(loss.item()), "FocalLoss must be finite"
    assert loss.item() > 0.0, "FocalLoss must be positive"


# ===========================================================================
# 10. FocalLoss gamma=0 matches standard cross-entropy
# ===========================================================================

def test_focal_loss_gamma_zero_matches_ce():
    logits = make_logits()
    targets = make_targets()

    focal = FocalLoss(gamma=0.0)
    focal_val = focal(logits, targets).item()

    ce_val = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1)).item()
    assert abs(focal_val - ce_val) < 1e-4, (
        f"FocalLoss(gamma=0) should match CE: {focal_val:.6f} vs {ce_val:.6f}"
    )


# ===========================================================================
# 11. FocalLoss — high-confidence correct predictions get lower weight
# ===========================================================================

def test_focal_loss_high_confidence_lower_weight():
    """
    Construct two scenarios:
      A) Model is very confident and correct  → low focal weight
      B) Model is uncertain                   → higher focal weight
    Focal loss A should be less than focal loss B.
    """
    gamma = 2.0
    loss_fn = FocalLoss(gamma=gamma)

    # Scenario A: very confident correct prediction
    targets = torch.zeros(B, T, dtype=torch.long)   # all class 0
    logits_confident = torch.full((B, T, V), -10.0)
    logits_confident[:, :, 0] = 10.0                # class 0 gets huge score

    # Scenario B: uniform (uncertain) logits
    logits_uniform = torch.zeros(B, T, V)

    loss_confident = loss_fn(logits_confident, targets).item()
    loss_uniform = loss_fn(logits_uniform, targets).item()

    assert loss_confident < loss_uniform, (
        f"High-confidence correct loss ({loss_confident:.6f}) should be less "
        f"than uncertain loss ({loss_uniform:.6f})"
    )


# ===========================================================================
# 12. MixupTrainer — mixup_batch returns correct embedding shape
# ===========================================================================

def test_mixup_batch_shape():
    model = TinyLM()
    trainer = MixupTrainer(model, alpha=0.2)
    ids_a = torch.randint(0, V, (B, T))
    ids_b = torch.randint(0, V, (B, T))
    mixed, lam = trainer.mixup_batch(ids_a, ids_b)
    d_model = model.embed.embedding_dim
    assert mixed.shape == (B, T, d_model), (
        f"Expected ({B},{T},{d_model}), got {mixed.shape}"
    )


# ===========================================================================
# 13. MixupTrainer — lambda in [0, 1]
# ===========================================================================

def test_mixup_lambda_range():
    model = TinyLM()
    trainer = MixupTrainer(model, alpha=0.5)
    ids_a = torch.randint(0, V, (B, T))
    ids_b = torch.randint(0, V, (B, T))
    for _ in range(20):
        _, lam = trainer.mixup_batch(ids_a, ids_b)
        assert 0.0 <= lam <= 1.0, f"Lambda {lam} out of [0, 1]"


# ===========================================================================
# 14. MixupTrainer — train_step returns finite loss
# ===========================================================================

def test_mixup_train_step_finite_loss():
    model = TinyLM()
    trainer = MixupTrainer(model, alpha=0.2)
    ids_a = torch.randint(0, V, (B, T))
    ids_b = torch.randint(0, V, (B, T))
    labels_a = torch.randint(0, V, (B, T))
    labels_b = torch.randint(0, V, (B, T))
    loss = trainer.train_step(model, ids_a, ids_b, labels_a, labels_b)
    assert math.isfinite(loss.item()), f"train_step loss is not finite: {loss.item()}"
    assert loss.item() >= 0.0, "Loss must be non-negative"


# ===========================================================================
# 15. CalibrationBenchmark — reliability_diagram_data returns correct keys
# ===========================================================================

def test_reliability_diagram_data_keys():
    bench = CalibrationBenchmark()
    probs = make_probs()
    labels = make_val_labels()
    result = bench.reliability_diagram_data(probs, labels, n_bins=10)
    assert "bin_accs" in result, "Missing key 'bin_accs'"
    assert "bin_confs" in result, "Missing key 'bin_confs'"
    assert "bin_counts" in result, "Missing key 'bin_counts'"
    assert len(result["bin_accs"]) == 10, "bin_accs length should equal n_bins"
    assert len(result["bin_confs"]) == 10, "bin_confs length should equal n_bins"
    assert len(result["bin_counts"]) == 10, "bin_counts length should equal n_bins"


# ===========================================================================
# 16. CalibrationBenchmark — mce in [0, 1]
# ===========================================================================

def test_calibration_benchmark_mce_range():
    bench = CalibrationBenchmark()
    probs = make_probs()
    labels = make_val_labels()
    mce = bench.mce(probs, labels, n_bins=15)
    assert isinstance(mce, float), "MCE should be a float"
    assert 0.0 <= mce <= 1.0, f"MCE must be in [0,1], got {mce}"


# ===========================================================================
# 17. CalibrationBenchmark — brier_score non-negative
# ===========================================================================

def test_calibration_benchmark_brier_score_non_negative():
    bench = CalibrationBenchmark()
    probs = make_probs()
    labels = make_val_labels()
    bs = bench.brier_score(probs, labels)
    assert isinstance(bs, float), "Brier score should be a float"
    assert bs >= 0.0, f"Brier score must be >= 0, got {bs}"


# ===========================================================================
# 18. CalibrationConfig — default values
# ===========================================================================

def test_calibration_config_defaults():
    cfg = CalibrationConfig()
    assert cfg.smoothing == 0.1
    assert cfg.gamma == 2.0
    assert cfg.alpha == 0.2
    assert cfg.n_bins == 15
    assert cfg.lr == 0.01
    assert cfg.n_steps == 100


# ===========================================================================
# 19. LabelSmoothingLoss — gradient flows through loss
# ===========================================================================

def test_label_smoothing_loss_gradient():
    logits = make_logits().requires_grad_(True)
    targets = make_targets()
    loss_fn = LabelSmoothingLoss(vocab_size=V, smoothing=0.1)
    loss = loss_fn(logits, targets)
    loss.backward()
    assert logits.grad is not None, "Gradient should flow to logits"
    assert logits.grad.shape == logits.shape


# ===========================================================================
# 20. VectorScalingTrainer — calibrated W and b are not all-zero/one after fit
# ===========================================================================

def test_vector_scaling_parameters_updated():
    trainer = VectorScalingTrainer(n_classes=V)
    trainer.calibrate(make_val_logits(), make_val_labels(), n_steps=50)
    # After fitting, b should have moved away from the zeros initialisation
    # (or W from ones) in at least some class dimensions
    b_changed = not torch.allclose(trainer.b, torch.zeros(V), atol=1e-6)
    w_changed = not torch.allclose(trainer.W, torch.ones(V), atol=1e-6)
    assert b_changed or w_changed, (
        "VectorScaling parameters should update during calibration"
    )
