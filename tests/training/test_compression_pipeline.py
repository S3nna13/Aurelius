"""Tests for the end-to-end compression pipeline."""
import torch
import torch.nn as nn
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.compression_pipeline import (
    CompressionConfig,
    compute_model_sparsity,
    magnitude_prune,
    distillation_loss,
    fake_quantize_weights,
    PruneStage,
    DistillStage,
    QuantizeStage,
    CompressionPipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(seed: int = 0) -> AureliusTransformer:
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


def _make_data(B: int = 2, S: int = 16, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, 256, (B, S))


# ---------------------------------------------------------------------------
# 1. CompressionConfig defaults
# ---------------------------------------------------------------------------

def test_compression_config_defaults():
    cfg = CompressionConfig()
    assert cfg.prune_fraction == 0.3
    assert cfg.distill_temperature == 4.0
    assert cfg.distill_alpha == 0.7
    assert cfg.quantize_bits == 8
    assert cfg.stages == ["prune", "distill", "quantize"]
    assert cfg.stage_epochs == {"prune": 1, "distill": 2, "quantize": 1}


# ---------------------------------------------------------------------------
# 2. compute_model_sparsity on fresh model — near 0 sparsity
# ---------------------------------------------------------------------------

def test_sparsity_fresh_model_near_zero():
    model = _make_model()
    info = compute_model_sparsity(model)
    # A freshly initialised model should have effectively zero sparsity
    assert info["overall_sparsity"] < 0.01
    assert info["n_total_params"] > 0
    assert isinstance(info["n_zero_params"], int)


# ---------------------------------------------------------------------------
# 3. compute_model_sparsity after zeroing weights shows sparsity
# ---------------------------------------------------------------------------

def test_sparsity_after_zeroing():
    model = _make_model()
    # Manually zero half the weights of the first linear layer
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.data.zero_()
                break  # zero just one layer to confirm detection

    info = compute_model_sparsity(model)
    assert info["overall_sparsity"] > 0.0
    assert info["n_zero_params"] > 0


# ---------------------------------------------------------------------------
# 4. magnitude_prune increases sparsity
# ---------------------------------------------------------------------------

def test_magnitude_prune_increases_sparsity():
    model = _make_model()
    before = compute_model_sparsity(model)
    magnitude_prune(model, 0.3)
    after = compute_model_sparsity(model)
    assert after["overall_sparsity"] > before["overall_sparsity"]


# ---------------------------------------------------------------------------
# 5. magnitude_prune prunes correct fraction
# ---------------------------------------------------------------------------

def test_magnitude_prune_correct_fraction():
    model = _make_model()
    fraction = 0.5
    magnitude_prune(model, fraction)
    info = compute_model_sparsity(model)
    # Allow ±5% tolerance due to ties at the threshold boundary
    assert abs(info["overall_sparsity"] - fraction) < 0.05


# ---------------------------------------------------------------------------
# 6. distillation_loss returns scalar
# ---------------------------------------------------------------------------

def test_distillation_loss_returns_scalar():
    B, S, V = 2, 8, 256
    student = torch.randn(B, S, V)
    teacher = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))
    loss = distillation_loss(student, teacher, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 7. distillation_loss alpha=1.0 ignores task loss
# ---------------------------------------------------------------------------

def test_distillation_loss_alpha_one_ignores_task():
    B, S, V = 2, 8, 256
    torch.manual_seed(1)
    student = torch.randn(B, S, V)
    teacher = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))

    # With alpha=1.0 the result equals pure KD loss
    loss_alpha1 = distillation_loss(student, teacher, labels, alpha=1.0)

    # Compute KD loss manually
    s_flat = student.view(-1, V)
    t_flat = teacher.view(-1, V)
    T = 4.0
    s_log = torch.nn.functional.log_softmax(s_flat / T, dim=-1)
    t_soft = torch.nn.functional.softmax(t_flat / T, dim=-1)
    expected_kd = torch.nn.functional.kl_div(s_log, t_soft, reduction="batchmean") * (T ** 2)

    assert abs(loss_alpha1.item() - expected_kd.item()) < 1e-4


# ---------------------------------------------------------------------------
# 8. distillation_loss alpha=0.0 ignores KD loss
# ---------------------------------------------------------------------------

def test_distillation_loss_alpha_zero_ignores_kd():
    B, S, V = 2, 8, 256
    torch.manual_seed(2)
    student = torch.randn(B, S, V)
    teacher = torch.randn(B, S, V)
    labels = torch.randint(0, V, (B, S))

    loss_alpha0 = distillation_loss(student, teacher, labels, alpha=0.0)

    # Should equal cross-entropy
    expected_ce = torch.nn.functional.cross_entropy(student.view(-1, V), labels.view(-1))
    assert abs(loss_alpha0.item() - expected_ce.item()) < 1e-4


# ---------------------------------------------------------------------------
# 9. fake_quantize_weights — low quantization error
# ---------------------------------------------------------------------------

def test_fake_quantize_low_error():
    model = _make_model()
    data = _make_data()

    model.eval()
    with torch.no_grad():
        out_before = model(data)
        logits_before = out_before[1] if isinstance(out_before, tuple) else out_before

    fake_quantize_weights(model, bits=8)

    with torch.no_grad():
        out_after = model(data)
        logits_after = out_after[1] if isinstance(out_after, tuple) else out_after

    mse = torch.nn.functional.mse_loss(logits_after, logits_before).item()
    # 8-bit fake quant on a small model should have very small MSE
    assert mse < 1.0, f"Quantization error too large: {mse:.4f}"


# ---------------------------------------------------------------------------
# 10. PruneStage.apply returns sparsity metric
# ---------------------------------------------------------------------------

def test_prune_stage_returns_sparsity():
    model = _make_model()
    data = _make_data()
    cfg = CompressionConfig(prune_fraction=0.3, stages=["prune"])

    stage = PruneStage()
    model_out, metrics = stage.apply(model, data, cfg)

    assert "sparsity" in metrics
    assert 0.0 <= metrics["sparsity"] <= 1.0
    assert isinstance(model_out, nn.Module)


# ---------------------------------------------------------------------------
# 11. DistillStage.apply returns distill_loss metric
# ---------------------------------------------------------------------------

def test_distill_stage_returns_distill_loss():
    student = _make_model(seed=0)
    teacher = _make_model(seed=1)
    data = _make_data()
    cfg = CompressionConfig(stages=["distill"])

    stage = DistillStage()
    model_out, metrics = stage.apply(student, data, cfg, teacher=teacher)

    assert "distill_loss" in metrics
    assert "kd_loss" in metrics
    assert "task_loss" in metrics
    assert isinstance(metrics["distill_loss"], float)
    assert torch.isfinite(torch.tensor(metrics["distill_loss"]))


# ---------------------------------------------------------------------------
# 12. QuantizeStage.apply returns quant_error metric
# ---------------------------------------------------------------------------

def test_quantize_stage_returns_quant_error():
    model = _make_model()
    data = _make_data()
    cfg = CompressionConfig(quantize_bits=8, stages=["quantize"])

    stage = QuantizeStage()
    model_out, metrics = stage.apply(model, data, cfg)

    assert "quant_error" in metrics
    assert isinstance(metrics["quant_error"], float)
    assert metrics["quant_error"] >= 0.0


# ---------------------------------------------------------------------------
# 13. CompressionPipeline.run returns metrics for each stage
# ---------------------------------------------------------------------------

def test_pipeline_run_returns_all_stage_metrics():
    student = _make_model(seed=0)
    teacher = _make_model(seed=1)
    data = _make_data()
    cfg = CompressionConfig(
        prune_fraction=0.2,
        stages=["prune", "distill", "quantize"],
    )

    pipeline = CompressionPipeline(student, teacher, cfg)
    results = pipeline.run(data)

    assert "prune" in results
    assert "distill" in results
    assert "quantize" in results

    assert "sparsity" in results["prune"]
    assert "distill_loss" in results["distill"]
    assert "quant_error" in results["quantize"]


# ---------------------------------------------------------------------------
# 14. CompressionPipeline.get_compressed_model returns nn.Module
# ---------------------------------------------------------------------------

def test_pipeline_get_compressed_model():
    student = _make_model(seed=0)
    teacher = _make_model(seed=1)
    data = _make_data()
    cfg = CompressionConfig(stages=["prune", "quantize"])

    pipeline = CompressionPipeline(student, teacher, cfg)
    pipeline.run(data)
    compressed = pipeline.get_compressed_model()

    assert isinstance(compressed, nn.Module)
