"""Tests for quantization_aware training module."""
import torch
import torch.nn as nn
import pytest

from src.training.quantization_aware import (
    QATConfig,
    compute_quantization_params,
    fake_quantize,
    FakeQuantize,
    QATLinear,
    apply_qat,
    compute_model_quantization_error,
)

# Constants used throughout tests
IN_FEATURES = 16
OUT_FEATURES = 8
N_BITS = 8
BATCH = 2

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# 1. QATConfig defaults
# ---------------------------------------------------------------------------
def test_qatconfig_defaults():
    """QATConfig must have n_bits=8, symmetric=True as defaults."""
    cfg = QATConfig()
    assert cfg.n_bits == 8
    assert cfg.symmetric is True
    assert cfg.per_channel is False


# ---------------------------------------------------------------------------
# 2. compute_quantization_params returns (scale, zero_point) tensors
# ---------------------------------------------------------------------------
def test_compute_quantization_params_returns_tensors():
    """compute_quantization_params must return two tensors."""
    x = torch.randn(BATCH, IN_FEATURES)
    scale, zp = compute_quantization_params(x, n_bits=N_BITS, symmetric=True)
    assert isinstance(scale, torch.Tensor)
    assert isinstance(zp, torch.Tensor)


# ---------------------------------------------------------------------------
# 3. Symmetric zero_point is zero
# ---------------------------------------------------------------------------
def test_symmetric_zero_point_is_zero():
    """Symmetric quantization must produce a zero zero_point."""
    x = torch.randn(BATCH, IN_FEATURES)
    _, zp = compute_quantization_params(x, n_bits=N_BITS, symmetric=True)
    assert torch.all(zp == 0), f"Expected zero_point=0 for symmetric, got {zp}"


# ---------------------------------------------------------------------------
# 4. Asymmetric zero_point is integral
# ---------------------------------------------------------------------------
def test_asymmetric_zero_point_is_integral():
    """Asymmetric quantization zero_point must be a rounded integer value."""
    x = torch.randn(BATCH, IN_FEATURES)
    _, zp = compute_quantization_params(x, n_bits=N_BITS, symmetric=False)
    assert torch.allclose(zp, torch.round(zp)), "zero_point must be integral for asymmetric QAT"


# ---------------------------------------------------------------------------
# 5. fake_quantize output is close to input (small error for 8-bit)
# ---------------------------------------------------------------------------
def test_fake_quantize_close_to_input():
    """8-bit fake quantization error should be small relative to signal range."""
    x = torch.randn(BATCH, IN_FEATURES)
    scale, zp = compute_quantization_params(x, n_bits=N_BITS, symmetric=True)
    cfg = QATConfig(n_bits=N_BITS)
    x_fq = fake_quantize(x, scale, zp, cfg.quant_min, cfg.quant_max)
    mae = (x - x_fq).abs().mean().item()
    # For 8-bit, error should be < 1% of range
    x_range = x.max().item() - x.min().item()
    assert mae < 0.01 * x_range, f"MAE {mae:.4f} too large for range {x_range:.4f}"


# ---------------------------------------------------------------------------
# 6. fake_quantize gradient is nonzero (STE works)
# ---------------------------------------------------------------------------
def test_fake_quantize_gradient_nonzero():
    """STE must pass gradient through fake_quantize unchanged."""
    x = torch.randn(BATCH, IN_FEATURES, requires_grad=True)
    scale, zp = compute_quantization_params(x.detach(), n_bits=N_BITS, symmetric=True)
    cfg = QATConfig(n_bits=N_BITS)
    x_fq = fake_quantize(x, scale, zp, cfg.quant_min, cfg.quant_max)
    x_fq.sum().backward()
    assert x.grad is not None, "No gradient flowed through fake_quantize"
    assert x.grad.abs().sum() > 0, "Gradient is all zeros"


# ---------------------------------------------------------------------------
# 7. fake_quantize output stays in dequantized range
# ---------------------------------------------------------------------------
def test_fake_quantize_dequantized_range():
    """fake_quantize output must lie within [quant_min, quant_max] * scale."""
    x = torch.randn(BATCH, IN_FEATURES)
    cfg = QATConfig(n_bits=N_BITS, symmetric=True)
    scale, zp = compute_quantization_params(x, n_bits=N_BITS, symmetric=True)
    x_fq = fake_quantize(x, scale, zp, cfg.quant_min, cfg.quant_max)
    lo = cfg.quant_min * scale
    hi = cfg.quant_max * scale
    assert x_fq.min() >= lo.min() - 1e-6
    assert x_fq.max() <= hi.max() + 1e-6


# ---------------------------------------------------------------------------
# 8. FakeQuantize train mode changes output (quantization noise)
# ---------------------------------------------------------------------------
def test_fake_quantize_module_train_mode_changes_output():
    """FakeQuantize in train mode must alter the output (introduce quant noise)."""
    fq = FakeQuantize(QATConfig(n_bits=N_BITS))
    fq.train()
    x = torch.randn(BATCH, IN_FEATURES)
    x_fq = fq(x)
    # Outputs should differ because of rounding (unless extremely unlikely)
    assert not torch.allclose(x, x_fq, atol=0.0), "FakeQuantize in train mode should alter output"


# ---------------------------------------------------------------------------
# 9. FakeQuantize eval mode returns unchanged input
# ---------------------------------------------------------------------------
def test_fake_quantize_module_eval_mode_unchanged():
    """FakeQuantize in eval mode must return the input tensor unchanged."""
    fq = FakeQuantize(QATConfig(n_bits=N_BITS))
    fq.eval()
    x = torch.randn(BATCH, IN_FEATURES)
    x_out = fq(x)
    assert torch.equal(x, x_out), "FakeQuantize in eval mode must return x unchanged"


# ---------------------------------------------------------------------------
# 10. FakeQuantize quant_error is small for 8-bit
# ---------------------------------------------------------------------------
def test_fake_quantize_quant_error_small_8bit():
    """quant_error must return a small float for 8-bit quantization."""
    fq = FakeQuantize(QATConfig(n_bits=N_BITS))
    x = torch.randn(BATCH, IN_FEATURES)
    err = fq.quant_error(x)
    assert isinstance(err, float)
    # For 8-bit, error should be very small relative to typical values ~N(0,1)
    assert err < 0.1, f"quant_error {err:.4f} too large for 8-bit"


# ---------------------------------------------------------------------------
# 11. QATLinear output shape matches nn.Linear
# ---------------------------------------------------------------------------
def test_qat_linear_output_shape():
    """QATLinear must produce the same output shape as nn.Linear."""
    cfg = QATConfig(n_bits=N_BITS)
    qat_lin = QATLinear(IN_FEATURES, OUT_FEATURES, config=cfg)
    qat_lin.train()
    x = torch.randn(BATCH, IN_FEATURES)
    out = qat_lin(x)
    assert out.shape == (BATCH, OUT_FEATURES), f"Unexpected shape {out.shape}"


# ---------------------------------------------------------------------------
# 12. QATLinear gradient flows to weights
# ---------------------------------------------------------------------------
def test_qat_linear_gradient_flows_to_weights():
    """Gradients must flow back to QATLinear weights via STE."""
    cfg = QATConfig(n_bits=N_BITS)
    qat_lin = QATLinear(IN_FEATURES, OUT_FEATURES, config=cfg)
    qat_lin.train()
    x = torch.randn(BATCH, IN_FEATURES)
    out = qat_lin(x)
    out.sum().backward()
    assert qat_lin.linear.weight.grad is not None, "No gradient reached QATLinear weights"
    assert qat_lin.linear.weight.grad.abs().sum() > 0, "Gradient is all zeros"


# ---------------------------------------------------------------------------
# 13. apply_qat returns correct count
# ---------------------------------------------------------------------------
def test_apply_qat_returns_correct_count():
    """apply_qat must return the number of modules replaced."""
    model = nn.Sequential(
        nn.Linear(IN_FEATURES, OUT_FEATURES),
        nn.ReLU(),
        nn.Linear(OUT_FEATURES, OUT_FEATURES),
    )
    cfg = QATConfig(n_bits=N_BITS)
    count = apply_qat(model, cfg)
    assert count == 2, f"Expected 2 replaced, got {count}"


# ---------------------------------------------------------------------------
# 14. apply_qat replaced modules are QATLinear
# ---------------------------------------------------------------------------
def test_apply_qat_replaced_modules_are_qat_linear():
    """After apply_qat, all direct-child Linear slots must be QATLinear instances."""
    model = nn.Sequential(
        nn.Linear(IN_FEATURES, OUT_FEATURES),
        nn.ReLU(),
        nn.Linear(OUT_FEATURES, OUT_FEATURES),
    )
    cfg = QATConfig(n_bits=N_BITS)
    apply_qat(model, cfg)
    # Direct children of the Sequential should all be QATLinear (or non-Linear)
    qat_linears = [m for m in model.modules() if isinstance(m, QATLinear)]
    assert len(qat_linears) == 2, f"Expected 2 QATLinear, found {len(qat_linears)}"
    # No direct child should still be a bare nn.Linear (QATLinear wraps an inner
    # nn.Linear, so we only check the Sequential's immediate children)
    for child in model.children():
        assert not (type(child) is nn.Linear), \
            f"Direct child {child} is still a plain nn.Linear after apply_qat"


# ---------------------------------------------------------------------------
# 15. compute_model_quantization_error returns required keys
# ---------------------------------------------------------------------------
def test_compute_model_quantization_error_keys():
    """compute_model_quantization_error must return mean_abs_error and max_abs_error."""
    model = nn.Sequential(nn.Linear(IN_FEATURES, OUT_FEATURES))
    cfg = QATConfig(n_bits=N_BITS)
    apply_qat(model, cfg)
    x = torch.randn(BATCH, IN_FEATURES)
    result = compute_model_quantization_error(model, x)
    assert "mean_abs_error" in result, "Missing 'mean_abs_error' key"
    assert "max_abs_error" in result, "Missing 'max_abs_error' key"
    assert isinstance(result["mean_abs_error"], float)
    assert isinstance(result["max_abs_error"], float)


# ---------------------------------------------------------------------------
# 16. per_channel scale has shape matching channel count
# ---------------------------------------------------------------------------
def test_per_channel_scale_shape():
    """Per-channel scale must have one entry per output channel."""
    x = torch.randn(OUT_FEATURES, IN_FEATURES)  # weight-shaped: [out, in]
    scale, zp = compute_quantization_params(
        x, n_bits=N_BITS, symmetric=True, per_channel=True, channel_dim=0
    )
    assert scale.shape == (OUT_FEATURES,), f"Expected scale shape ({OUT_FEATURES},), got {scale.shape}"
    assert zp.shape == (OUT_FEATURES,), f"Expected zp shape ({OUT_FEATURES},), got {zp.shape}"
