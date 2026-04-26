"""Tests for advanced quantization-aware training (LSQ, STE, mixed-precision)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.qat_advanced import (
    AdvancedQATConfig,
    FakeQuantLinear,
    LSQQuantizer,
    MixedPrecisionScheduler,
    clamp_ste,
    convert_to_fake_quant,
    round_ste,
)

IN_FEATURES = 32
OUT_FEATURES = 32


# ---------------------------------------------------------------------------
# 1. AdvancedQATConfig defaults
# ---------------------------------------------------------------------------


def test_advanced_qat_config_defaults():
    """AdvancedQATConfig must have the specified default values."""
    cfg = AdvancedQATConfig()
    assert cfg.bits == 8
    assert cfg.per_channel is True
    assert cfg.symmetric is True
    assert cfg.use_lsq is True
    assert cfg.lsq_init_factor == 2.0
    assert cfg.clip_val == 6.0
    assert cfg.mixed_precision_layers == ["lm_head"]
    assert cfg.warmup_steps == 1000


# ---------------------------------------------------------------------------
# 2. round_ste forward rounds correctly
# ---------------------------------------------------------------------------


def test_round_ste_forward():
    """round_ste must round values in the forward pass."""
    x = torch.tensor([0.4, 0.6, 1.3, -0.7, -1.5])
    y = round_ste(x)
    expected = x.round()
    assert torch.allclose(y, expected), f"Expected {expected}, got {y}"


# ---------------------------------------------------------------------------
# 3. round_ste backward passes gradient through (STE)
# ---------------------------------------------------------------------------


def test_round_ste_backward():
    """round_ste backward must pass gradient through unchanged (STE)."""
    x = torch.tensor([0.4, 0.6, 1.3], requires_grad=True)
    y = round_ste(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones(3)), f"Expected gradient of ones, got {x.grad}"


# ---------------------------------------------------------------------------
# 4. clamp_ste forward clamps correctly
# ---------------------------------------------------------------------------


def test_clamp_ste_forward():
    """clamp_ste must clamp values in the forward pass."""
    x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
    y = clamp_ste(x, -2.0, 2.0)
    expected = x.clamp(-2.0, 2.0)
    assert torch.allclose(y, expected), f"Expected {expected}, got {y}"


# ---------------------------------------------------------------------------
# 5. clamp_ste backward passes gradient through for clamped values
# ---------------------------------------------------------------------------


def test_clamp_ste_backward_passes_through():
    """clamp_ste backward must pass gradient through (STE) even for clamped values."""
    x = torch.tensor([-5.0, 0.0, 5.0], requires_grad=True)
    y = clamp_ste(x, -2.0, 2.0)
    y.sum().backward()
    assert x.grad is not None
    # STE: gradient is 1 everywhere (passes through clamp)
    assert torch.allclose(x.grad, torch.ones(3)), f"Expected gradient of ones via STE, got {x.grad}"


# ---------------------------------------------------------------------------
# 6. LSQQuantizer output shape matches input
# ---------------------------------------------------------------------------


def test_lsq_quantizer_output_shape():
    """LSQQuantizer output must have the same shape as input."""
    AdvancedQATConfig()
    q = LSQQuantizer(bits=8, per_channel=True, n_channels=OUT_FEATURES)
    x = torch.randn(OUT_FEATURES, IN_FEATURES)
    out = q(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 7. LSQQuantizer per_channel has n_channels step sizes
# ---------------------------------------------------------------------------


def test_lsq_quantizer_per_channel_step_sizes():
    """LSQQuantizer with per_channel=True must have n_channels step sizes."""
    n_channels = OUT_FEATURES
    q = LSQQuantizer(bits=8, per_channel=True, n_channels=n_channels)
    assert q.step_size.shape == (n_channels,), (
        f"Expected step_size shape ({n_channels},), got {q.step_size.shape}"
    )


# ---------------------------------------------------------------------------
# 8. LSQQuantizer initialize_from_weights sets step_size > 0
# ---------------------------------------------------------------------------


def test_lsq_quantizer_initialize_from_weights():
    """initialize_from_weights must set step_size to a positive value."""
    q = LSQQuantizer(bits=8, per_channel=True, n_channels=OUT_FEATURES)
    weights = torch.randn(OUT_FEATURES, IN_FEATURES)
    q.initialize_from_weights(weights)
    assert (q.step_size > 0).all(), "All step sizes must be positive after init"


# ---------------------------------------------------------------------------
# 9. FakeQuantLinear disabled: same as regular linear
# ---------------------------------------------------------------------------


def test_fake_quant_linear_disabled_matches_regular():
    """Disabled FakeQuantLinear must produce the same output as nn.Linear."""
    torch.manual_seed(42)
    cfg = AdvancedQATConfig()
    fq = FakeQuantLinear(IN_FEATURES, OUT_FEATURES, cfg, bias=True)
    fq.enabled = False

    # Mirror weights into a regular linear
    lin = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True)
    with torch.no_grad():
        lin.weight.copy_(fq.weight)
        lin.bias.copy_(fq.bias)  # type: ignore[arg-type]

    x = torch.randn(4, IN_FEATURES)
    out_fq = fq(x)
    out_lin = lin(x)
    assert torch.allclose(out_fq, out_lin, atol=1e-6), (
        "Disabled FakeQuantLinear must match nn.Linear output"
    )


# ---------------------------------------------------------------------------
# 10. FakeQuantLinear enabled: output shape correct
# ---------------------------------------------------------------------------


def test_fake_quant_linear_enabled_output_shape():
    """Enabled FakeQuantLinear must produce output of correct shape."""
    cfg = AdvancedQATConfig()
    fq = FakeQuantLinear(IN_FEATURES, OUT_FEATURES, cfg, bias=True)
    fq.enable_quantization()

    x = torch.randn(2, 8, IN_FEATURES)
    out = fq(x)
    assert out.shape == (2, 8, OUT_FEATURES), f"Expected (2, 8, {OUT_FEATURES}), got {out.shape}"


# ---------------------------------------------------------------------------
# 11. FakeQuantLinear enable/disable quantization works
# ---------------------------------------------------------------------------


def test_fake_quant_linear_enable_disable():
    """enable_quantization and disable_quantization must toggle the enabled flag."""
    cfg = AdvancedQATConfig()
    fq = FakeQuantLinear(IN_FEATURES, OUT_FEATURES, cfg)

    assert not fq.enabled, "FakeQuantLinear must start disabled"

    fq.enable_quantization()
    assert fq.enabled, "enabled must be True after enable_quantization()"

    fq.disable_quantization()
    assert not fq.enabled, "enabled must be False after disable_quantization()"


# ---------------------------------------------------------------------------
# 12. MixedPrecisionScheduler step enables quantization after warmup
# ---------------------------------------------------------------------------


def test_mixed_precision_scheduler_enables_after_warmup():
    """MixedPrecisionScheduler must enable quantization only after warmup_steps."""
    cfg = AdvancedQATConfig(warmup_steps=100, mixed_precision_layers=["lm_head"])

    # Build a simple model with FakeQuantLinear layers
    model = nn.Sequential(
        FakeQuantLinear(IN_FEATURES, OUT_FEATURES, cfg),
        FakeQuantLinear(OUT_FEATURES, IN_FEATURES, cfg),
    )

    scheduler = MixedPrecisionScheduler(model, cfg)

    # Before warmup: no layers should be quantized
    scheduler.step(0)
    assert all(not m.enabled for m in model.modules() if isinstance(m, FakeQuantLinear))

    scheduler.step(99)
    assert all(not m.enabled for m in model.modules() if isinstance(m, FakeQuantLinear))

    # At/after warmup: layers should be enabled
    scheduler.step(100)
    assert any(m.enabled for m in model.modules() if isinstance(m, FakeQuantLinear)), (
        "At least one layer must be enabled after warmup_steps"
    )


# ---------------------------------------------------------------------------
# 13. convert_to_fake_quant replaces linear layers
# ---------------------------------------------------------------------------


def test_convert_to_fake_quant_replaces_linears():
    """convert_to_fake_quant must replace nn.Linear with FakeQuantLinear."""
    torch.manual_seed(0)
    model_cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    model = AureliusTransformer(model_cfg)
    qat_cfg = AdvancedQATConfig(mixed_precision_layers=["lm_head"])

    convert_to_fake_quant(model, qat_cfg)

    n_fq = sum(1 for m in model.modules() if isinstance(m, FakeQuantLinear))
    assert n_fq > 0, "convert_to_fake_quant must create at least one FakeQuantLinear"


# ---------------------------------------------------------------------------
# 14. convert_to_fake_quant skips mixed_precision_layers
# ---------------------------------------------------------------------------


def test_convert_to_fake_quant_skips_mixed_precision():
    """convert_to_fake_quant must leave mixed_precision_layers as nn.Linear."""
    torch.manual_seed(0)
    model_cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    model = AureliusTransformer(model_cfg)
    qat_cfg = AdvancedQATConfig(mixed_precision_layers=["lm_head"])

    convert_to_fake_quant(model, qat_cfg)

    # lm_head must still be an nn.Linear (not FakeQuantLinear)
    for name, module in model.named_modules():
        if "lm_head" in name and isinstance(module, (nn.Linear, FakeQuantLinear)):
            assert not isinstance(module, FakeQuantLinear), (
                f"Layer '{name}' is in mixed_precision_layers but was converted to FakeQuantLinear"
            )
