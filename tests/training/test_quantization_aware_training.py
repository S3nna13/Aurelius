"""
Tests for quantization_aware_training.py

Covers:
  - FakeQuantize forward / backward
  - PerTensorQuantizer / PerChannelQuantizer
  - QATLinear / QATEmbedding
  - QATConverter (convert, export_int_model)
  - QuantizationTrainer
  - QATConfig defaults
"""

import torch
import torch.nn as nn
import pytest

from src.training.quantization_aware_training import (
    FakeQuantize,
    PerTensorQuantizer,
    PerChannelQuantizer,
    QATLinear,
    QATEmbedding,
    QATConverter,
    QuantizationTrainer,
    QATConfig,
)

# ---------------------------------------------------------------------------
# Tiny model used by several tests
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    """Minimal embedding + linear LM for testing."""

    def __init__(self, vocab: int = 32, d: int = 16, seq: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)          # [B, T, d]
        return self.proj(x)                # [B, T, vocab]


# ===========================================================================
# 1. FakeQuantize forward — values stay in valid dequantized range
# ===========================================================================
def test_fake_quantize_forward_values_in_range():
    x = torch.randn(8, 16)
    n_bits = 8
    q_max = 2 ** n_bits - 1
    scale = torch.full_like(x, 0.01)
    zp = torch.zeros_like(x)

    out = FakeQuantize.apply(x, scale, zp, n_bits)

    assert out.shape == x.shape
    # After dequantization every value must be a multiple of scale (0.01)
    residual = (out / 0.01).round() - out / 0.01
    assert residual.abs().max().item() < 1e-4


# ===========================================================================
# 2. FakeQuantize forward — output shape equals input shape
# ===========================================================================
def test_fake_quantize_forward_shape():
    x = torch.randn(4, 8, 16)
    scale = torch.ones_like(x) * 0.1
    zp = torch.zeros_like(x)
    out = FakeQuantize.apply(x, scale, zp, 8)
    assert out.shape == x.shape


# ===========================================================================
# 3. FakeQuantize backward — STE passes gradient through unchanged
# ===========================================================================
def test_fake_quantize_backward_ste():
    x = torch.randn(4, 8, requires_grad=True)
    scale = torch.ones_like(x) * 0.1
    zp = torch.zeros_like(x)

    out = FakeQuantize.apply(x, scale, zp, 8)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    # STE: gradient of sum w.r.t. x is all-ones
    assert torch.allclose(x.grad, torch.ones_like(x))


# ===========================================================================
# 4. FakeQuantize n_bits=4 has coarser quantization than n_bits=8
# ===========================================================================
def test_fake_quantize_coarser_at_4bits():
    torch.manual_seed(0)
    x = torch.randn(64)
    scale8 = torch.full_like(x, 0.05)
    scale4 = torch.full_like(x, 0.1)  # coarser scale for 4-bit
    zp = torch.zeros_like(x)

    out8 = FakeQuantize.apply(x, scale8, zp, 8)
    out4 = FakeQuantize.apply(x, scale4, zp, 4)

    err8 = (x - out8).abs().mean().item()
    err4 = (x - out4).abs().mean().item()

    # 4-bit should have higher reconstruction error than 8-bit
    assert err4 > err8


# ===========================================================================
# 5. PerTensorQuantizer — forward output same shape as input
# ===========================================================================
def test_per_tensor_quantizer_output_shape():
    q = PerTensorQuantizer(n_bits=8)
    x = torch.randn(4, 16)
    out = q(x)
    assert out.shape == x.shape


# ===========================================================================
# 6. PerTensorQuantizer — calibrate sets a reasonable (non-trivial) scale
# ===========================================================================
def test_per_tensor_quantizer_calibrate_sets_scale():
    q = PerTensorQuantizer(n_bits=8, symmetric=True)
    x = torch.randn(128) * 3.0  # std ~ 3
    q.calibrate(x)
    # scale should be > 0 and < some sane upper bound
    assert q.scale.item() > 1e-6
    assert q.scale.item() < 1.0  # 3 / 127 ~ 0.024


# ===========================================================================
# 7. PerTensorQuantizer — output values are grid-aligned to scale
# ===========================================================================
def test_per_tensor_quantizer_grid_aligned():
    q = PerTensorQuantizer(n_bits=8, symmetric=True)
    x = torch.randn(64) * 2.0
    out = q(x)
    scale = q.scale.item()
    # Each output value / scale should be (almost) integer
    residual = (out / scale).round() - out / scale
    assert residual.abs().max().item() < 1e-3


# ===========================================================================
# 8. PerChannelQuantizer — forward output shape unchanged
# ===========================================================================
def test_per_channel_quantizer_shape():
    C_out = 8
    q = PerChannelQuantizer(n_bits=8, num_channels=C_out)
    x = torch.randn(C_out, 16)
    out = q(x)
    assert out.shape == x.shape


# ===========================================================================
# 9. PerChannelQuantizer — different scales per channel
# ===========================================================================
def test_per_channel_quantizer_different_scales():
    C_out = 4
    # Give channels very different magnitude ranges
    x = torch.zeros(C_out, 32)
    for i in range(C_out):
        x[i] = torch.randn(32) * (i + 1) * 2.0

    q = PerChannelQuantizer(n_bits=8, symmetric=True, num_channels=C_out)
    q.calibrate(x)

    scales = q.scale
    assert scales.shape[0] == C_out
    # Scales should not all be equal (different channel magnitudes)
    assert not torch.allclose(scales[0:1], scales[-1:])


# ===========================================================================
# 10. QATLinear — forward output shape correct
# ===========================================================================
def test_qat_linear_output_shape():
    layer = QATLinear(in_features=16, out_features=8)
    x = torch.randn(4, 16)
    out = layer(x)
    assert out.shape == (4, 8)


# ===========================================================================
# 11. QATLinear — backward gradients flow to weight
# ===========================================================================
def test_qat_linear_backward_gradients():
    layer = QATLinear(in_features=16, out_features=8)
    x = torch.randn(4, 16)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert not layer.weight.grad.isnan().any()


# ===========================================================================
# 12. QATEmbedding — forward output shape [B, T, d_emb]
# ===========================================================================
def test_qat_embedding_output_shape():
    B, T, n_emb, d_emb = 4, 8, 16, 16
    emb = QATEmbedding(num_embeddings=n_emb, embedding_dim=d_emb)
    input_ids = torch.randint(0, n_emb, (B, T))
    out = emb(input_ids)
    assert out.shape == (B, T, d_emb)


# ===========================================================================
# 13. QATConverter — convert replaces Linear with QATLinear
# ===========================================================================
def test_qat_converter_replaces_linear():
    model = nn.Sequential(
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )
    converter = QATConverter()
    qat_model = converter.convert(model)

    # All Linear layers should be replaced
    for module in qat_model.modules():
        assert not isinstance(module, nn.Linear), \
            "Found unreplaced nn.Linear after conversion"
    # At least one QATLinear present
    assert any(isinstance(m, QATLinear) for m in qat_model.modules())


# ===========================================================================
# 14. QATConverter — converted model produces same-shape outputs
# ===========================================================================
def test_qat_converter_output_shape():
    model = TinyLM(vocab=32, d=16)
    converter = QATConverter()
    qat_model = converter.convert(model)

    B, T = 4, 8
    input_ids = torch.randint(0, 32, (B, T))

    with torch.no_grad():
        out_fp = model(input_ids)
        out_qat = qat_model(input_ids)

    assert out_fp.shape == out_qat.shape


# ===========================================================================
# 15. export_int_model — returns integer tensors
# ===========================================================================
def test_export_int_model_returns_int_tensors():
    model = TinyLM(vocab=32, d=16)
    converter = QATConverter()
    qat_model = converter.convert(model)

    # Trigger calibration via forward pass
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        qat_model(input_ids)

    int_weights = converter.export_int_model(qat_model)

    assert len(int_weights) > 0
    for key, tensor in int_weights.items():
        assert tensor.dtype in (torch.int8, torch.int32, torch.int64), \
            f"Expected integer dtype for {key}, got {tensor.dtype}"


# ===========================================================================
# 16. QuantizationTrainer — train_step returns finite loss
# ===========================================================================
def test_quantization_trainer_train_step_finite():
    vocab = 32
    model = TinyLM(vocab=vocab, d=16)
    converter = QATConverter()
    qat_model = converter.convert(model)

    trainer = QuantizationTrainer(qat_model, lr=1e-4, n_bits=8)

    B, T = 4, 8
    input_ids = torch.randint(0, vocab, (B, T))
    labels = torch.randint(0, vocab, (B, T))

    loss = trainer.train_step(input_ids, labels)

    assert loss.ndim == 0, "Loss should be scalar"
    assert torch.isfinite(loss), "Loss should be finite"


# ===========================================================================
# 17. QATConfig — defaults are correct
# ===========================================================================
def test_qat_config_defaults():
    cfg = QATConfig()
    assert cfg.n_bits_weight == 8
    assert cfg.n_bits_act == 8
    assert cfg.symmetric is True
    assert cfg.lr == 1e-4
    assert cfg.per_channel_weight is True


# ===========================================================================
# 18. PerTensorQuantizer — asymmetric mode calibrate & forward
# ===========================================================================
def test_per_tensor_quantizer_asymmetric():
    q = PerTensorQuantizer(n_bits=8, symmetric=False)
    # Positive-only data — asymmetric should use the full range
    x = torch.rand(64) * 5.0
    q.calibrate(x)
    out = q(x)
    assert out.shape == x.shape
    # All values should be non-negative after calibration on [0, 5]
    assert out.min().item() >= -1e-4


# ===========================================================================
# 19. QATConverter — original model unchanged after convert
# ===========================================================================
def test_qat_converter_does_not_modify_original():
    model = nn.Sequential(nn.Linear(16, 8))
    converter = QATConverter()
    _ = converter.convert(model)

    # Original model must still have nn.Linear
    assert any(isinstance(m, nn.Linear) for m in model.modules())


# ===========================================================================
# 20. QuantizationTrainer — quantization_error is non-negative float
# ===========================================================================
def test_quantization_trainer_error_non_negative():
    vocab = 32
    fp_model = TinyLM(vocab=vocab, d=16)
    converter = QATConverter()
    qat_model = converter.convert(fp_model)

    trainer = QuantizationTrainer(qat_model, lr=1e-4)

    B, T = 4, 8
    input_ids = torch.randint(0, vocab, (B, T))
    err = trainer.quantization_error(fp_model, qat_model, input_ids)

    assert isinstance(err, float)
    assert err >= 0.0
