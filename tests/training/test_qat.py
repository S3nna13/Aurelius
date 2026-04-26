"""Tests for quantization-aware training."""

import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.qat import (
    QATConfig,
    QATLinear,
    _StraightThroughRound,
    convert_qat,
    fake_quantize_int4,
    fake_quantize_int8,
    prepare_qat,
)


def test_ste_round_forward():
    """STE round must round values in forward pass."""
    x = torch.tensor([0.4, 0.6, 1.3, -0.7])
    y = _StraightThroughRound.apply(x)
    assert torch.allclose(y, torch.tensor([0.0, 1.0, 1.0, -1.0]))


def test_ste_round_backward():
    """STE round gradient must be identity (not zero)."""
    x = torch.tensor([0.4, 0.6, 1.3], requires_grad=True)
    y = _StraightThroughRound.apply(x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones(3))


def test_fake_quantize_int8_shape():
    """fake_quantize_int8 must return same shape as input."""
    w = torch.randn(32, 64, requires_grad=True)
    w_fq = fake_quantize_int8(w, per_channel=True)
    assert w_fq.shape == (32, 64)
    # Gradient flows back through STE
    w_fq.sum().backward()
    assert w.grad is not None


def test_fake_quantize_int4_shape():
    """fake_quantize_int4 must return same shape as input."""
    w = torch.randn(16, 128, requires_grad=True)
    w_fq = fake_quantize_int4(w, group_size=128)
    assert w_fq.shape == (16, 128)
    w_fq.sum().backward()
    assert w.grad is not None


def test_qat_linear_forward():
    """QATLinear forward must return correct output shape."""
    w = torch.randn(32, 64)
    q = QATLinear(w, bias=None, cfg=QATConfig(bits=8))
    x = torch.randn(2, 10, 64)
    out = q(x)
    assert out.shape == (2, 10, 32)
    assert torch.isfinite(out).all()


def test_prepare_qat_replaces_linears():
    """prepare_qat must replace nn.Linear with QATLinear."""
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    model = AureliusTransformer(cfg)
    prepare_qat(model, QATConfig(bits=8))

    n_qat = sum(1 for m in model.modules() if isinstance(m, QATLinear))
    assert n_qat > 0


def test_qat_gradient_flows():
    """QAT forward+backward must produce gradients through fake quantization."""
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    model = AureliusTransformer(cfg)
    prepare_qat(model, QATConfig(bits=8))

    ids = torch.randint(0, 256, (2, 16))
    loss, _, _ = model(ids, labels=ids)
    loss.backward()

    # QATLinear weights must have gradients
    has_grad = any(isinstance(m, QATLinear) and m.weight.grad is not None for m in model.modules())
    assert has_grad, "No gradients flowed through QATLinear layers"


def test_convert_qat():
    """convert_qat must replace QATLinear with QuantizedLinear."""
    from src.inference.quantize import QuantizedLinear

    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    model = AureliusTransformer(cfg)
    prepare_qat(model, QATConfig(bits=8))
    convert_qat(model)

    n_quantized = sum(1 for m in model.modules() if isinstance(m, QuantizedLinear))
    assert n_quantized > 0
    # No more QATLinear
    n_qat = sum(1 for m in model.modules() if isinstance(m, QATLinear))
    assert n_qat == 0
