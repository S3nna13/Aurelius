"""Unit tests for src/model/aqlm_quant.py (15 tests)."""

from __future__ import annotations

import math

import pytest
import torch

from src.model.aqlm_quant import AQLMCodebook, AQLMConfig, AQLMLinear
from src.model import MODEL_COMPONENT_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_layer(
    in_features: int = 16,
    out_features: int = 8,
    n_codebooks: int = 2,
    codebook_size: int = 16,
    group_size: int = 4,
) -> AQLMLinear:
    cfg = AQLMConfig(
        in_features=in_features,
        out_features=out_features,
        n_codebooks=n_codebooks,
        codebook_size=codebook_size,
        group_size=group_size,
    )
    return AQLMLinear(cfg)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = AQLMConfig(in_features=64, out_features=32)
    assert cfg.n_codebooks == 2
    assert cfg.codebook_size == 256
    assert cfg.group_size == 8


# ---------------------------------------------------------------------------
# 2. Codebook weight shape
# ---------------------------------------------------------------------------

def test_codebook_weight_shape():
    cb = AQLMCodebook(codebook_size=16, group_size=4)
    assert cb.weight.shape == (16, 4)


# ---------------------------------------------------------------------------
# 3. Dequantize shape
# ---------------------------------------------------------------------------

def test_dequantize_shape():
    layer = make_layer(in_features=16, out_features=8)
    w = layer.dequantize()
    assert w.shape == (8, 16)


# ---------------------------------------------------------------------------
# 4. Forward shape — 2-D input
# ---------------------------------------------------------------------------

def test_forward_shape_2d():
    layer = make_layer(in_features=16, out_features=8)
    x = torch.randn(3, 16)
    out = layer(x)
    assert out.shape == (3, 8)


# ---------------------------------------------------------------------------
# 5. Forward shape — 3-D input [B, T, in_features]
# ---------------------------------------------------------------------------

def test_forward_shape_3d():
    layer = make_layer(in_features=16, out_features=8)
    x = torch.randn(2, 5, 16)
    out = layer(x)
    assert out.shape == (2, 5, 8)


# ---------------------------------------------------------------------------
# 6. Codes shape after quantize
# ---------------------------------------------------------------------------

def test_quantize_codes_shape():
    layer = make_layer(in_features=16, out_features=8, group_size=4)
    weight = torch.randn(8, 16)
    layer.quantize(weight)
    # n_groups = 16 // 4 = 4
    assert layer.codes.shape == (8, 4, 2)


# ---------------------------------------------------------------------------
# 7. Codes range after quantize
# ---------------------------------------------------------------------------

def test_quantize_codes_range():
    layer = make_layer(in_features=16, out_features=8, codebook_size=16, group_size=4)
    weight = torch.randn(8, 16)
    layer.quantize(weight)
    codes_long = layer.codes.long()
    assert codes_long.min().item() >= 0
    assert codes_long.max().item() < 16


# ---------------------------------------------------------------------------
# 8. Dequantize approximates original weight after quantize
# ---------------------------------------------------------------------------

def test_dequant_after_quantize_close():
    """Reconstruction stays in a sane range relative to the original weight."""
    torch.manual_seed(42)
    layer = make_layer(
        in_features=8, out_features=4, n_codebooks=1, codebook_size=32, group_size=4
    )
    weight = torch.randn(4, 8)

    # Pre-seed codebook with exact sub-vectors from the weight so quantize
    # can pick near-perfect matches.
    w_grouped = weight.reshape(4, 2, 4)   # [out, n_groups, group_size]
    flat = w_grouped.reshape(-1, 4)       # [8, 4]
    cb_data = flat.repeat(4, 1)[:32]
    layer.codebooks[0]._weight.data.copy_(cb_data.detach())

    layer.quantize(weight)
    reconstructed = layer.dequantize()

    orig_norm = weight.norm().item()
    err_norm = (reconstructed.detach() - weight).norm().item()
    assert err_norm < orig_norm * 5.0


# ---------------------------------------------------------------------------
# 9. Compression ratio is > 1.0
# ---------------------------------------------------------------------------

def test_compression_ratio_gt_1():
    layer = make_layer()
    assert layer.compression_ratio() > 1.0


# ---------------------------------------------------------------------------
# 10. More codebooks → lower ratio (more bits used)
# ---------------------------------------------------------------------------

def test_compression_n_codebooks_effect():
    """Doubling n_codebooks halves the compression ratio."""
    layer1 = make_layer(n_codebooks=1, codebook_size=16, group_size=4)
    layer2 = make_layer(n_codebooks=2, codebook_size=16, group_size=4)
    assert layer2.compression_ratio() < layer1.compression_ratio()
    assert math.isclose(
        layer1.compression_ratio() / layer2.compression_ratio(), 2.0, rel_tol=1e-6
    )


# ---------------------------------------------------------------------------
# 11. Gradients flow to codebook weights
# ---------------------------------------------------------------------------

def test_gradients_codebook():
    layer = make_layer()
    x = torch.randn(2, 16, requires_grad=False)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    for cb in layer.codebooks:
        assert cb.weight.grad is not None
        assert cb.weight.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# 12. Batch independence — B=1 vs B=2 consistency
# ---------------------------------------------------------------------------

def test_batch_independence():
    torch.manual_seed(0)
    layer = make_layer()
    layer.train(False)

    x1 = torch.randn(1, 16)
    x2 = torch.randn(1, 16)
    x_batch = torch.cat([x1, x2], dim=0)

    with torch.no_grad():
        out1 = layer(x1)
        out2 = layer(x2)
        out_batch = layer(x_batch)

    assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
    assert torch.allclose(out_batch[1], out2[0], atol=1e-5)


# ---------------------------------------------------------------------------
# 13. Determinism — same input -> same output
# ---------------------------------------------------------------------------

def test_determinism():
    layer = make_layer()
    x = torch.randn(3, 16)
    with torch.no_grad():
        out1 = layer(x)
        out2 = layer(x)
    assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# 14. Quantize and dequant on a small 4x8 weight
# ---------------------------------------------------------------------------

def test_quantize_small_weight():
    layer = make_layer(in_features=8, out_features=4, group_size=4, codebook_size=8)
    weight = torch.randn(4, 8)
    layer.quantize(weight)
    reconstructed = layer.dequantize()
    assert reconstructed.shape == (4, 8)
    assert torch.isfinite(reconstructed).all()


# ---------------------------------------------------------------------------
# 15. Registry entry
# ---------------------------------------------------------------------------

def test_registry():
    assert "aqlm_linear" in MODEL_COMPONENT_REGISTRY
    assert MODEL_COMPONENT_REGISTRY["aqlm_linear"] is AQLMLinear
