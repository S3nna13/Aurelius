"""Integration test for AQLMLinear end-to-end."""

from __future__ import annotations

import torch

from src.model.aqlm_quant import AQLMConfig, AQLMLinear
from src.model import MODEL_COMPONENT_REGISTRY


def test_aqlm_integration_end_to_end():
    """Full integration: init -> quantize -> forward -> backward -> checks."""

    # --- Build layer ---
    cfg = AQLMConfig(
        in_features=16,
        out_features=8,
        n_codebooks=2,
        codebook_size=16,
        group_size=4,
    )
    layer = AQLMLinear(cfg)

    # --- Quantize a random weight ---
    torch.manual_seed(7)
    weight = torch.randn(8, 16)
    layer.quantize(weight)

    # codes shape: [out_features, n_groups, n_codebooks] = [8, 4, 2]
    assert layer.codes.shape == (8, 4, 2)
    assert layer.codes.long().min().item() >= 0
    assert layer.codes.long().max().item() < 16

    # --- Dequantize ---
    dequant = layer.dequantize()
    assert dequant.shape == (8, 16)
    assert torch.isfinite(dequant).all()

    # --- Forward pass with 3-D input [B=2, T=3, in_features=16] ---
    x = torch.randn(2, 3, 16)
    out = layer(x)
    assert out.shape == (2, 3, 8), f"Expected (2, 3, 8), got {tuple(out.shape)}"

    # --- Backward pass ---
    loss = out.sum()
    loss.backward()

    for m, cb in enumerate(layer.codebooks):
        assert cb.weight.grad is not None, f"Codebook {m} has no gradient"
        assert cb.weight.grad.abs().sum().item() > 0.0

    # --- Compression ratio ---
    ratio = layer.compression_ratio()
    assert ratio > 1.0, f"Expected compression_ratio > 1, got {ratio}"

    # --- Registry ---
    assert "aqlm_linear" in MODEL_COMPONENT_REGISTRY
    assert MODEL_COMPONENT_REGISTRY["aqlm_linear"] is AQLMLinear
