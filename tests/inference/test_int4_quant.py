"""Tests for src/inference/int4_quant.py — INT4 weight quantization."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.inference.int4_quant import (
    INT4Config,
    quantize_to_int4,
    pack_int4,
    unpack_int4,
    dequantize_int4,
    INT4Linear,
    convert_model_to_int4,
    estimate_int4_memory_savings,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

OUT_FEATURES = 16
IN_FEATURES = 32
GROUP_SIZE = 16
SEED = 0


def make_weight() -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randn(OUT_FEATURES, IN_FEATURES)


# ---------------------------------------------------------------------------
# 1. INT4Config defaults
# ---------------------------------------------------------------------------

def test_int4_config_defaults():
    cfg = INT4Config()
    assert cfg.group_size == 128
    assert cfg.symmetric is True
    assert cfg.pack_format == "row_major"
    assert cfg.compute_dtype == torch.float32


# ---------------------------------------------------------------------------
# 2. quantize_to_int4 — shapes
# ---------------------------------------------------------------------------

def test_quantize_to_int4_shapes():
    w = make_weight()
    weight_int4, scales, zero_points = quantize_to_int4(w, group_size=GROUP_SIZE, symmetric=True)

    n_groups = IN_FEATURES // GROUP_SIZE
    assert weight_int4.shape == (OUT_FEATURES, IN_FEATURES), (
        f"Expected ({OUT_FEATURES}, {IN_FEATURES}), got {weight_int4.shape}"
    )
    assert scales.shape == (OUT_FEATURES, n_groups), (
        f"Expected ({OUT_FEATURES}, {n_groups}), got {scales.shape}"
    )
    assert zero_points is None, "Symmetric mode should return None for zero_points"


# ---------------------------------------------------------------------------
# 3. quantize_to_int4 — value range
# ---------------------------------------------------------------------------

def test_quantize_to_int4_range():
    w = make_weight()

    # Symmetric: [-8, 7]
    w_sym, _, _ = quantize_to_int4(w, group_size=GROUP_SIZE, symmetric=True)
    assert w_sym.dtype == torch.int8
    assert w_sym.min().item() >= -8
    assert w_sym.max().item() <= 7

    # Asymmetric: [0, 15]
    w_asym, _, zp = quantize_to_int4(w, group_size=GROUP_SIZE, symmetric=False)
    assert w_asym.dtype == torch.int8
    assert w_asym.min().item() >= 0
    assert w_asym.max().item() <= 15
    assert zp is not None


# ---------------------------------------------------------------------------
# 4. pack_int4 — shape
# ---------------------------------------------------------------------------

def test_pack_int4_shape():
    w = make_weight()
    # Use asymmetric so values are already [0, 15]
    w_int4, _, _ = quantize_to_int4(w, group_size=GROUP_SIZE, symmetric=False)
    packed = pack_int4(w_int4)
    assert packed.shape == (OUT_FEATURES, IN_FEATURES // 2), (
        f"Expected ({OUT_FEATURES}, {IN_FEATURES // 2}), got {packed.shape}"
    )
    assert packed.dtype == torch.int8


# ---------------------------------------------------------------------------
# 5. unpack_int4 roundtrip
# ---------------------------------------------------------------------------

def test_unpack_int4_roundtrip():
    w = make_weight()
    w_int4, _, _ = quantize_to_int4(w, group_size=GROUP_SIZE, symmetric=False)
    packed = pack_int4(w_int4)
    unpacked = unpack_int4(packed, IN_FEATURES)

    assert unpacked.shape == (OUT_FEATURES, IN_FEATURES)
    assert torch.all(unpacked == w_int4), (
        "Roundtrip pack->unpack should recover original int4 values"
    )


# ---------------------------------------------------------------------------
# 6. dequantize_identity — quantize then dequantize ≈ original
# ---------------------------------------------------------------------------

def test_dequantize_identity():
    w = make_weight()
    w_int4, scales, zero_points = quantize_to_int4(w, group_size=GROUP_SIZE, symmetric=True)
    dequant = dequantize_int4(w_int4, scales, zero_points, group_size=GROUP_SIZE)

    assert dequant.shape == w.shape
    # Allow generous tolerance for 4-bit quantization error
    max_err = (dequant - w).abs().max().item()
    assert max_err < 1.0, f"Max dequant error too large: {max_err:.4f}"


# ---------------------------------------------------------------------------
# 7. INT4Linear.from_linear — forward output same shape
# ---------------------------------------------------------------------------

def test_int4_linear_from_linear_shape():
    torch.manual_seed(SEED)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    config = INT4Config(group_size=GROUP_SIZE)
    int4_layer = INT4Linear.from_linear(linear, config)

    x = torch.randn(4, IN_FEATURES)
    y = int4_layer(x)
    assert y.shape == (4, OUT_FEATURES), f"Expected (4, {OUT_FEATURES}), got {y.shape}"


# ---------------------------------------------------------------------------
# 8. INT4Linear forward dtype
# ---------------------------------------------------------------------------

def test_int4_linear_forward_dtype():
    torch.manual_seed(SEED)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    config = INT4Config(group_size=GROUP_SIZE, compute_dtype=torch.float32)
    int4_layer = INT4Linear.from_linear(linear, config)

    x = torch.randn(4, IN_FEATURES)
    y = int4_layer(x)
    assert y.dtype == torch.float32, f"Expected float32, got {y.dtype}"


# ---------------------------------------------------------------------------
# 9. INT4Linear forward ≈ original (atol=0.5)
# ---------------------------------------------------------------------------

def test_int4_linear_close_to_original():
    torch.manual_seed(SEED)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    config = INT4Config(group_size=GROUP_SIZE)
    int4_layer = INT4Linear.from_linear(linear, config)

    torch.manual_seed(SEED + 1)
    x = torch.randn(4, IN_FEATURES)

    with torch.no_grad():
        y_orig = linear(x)
        y_int4 = int4_layer(x)

    max_diff = (y_orig - y_int4).abs().max().item()
    assert max_diff < 0.5, f"INT4 output differs too much from original: max diff = {max_diff:.4f}"


# ---------------------------------------------------------------------------
# 10. convert_model_to_int4 replaces linear layers
# ---------------------------------------------------------------------------

def test_convert_model_to_int4_replaces_linears():
    torch.manual_seed(SEED)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
            self.fc2 = nn.Linear(OUT_FEATURES, IN_FEATURES, bias=False)
            self.lm_head = nn.Linear(IN_FEATURES, 10, bias=False)

    model = TinyModel()
    config = INT4Config(group_size=GROUP_SIZE)
    convert_model_to_int4(model, config, skip_layers=["lm_head"])

    assert isinstance(model.fc1, INT4Linear), "fc1 should be converted to INT4Linear"
    assert isinstance(model.fc2, INT4Linear), "fc2 should be converted to INT4Linear"
    # lm_head should be skipped
    assert isinstance(model.lm_head, nn.Linear), "lm_head should not be converted (skipped)"


# ---------------------------------------------------------------------------
# 11. estimate_memory_savings — correct keys
# ---------------------------------------------------------------------------

def test_estimate_memory_savings_keys():
    torch.manual_seed(SEED)
    model = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    result = estimate_int4_memory_savings(model)

    assert "float32_mb" in result
    assert "int4_mb" in result
    assert "reduction_factor" in result


# ---------------------------------------------------------------------------
# 12. estimate_memory_savings — reduction_factor > 1.0
# ---------------------------------------------------------------------------

def test_estimate_memory_savings_reduction():
    torch.manual_seed(SEED)
    model = nn.Sequential(
        nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False),
        nn.Linear(OUT_FEATURES, IN_FEATURES, bias=False),
    )
    result = estimate_int4_memory_savings(model)

    assert result["float32_mb"] > result["int4_mb"], (
        "float32 should be larger than int4 representation"
    )
    assert result["reduction_factor"] > 1.0, (
        f"reduction_factor should be > 1.0, got {result['reduction_factor']}"
    )
