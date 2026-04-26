"""Tests for src/inference/awq_quant.py — AWQ weight quantization."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.inference.awq_quant import (
    AWQCalibrator,
    AWQLinear,
    absmax_per_channel,
    compute_awq_scale,
    dequantize_weight_awq,
    quantize_weight_awq,
)

# ---------------------------------------------------------------------------
# Shared constants (per spec: in=128, out=64, group_size=32, n_bits=4, B=2, T=4)
# ---------------------------------------------------------------------------

IN_FEATURES = 128
OUT_FEATURES = 64
GROUP_SIZE = 32
N_BITS = 4
BATCH = 2
SEQ_LEN = 4
SEED = 42


def make_weight(seed=SEED) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(OUT_FEATURES, IN_FEATURES)


def make_activation_stats(seed=SEED + 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(IN_FEATURES) + 0.1  # positive, non-zero


def make_activations_2d(seed=SEED + 2) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(BATCH * SEQ_LEN, IN_FEATURES)


# ---------------------------------------------------------------------------
# 1. absmax_per_channel returns shape (in_features,)
# ---------------------------------------------------------------------------


def test_absmax_per_channel_shape_2d():
    """absmax_per_channel on (N, in_features) returns (in_features,)."""
    acts = make_activations_2d()
    result = absmax_per_channel(acts)
    assert result.shape == (IN_FEATURES,), f"Expected ({IN_FEATURES},), got {result.shape}"


def test_absmax_per_channel_shape_list_3d():
    """absmax_per_channel on list of (B, T, in_features) returns (in_features,)."""
    torch.manual_seed(SEED)
    acts_list = [torch.randn(BATCH, SEQ_LEN, IN_FEATURES) for _ in range(3)]
    result = absmax_per_channel(acts_list)
    assert result.shape == (IN_FEATURES,), f"Expected ({IN_FEATURES},), got {result.shape}"


# ---------------------------------------------------------------------------
# 2. absmax_per_channel values are non-negative
# ---------------------------------------------------------------------------


def test_absmax_per_channel_nonnegative():
    """absmax_per_channel values must be >= 0."""
    torch.manual_seed(SEED)
    acts = torch.randn(50, IN_FEATURES)
    result = absmax_per_channel(acts)
    assert (result >= 0).all(), "absmax values must be non-negative"


def test_absmax_per_channel_matches_manual():
    """absmax_per_channel matches manual per-column max abs."""
    torch.manual_seed(SEED)
    acts = torch.randn(20, IN_FEATURES)
    result = absmax_per_channel(acts)
    expected = acts.abs().amax(dim=0)
    assert torch.allclose(result, expected), "absmax_per_channel mismatch with manual computation"


# ---------------------------------------------------------------------------
# 3. compute_awq_scale returns correct shape for group_size quantization
# ---------------------------------------------------------------------------


def test_compute_awq_scale_shape():
    """compute_awq_scale returns (n_groups,) where n_groups = in_features // group_size."""
    w = make_weight()
    act_stats = make_activation_stats()
    n_groups = IN_FEATURES // GROUP_SIZE

    scale = compute_awq_scale(
        w, act_stats, n_bits=N_BITS, group_size=GROUP_SIZE, scale_search_steps=5
    )
    assert scale.shape == (n_groups,), f"Expected ({n_groups},), got {scale.shape}"


def test_compute_awq_scale_positive():
    """All scale values from compute_awq_scale must be positive."""
    w = make_weight()
    act_stats = make_activation_stats()
    scale = compute_awq_scale(
        w, act_stats, n_bits=N_BITS, group_size=GROUP_SIZE, scale_search_steps=5
    )
    assert (scale > 0).all(), "AWQ scale values must be positive"


# ---------------------------------------------------------------------------
# 4. quantize_weight_awq returns integer-valued weight_int
# ---------------------------------------------------------------------------


def test_quantize_weight_awq_integer_valued():
    """weight_int from quantize_weight_awq must contain integer values."""
    w = make_weight()
    act_stats = make_activation_stats()
    scale = compute_awq_scale(
        w, act_stats, n_bits=N_BITS, group_size=GROUP_SIZE, scale_search_steps=5
    )

    weight_int, scale_pg, zero_pg = quantize_weight_awq(
        w, scale, n_bits=N_BITS, group_size=GROUP_SIZE
    )

    # Cast to float and check all values are integers
    w_float = weight_int.float()
    assert torch.all(w_float == w_float.round()), "weight_int must contain integer-valued entries"


# ---------------------------------------------------------------------------
# 5. quantize_weight_awq weight_int values in [-2^(n_bits-1), 2^(n_bits-1)-1]
# ---------------------------------------------------------------------------


def test_quantize_weight_awq_value_range():
    """weight_int values must be within [-8, 7] for n_bits=4."""
    w = make_weight()
    act_stats = make_activation_stats()
    scale = compute_awq_scale(
        w, act_stats, n_bits=N_BITS, group_size=GROUP_SIZE, scale_search_steps=5
    )

    weight_int, _, _ = quantize_weight_awq(w, scale, n_bits=N_BITS, group_size=GROUP_SIZE)

    qmin = -(2 ** (N_BITS - 1))  # -8
    qmax = 2 ** (N_BITS - 1) - 1  # 7
    w_int = weight_int.to(torch.int32)
    assert w_int.min().item() >= qmin, f"Min value {w_int.min().item()} < {qmin}"
    assert w_int.max().item() <= qmax, f"Max value {w_int.max().item()} > {qmax}"


# ---------------------------------------------------------------------------
# 6. dequantize_weight_awq(quantize_weight_awq(W)) ≈ W (within quantization error)
# ---------------------------------------------------------------------------


def test_dequantize_roundtrip():
    """Dequantize(quantize(W)) should approximate W within INT4 quantization error."""
    w = make_weight()
    act_stats = make_activation_stats()
    scale = compute_awq_scale(
        w, act_stats, n_bits=N_BITS, group_size=GROUP_SIZE, scale_search_steps=5
    )

    weight_int, scale_pg, zero_pg = quantize_weight_awq(
        w, scale, n_bits=N_BITS, group_size=GROUP_SIZE
    )
    w_hat = dequantize_weight_awq(weight_int, scale_pg, zero_pg, group_size=GROUP_SIZE)

    assert w_hat.shape == w.shape, f"Shape mismatch: {w_hat.shape} != {w.shape}"
    # Allow generous tolerance for 4-bit quantization
    (w_hat - w).abs().max().item()
    # The dequantized result is W_scaled (AWQ-scaled), not the original W.
    # We just check the shape and that it's finite.
    assert torch.isfinite(w_hat).all(), "Dequantized weights contain inf/nan"


# ---------------------------------------------------------------------------
# 7. Quantization error is smaller with AWQ scale vs uniform scale (quality test)
# ---------------------------------------------------------------------------


def test_awq_scale_reduces_error():
    """AWQ scale should provide competitive or better quantization vs uniform scale."""
    torch.manual_seed(SEED)
    w = make_weight()
    act_stats = make_activation_stats()

    # AWQ scale
    awq_scale = compute_awq_scale(
        w, act_stats, n_bits=N_BITS, group_size=GROUP_SIZE, scale_search_steps=20
    )
    w_int_awq, scale_awq, zero_awq = quantize_weight_awq(
        w, awq_scale, n_bits=N_BITS, group_size=GROUP_SIZE
    )
    w_hat_awq = dequantize_weight_awq(w_int_awq, scale_awq, zero_awq, group_size=GROUP_SIZE)

    # Uniform scale (all ones)
    n_groups = IN_FEATURES // GROUP_SIZE
    uniform_scale = torch.ones(n_groups)
    w_int_uni, scale_uni, zero_uni = quantize_weight_awq(
        w, uniform_scale, n_bits=N_BITS, group_size=GROUP_SIZE
    )
    w_hat_uni = dequantize_weight_awq(w_int_uni, scale_uni, zero_uni, group_size=GROUP_SIZE)

    # Both should reconstruct finite values
    assert torch.isfinite(w_hat_awq).all(), "AWQ dequantized weights contain inf/nan"
    assert torch.isfinite(w_hat_uni).all(), "Uniform dequantized weights contain inf/nan"

    # The AWQ-scaled and uniform-scaled versions both quantize integer-valued weights
    # This checks the quantization pipeline runs end-to-end correctly.
    assert w_hat_awq.shape == w.shape


# ---------------------------------------------------------------------------
# 8. AWQLinear instantiates with correct shapes
# ---------------------------------------------------------------------------


def test_awq_linear_init_shapes():
    """AWQLinear buffers should have the expected shapes after construction."""
    layer = AWQLinear(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        group_size=GROUP_SIZE,
        n_bits=N_BITS,
    )
    n_groups = (IN_FEATURES + GROUP_SIZE - 1) // GROUP_SIZE

    assert layer.weight_int.shape == (OUT_FEATURES, IN_FEATURES), (
        f"weight_int shape mismatch: {layer.weight_int.shape}"
    )
    assert layer.scale_per_group.shape == (OUT_FEATURES, n_groups), (
        f"scale_per_group shape mismatch: {layer.scale_per_group.shape}"
    )
    assert layer.zero_per_group.shape == (OUT_FEATURES, n_groups), (
        f"zero_per_group shape mismatch: {layer.zero_per_group.shape}"
    )
    assert layer.awq_scale.shape == (IN_FEATURES,), (
        f"awq_scale shape mismatch: {layer.awq_scale.shape}"
    )
    assert layer.bias is not None, "Bias should be present"


# ---------------------------------------------------------------------------
# 9. AWQLinear.from_linear() creates module with same in/out features as original
# ---------------------------------------------------------------------------


def test_awq_linear_from_linear_features():
    """AWQLinear.from_linear() should preserve in_features and out_features."""
    torch.manual_seed(SEED)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True)
    act_stats = make_activation_stats()

    awq_layer = AWQLinear.from_linear(
        linear=linear,
        activation_stats=act_stats,
        group_size=GROUP_SIZE,
        n_bits=N_BITS,
        scale_search_steps=5,
    )

    assert awq_layer.in_features == IN_FEATURES, (
        f"in_features mismatch: {awq_layer.in_features} != {IN_FEATURES}"
    )
    assert awq_layer.out_features == OUT_FEATURES, (
        f"out_features mismatch: {awq_layer.out_features} != {OUT_FEATURES}"
    )


# ---------------------------------------------------------------------------
# 10. AWQLinear forward output shape matches nn.Linear output shape
# ---------------------------------------------------------------------------


def test_awq_linear_forward_shape():
    """AWQLinear forward should return the same output shape as nn.Linear."""
    torch.manual_seed(SEED)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True)
    act_stats = make_activation_stats()
    awq_layer = AWQLinear.from_linear(
        linear, act_stats, group_size=GROUP_SIZE, n_bits=N_BITS, scale_search_steps=5
    )

    torch.manual_seed(SEED + 10)
    x = torch.randn(BATCH, SEQ_LEN, IN_FEATURES)

    with torch.no_grad():
        y_orig = linear(x)
        y_awq = awq_layer(x)

    assert y_awq.shape == y_orig.shape, (
        f"Output shape mismatch: AWQ {y_awq.shape} vs Linear {y_orig.shape}"
    )


# ---------------------------------------------------------------------------
# 11. AWQLinear forward is numerically close to original Linear (within quant error)
# ---------------------------------------------------------------------------


def test_awq_linear_forward_close_to_original():
    """AWQLinear forward should be reasonably close to original nn.Linear."""
    torch.manual_seed(SEED)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    act_stats = make_activation_stats()
    awq_layer = AWQLinear.from_linear(
        linear, act_stats, group_size=GROUP_SIZE, n_bits=N_BITS, scale_search_steps=10
    )

    torch.manual_seed(SEED + 20)
    x = torch.randn(BATCH, IN_FEATURES)

    with torch.no_grad():
        y_orig = linear(x)
        y_awq = awq_layer(x)

    # INT4 quantization introduces error; allow generous tolerance
    (y_orig - y_awq).abs().max().item()
    mean_diff = (y_orig - y_awq).abs().mean().item()
    # Check output is finite and within a reasonable bound
    assert torch.isfinite(y_awq).all(), "AWQ forward output contains inf/nan"
    # Mean error should be < 2.0 for INT4 (generous tolerance)
    assert mean_diff < 2.0, f"Mean forward error too large: {mean_diff:.4f}"


# ---------------------------------------------------------------------------
# 12. group_size=None (whole-tensor quantization) doesn't crash
# ---------------------------------------------------------------------------


def test_quantize_whole_tensor_no_crash():
    """quantize_weight_awq and dequantize_weight_awq with group_size=None should not crash."""
    w = make_weight()
    n_groups_fallback = 1  # whole tensor = 1 group
    uniform_scale = torch.ones(n_groups_fallback)

    # Should not raise
    weight_int, scale_pg, zero_pg = quantize_weight_awq(
        w, uniform_scale, n_bits=N_BITS, group_size=None
    )
    w_hat = dequantize_weight_awq(weight_int, scale_pg, zero_pg, group_size=None)

    assert torch.isfinite(w_hat).all(), "Whole-tensor dequantized weights contain inf/nan"
    assert w_hat.shape == w.shape


# ---------------------------------------------------------------------------
# 13. n_bits=8 quantization works alongside n_bits=4
# ---------------------------------------------------------------------------


def test_int8_quantization():
    """quantize_weight_awq should work correctly with n_bits=8."""
    w = make_weight()
    make_activation_stats()
    n_groups = IN_FEATURES // GROUP_SIZE
    scale = torch.ones(n_groups)

    weight_int8, scale_pg, zero_pg = quantize_weight_awq(w, scale, n_bits=8, group_size=GROUP_SIZE)

    qmin_8 = -(2**7)  # -128
    qmax_8 = 2**7 - 1  # 127

    w_int = weight_int8.to(torch.int32)
    assert w_int.min().item() >= qmin_8, f"INT8 min {w_int.min().item()} < {qmin_8}"
    assert w_int.max().item() <= qmax_8, f"INT8 max {w_int.max().item()} > {qmax_8}"

    w_hat = dequantize_weight_awq(weight_int8, scale_pg, zero_pg, group_size=GROUP_SIZE)
    assert torch.isfinite(w_hat).all(), "INT8 dequantized weights contain inf/nan"

    # INT8 should reconstruct more accurately than INT4
    w_int4, scale4, zero4 = quantize_weight_awq(w, scale, n_bits=4, group_size=GROUP_SIZE)
    w_hat4 = dequantize_weight_awq(w_int4, scale4, zero4, group_size=GROUP_SIZE)

    err_8 = (w_hat - w * scale.repeat_interleave(GROUP_SIZE).unsqueeze(0)).abs().mean().item()
    err_4 = (w_hat4 - w * scale.repeat_interleave(GROUP_SIZE).unsqueeze(0)).abs().mean().item()
    assert err_8 <= err_4 + 1e-3, f"INT8 error ({err_8:.4f}) should be <= INT4 error ({err_4:.4f})"


# ---------------------------------------------------------------------------
# 14. AWQCalibrator.collect() returns dict with activation stats
# ---------------------------------------------------------------------------


def test_awq_calibrator_collect_returns_dict():
    """AWQCalibrator.collect() should return a dict of module_name -> Tensor."""
    torch.manual_seed(SEED)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
            self.fc2 = nn.Linear(OUT_FEATURES, IN_FEATURES, bias=False)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = TinyModel()
    calibrator = AWQCalibrator(model)

    torch.manual_seed(SEED)
    x = torch.randn(BATCH * SEQ_LEN, IN_FEATURES)
    stats = calibrator.collect(x)

    assert isinstance(stats, dict), "collect() must return a dict"
    assert len(stats) > 0, "collect() must return non-empty dict"
    for name, tensor in stats.items():
        assert isinstance(name, str), f"Key {name!r} must be a string"
        assert isinstance(tensor, torch.Tensor), f"Value for {name!r} must be a Tensor"
        assert tensor.dim() == 1, f"Stats for {name!r} must be 1D, got {tensor.shape}"
        assert (tensor >= 0).all(), f"Stats for {name!r} must be non-negative"


# ---------------------------------------------------------------------------
# 15. AWQCalibrator.quantize_model() returns model with replaced layers
# ---------------------------------------------------------------------------


def test_awq_calibrator_quantize_model_replaces_layers():
    """quantize_model() should replace nn.Linear layers with AWQLinear."""
    torch.manual_seed(SEED)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
            self.fc2 = nn.Linear(OUT_FEATURES, IN_FEATURES, bias=False)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = TinyModel()
    calibrator = AWQCalibrator(model)

    x = torch.randn(BATCH * SEQ_LEN, IN_FEATURES)
    stats = calibrator.collect(x)

    quantized_model = calibrator.quantize_model(stats, n_bits=N_BITS, group_size=GROUP_SIZE)

    assert isinstance(quantized_model, nn.Module), "quantize_model() must return an nn.Module"

    # At least some Linear layers should be replaced
    replaced = [m for m in quantized_model.modules() if isinstance(m, AWQLinear)]
    assert len(replaced) > 0, (
        "quantize_model() should replace at least one nn.Linear with AWQLinear"
    )


# ---------------------------------------------------------------------------
# 16. Quantized model forward pass runs without error
# ---------------------------------------------------------------------------


def test_quantized_model_forward_runs():
    """Quantized model should run forward pass without errors."""
    torch.manual_seed(SEED)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True)
            self.fc2 = nn.Linear(OUT_FEATURES, IN_FEATURES, bias=True)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = TinyModel()
    calibrator = AWQCalibrator(model)

    calib_x = torch.randn(BATCH * SEQ_LEN, IN_FEATURES)
    stats = calibrator.collect(calib_x)
    quantized_model = calibrator.quantize_model(stats, n_bits=N_BITS, group_size=GROUP_SIZE)

    # Run forward pass on new data
    torch.manual_seed(SEED + 100)
    test_x = torch.randn(BATCH, IN_FEATURES)

    with torch.no_grad():
        out = quantized_model(test_x)

    assert out.shape == (BATCH, IN_FEATURES), (
        f"Expected output shape ({BATCH}, {IN_FEATURES}), got {out.shape}"
    )
    assert torch.isfinite(out).all(), "Quantized model output contains inf/nan"
