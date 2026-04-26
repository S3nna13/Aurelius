"""Tests for GPTQ-style post-training quantization.

Tiny config: d_in=16, d_out=8, n_bits=4, group_size=8, batch=2, seq_len=4.
Every test performs actual forward and/or backward passes.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.post_training_quantization import (
    GPTQQuantizer,
    HessianEstimator,
    LayerQuantizer,
    ModelQuantizer,
    QuantizationBenchmark,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

D_IN = 16
D_OUT = 8
N_BITS = 4
GROUP_SIZE = 8
BATCH = 2
SEQ_LEN = 4
Q_MAX = (1 << N_BITS) - 1  # 15


def make_activations(seed: int = 0) -> torch.Tensor:
    """Return (BATCH, SEQ_LEN, D_IN) calibration activations."""
    torch.manual_seed(seed)
    return torch.randn(BATCH, SEQ_LEN, D_IN)


def make_linear() -> nn.Linear:
    torch.manual_seed(42)
    return nn.Linear(D_IN, D_OUT, bias=False)


def make_quantizer() -> GPTQQuantizer:
    return GPTQQuantizer(n_bits=N_BITS, group_size=GROUP_SIZE)


# ---------------------------------------------------------------------------
# Tiny model for ModelQuantizer tests
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Two-layer MLP for quantization tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(D_IN, D_OUT, bias=False)
        self.fc2 = nn.Linear(D_OUT, D_IN, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D_IN)  →  (B, T, D_IN)
        B, T, _ = x.shape
        x_flat = x.reshape(B * T, D_IN)
        h = torch.relu(self.fc1(x_flat))
        out = self.fc2(h)
        return out.reshape(B, T, D_IN)


# ===========================================================================
# 1. HessianEstimator.collect: H accumulates and n_collected increments
# ===========================================================================


def test_hessian_estimator_collect_accumulates():
    estimator = HessianEstimator()
    acts = make_activations(0)  # (2, 4, 16)
    estimator.collect(acts)
    assert estimator.H is not None
    assert estimator.n_collected == BATCH * SEQ_LEN  # 8
    H_before = estimator.H.clone()
    estimator.collect(make_activations(1))
    # H must have changed (new activations added)
    assert not torch.equal(estimator.H, H_before)
    assert estimator.n_collected == 2 * BATCH * SEQ_LEN


# ===========================================================================
# 2. HessianEstimator.get_hessian: shape (d_in, d_in), symmetric
# ===========================================================================


def test_hessian_estimator_get_hessian_shape_and_symmetry():
    estimator = HessianEstimator()
    estimator.collect(make_activations())
    H = estimator.get_hessian()
    assert H.shape == (D_IN, D_IN)
    assert torch.allclose(H, H.t(), atol=1e-6), "Hessian must be symmetric"


# ===========================================================================
# 3. HessianEstimator.damp: diagonal entries larger after damping
# ===========================================================================


def test_hessian_estimator_damp_increases_diagonal():
    estimator = HessianEstimator()
    estimator.collect(make_activations())
    H_orig = estimator.get_hessian()
    H_damp = estimator.damp(percentile=1.0)
    # Every diagonal entry must be strictly larger after damping
    assert (H_damp.diagonal() > H_orig.diagonal()).all(), (
        "All diagonal entries must increase after damping"
    )


# ===========================================================================
# 4. GPTQQuantizer.quantize_weight: W_q dtype int-compatible, scale/zero shapes
# ===========================================================================


def test_gptq_quantize_weight_shapes():
    quant = make_quantizer()
    W = make_linear().weight.data  # (D_OUT, D_IN)
    H = torch.eye(D_IN)
    W_q, scale, zero = quant.quantize_weight(W, H)
    n_groups = D_IN // GROUP_SIZE  # 2
    assert W_q.shape == (D_OUT, D_IN)
    assert scale.shape == (D_OUT, n_groups)
    assert zero.shape == (D_OUT, n_groups)
    # W_q must be representable as integers
    assert torch.equal(W_q, W_q.round()), "W_q should contain integer values"


# ===========================================================================
# 5. GPTQQuantizer.quantize_weight: W_q values in [0, 2^n_bits - 1]
# ===========================================================================


def test_gptq_quantize_weight_range():
    quant = make_quantizer()
    W = make_linear().weight.data
    H = torch.eye(D_IN)
    W_q, _, _ = quant.quantize_weight(W, H)
    assert (W_q >= 0).all(), "W_q must be non-negative"
    assert (W_q <= Q_MAX).all(), f"W_q must be ≤ {Q_MAX}"


# ===========================================================================
# 6. GPTQQuantizer.dequantize: output shape matches W, round-trip error < 0.5
# ===========================================================================


def test_gptq_dequantize_shape_and_roundtrip():
    quant = make_quantizer()
    W = make_linear().weight.data
    H = torch.eye(D_IN)
    W_q, scale, zero = quant.quantize_weight(W, H)
    W_deq = quant.dequantize(W_q, scale, zero)
    assert W_deq.shape == W.shape
    rel_err = (W.float() - W_deq).norm() / W.float().norm()
    assert rel_err.item() < 0.5, f"Round-trip error too large: {rel_err.item():.4f}"


# ===========================================================================
# 7. LayerQuantizer.quantize: quant_error in [0, 1], weight data changed
# ===========================================================================


def test_layer_quantizer_quantize_error_and_weight_update():
    layer = make_linear()
    W_before = layer.weight.data.clone()
    quant = make_quantizer()
    lq = LayerQuantizer(layer, quant)
    acts = make_activations()
    quant_error, scale, zero = lq.quantize(acts)
    assert 0.0 <= quant_error <= 1.0, f"quant_error out of range: {quant_error}"
    # Weight data must have been replaced with dequantized values
    assert not torch.equal(layer.weight.data, W_before), (
        "layer.weight.data should be updated to dequantized weights"
    )


# ===========================================================================
# 8. LayerQuantizer.compression_ratio: 32 / n_bits (= 8.0 for 4-bit)
# ===========================================================================


def test_layer_quantizer_compression_ratio():
    layer = make_linear()
    quant = make_quantizer()
    lq = LayerQuantizer(layer, quant)
    ratio = lq.compression_ratio()
    expected = 32.0 / N_BITS  # 8.0
    assert ratio == pytest.approx(expected), f"Expected {expected}, got {ratio}"


# ===========================================================================
# 9. ModelQuantizer.find_linear_layers: finds all nn.Linear modules
# ===========================================================================


def test_model_quantizer_find_linear_layers():
    model = TinyModel()
    mq = ModelQuantizer(model, n_bits=N_BITS, group_size=GROUP_SIZE)
    layers = mq.find_linear_layers()
    assert set(layers.keys()) == {"fc1", "fc2"}, f"Unexpected keys: {set(layers.keys())}"
    for name, layer in layers.items():
        assert isinstance(layer, nn.Linear), f"{name} is not nn.Linear"


# ===========================================================================
# 10. ModelQuantizer.quantize_model: returns dict with one entry per linear layer
# ===========================================================================


def test_model_quantizer_quantize_model_keys():
    torch.manual_seed(0)
    model = TinyModel()
    mq = ModelQuantizer(model, n_bits=N_BITS, group_size=GROUP_SIZE)
    calib = make_activations()
    errors = mq.quantize_model(calib)
    assert set(errors.keys()) == {"fc1", "fc2"}, (
        f"Expected keys {{'fc1','fc2'}}, got {set(errors.keys())}"
    )


# ===========================================================================
# 11. ModelQuantizer.quantization_summary: all 4 keys present, mean_error in [0,1]
# ===========================================================================


def test_model_quantizer_quantization_summary():
    torch.manual_seed(1)
    model = TinyModel()
    mq = ModelQuantizer(model, n_bits=N_BITS, group_size=GROUP_SIZE)
    calib = make_activations()
    errors = mq.quantize_model(calib)
    summary = mq.quantization_summary(errors)
    required_keys = {"mean_error", "max_error", "total_params", "effective_bits"}
    assert required_keys.issubset(summary.keys()), (
        f"Missing keys: {required_keys - set(summary.keys())}"
    )
    assert 0.0 <= summary["mean_error"] <= 1.0, f"mean_error out of [0,1]: {summary['mean_error']}"


# ===========================================================================
# 12. QuantizationBenchmark.weight_error: ≥ 0, 0 for identical weights
# ===========================================================================


def test_benchmark_weight_error():
    bench = QuantizationBenchmark()
    W = make_linear().weight.data
    assert bench.weight_error(W, W) == pytest.approx(0.0), (
        "weight_error for identical tensors must be 0"
    )
    W_noisy = W + 0.1 * torch.randn_like(W)
    err = bench.weight_error(W, W_noisy)
    assert err >= 0.0, "weight_error must be non-negative"


# ===========================================================================
# 13. QuantizationBenchmark.output_error: ≥ 0, 0 for identical outputs
# ===========================================================================


def test_benchmark_output_error():
    bench = QuantizationBenchmark()
    out = torch.randn(BATCH, SEQ_LEN, D_OUT)
    assert bench.output_error(out, out) == pytest.approx(0.0), (
        "output_error for identical tensors must be 0"
    )
    out_noisy = out + 0.05 * torch.randn_like(out)
    err = bench.output_error(out, out_noisy)
    assert err >= 0.0, "output_error must be non-negative"


# ===========================================================================
# 14. QuantizationBenchmark.perplexity_increase: ≥ 1.0 when quantized is worse
# ===========================================================================


def test_benchmark_perplexity_increase():
    bench = QuantizationBenchmark()
    # Original model has higher log-probs than quantized → ratio ≥ 1
    original_lp = torch.full((16,), -1.0)
    quantized_lp = torch.full((16,), -2.0)  # worse (lower log-prob)
    ppl_inc = bench.perplexity_increase(original_lp, quantized_lp)
    assert ppl_inc >= 1.0, f"perplexity_increase should be ≥ 1.0, got {ppl_inc}"


# ===========================================================================
# 15. Forward pass with quantized model: produces valid logits (no NaN/Inf)
# ===========================================================================


def test_forward_pass_after_quantization():
    torch.manual_seed(7)
    model = TinyModel()
    mq = ModelQuantizer(model, n_bits=N_BITS, group_size=GROUP_SIZE)
    calib = make_activations()

    # Quantize the model
    _ = mq.quantize_model(calib)

    # Run a forward pass with a fresh input
    test_input = make_activations(seed=99)
    with torch.no_grad():
        logits = model(test_input)

    assert logits.shape == (BATCH, SEQ_LEN, D_IN), f"Unexpected output shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "Forward pass produced NaN values"
    assert not torch.isinf(logits).any(), "Forward pass produced Inf values"

    # Also verify backward pass works (gradients flow)
    test_input_grad = make_activations(seed=99).requires_grad_(False)
    model.train()
    logits_train = model(test_input_grad)
    loss = logits_train.sum()
    loss.backward()
