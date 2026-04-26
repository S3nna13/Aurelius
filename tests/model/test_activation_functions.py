"""Tests for src/model/activation_functions.py.

Import via the public aurelius namespace:
    from aurelius.model.activation_functions import ...
"""

from __future__ import annotations

import pytest
import torch
from aurelius.model.activation_functions import (
    ActivationBenchmark,
    ActivationConfig,
    FFNFactory,
    GeGLUFFN,
    ReGLUFFN,
    SquaredReLUFFN,
    SwiGLUFFN,
    _StandardFFN,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_MODEL = 16
D_FF = 32
BATCH = 2
SEQ = 4


@pytest.fixture
def tiny_x():
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ, D_MODEL)


@pytest.fixture
def factory():
    return FFNFactory()


@pytest.fixture
def swiglu():
    return SwiGLUFFN(D_MODEL, D_FF)


@pytest.fixture
def geglu():
    return GeGLUFFN(D_MODEL, D_FF)


@pytest.fixture
def reglu():
    return ReGLUFFN(D_MODEL, D_FF)


@pytest.fixture
def squared_relu():
    return SquaredReLUFFN(D_MODEL, D_FF)


# ---------------------------------------------------------------------------
# 1. SwiGLUFFN output shape
# ---------------------------------------------------------------------------


def test_swiglu_output_shape(swiglu, tiny_x):
    out = swiglu(tiny_x)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected ({BATCH}, {SEQ}, {D_MODEL}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 2. SwiGLUFFN gradient flows
# ---------------------------------------------------------------------------


def test_swiglu_gradient_flows(swiglu):
    x = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=True)
    out = swiglu(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient for input"
    assert x.grad.shape == x.shape
    # All weight matrices should also receive gradients
    for name, param in swiglu.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# 3. GeGLUFFN output shape
# ---------------------------------------------------------------------------


def test_geglu_output_shape(geglu, tiny_x):
    out = geglu(tiny_x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# 4. ReGLUFFN output shape
# ---------------------------------------------------------------------------


def test_reglu_output_shape(reglu, tiny_x):
    out = reglu(tiny_x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# 5. SquaredReLUFFN output shape
# ---------------------------------------------------------------------------


def test_squared_relu_output_shape(squared_relu, tiny_x):
    out = squared_relu(tiny_x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# 6. SquaredReLUFFN hidden activations are non-negative
#    (The hidden layer after ReLU^2 is always >= 0)
# ---------------------------------------------------------------------------


def test_squared_relu_hidden_nonnegative(tiny_x):
    """Hidden activations h*h are always >= 0 because ReLU(.) >= 0."""
    import torch.nn.functional as F

    ffn = SquaredReLUFFN(D_MODEL, D_FF)
    with torch.no_grad():
        h = F.relu(ffn.W1(tiny_x))
        hidden = h * h
    assert (hidden >= 0).all(), "SquaredReLU hidden layer has negative values"


# ---------------------------------------------------------------------------
# 7-12. FFNFactory.create returns the correct class for each variant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "activation,expected_cls",
    [
        ("swiglu", SwiGLUFFN),
        ("geglu", GeGLUFFN),
        ("reglu", ReGLUFFN),
        ("squared_relu", SquaredReLUFFN),
        ("gelu", _StandardFFN),
        ("silu", _StandardFFN),
    ],
)
def test_factory_create_returns_correct_class(factory, activation, expected_cls):
    cfg = ActivationConfig(activation=activation, d_model=D_MODEL, d_ff=D_FF)
    module = factory.create(cfg)
    assert isinstance(module, expected_cls), (
        f"activation={activation!r}: expected {expected_cls.__name__}, got {type(module).__name__}"
    )


# ---------------------------------------------------------------------------
# 13. FFNFactory.parameter_count returns a positive integer
# ---------------------------------------------------------------------------


def test_factory_parameter_count_positive(factory, swiglu):
    count = factory.parameter_count(swiglu)
    assert isinstance(count, int)
    assert count > 0


# ---------------------------------------------------------------------------
# 14. flop_estimate returns the correct keys
# ---------------------------------------------------------------------------


def test_flop_estimate_keys(factory):
    cfg = ActivationConfig(activation="swiglu", d_model=D_MODEL, d_ff=D_FF)
    result = factory.flop_estimate(cfg, batch_size=BATCH, seq_len=SEQ)
    assert "forward_flops" in result
    assert "backward_flops" in result


# ---------------------------------------------------------------------------
# 15. ActivationBenchmark.compare_outputs returns dict with expected keys
# ---------------------------------------------------------------------------


def test_benchmark_compare_outputs_keys(tiny_x):
    configs = [
        ActivationConfig("swiglu", D_MODEL, D_FF),
        ActivationConfig("geglu", D_MODEL, D_FF),
        ActivationConfig("squared_relu", D_MODEL, D_FF),
    ]
    bench = ActivationBenchmark(configs)
    outputs = bench.compare_outputs(tiny_x)
    assert set(outputs.keys()) == {"swiglu", "geglu", "squared_relu"}


# ---------------------------------------------------------------------------
# 16. compare_outputs produces finite tensors
# ---------------------------------------------------------------------------


def test_benchmark_outputs_finite(tiny_x):
    configs = [
        ActivationConfig("swiglu", D_MODEL, D_FF),
        ActivationConfig("gelu", D_MODEL, D_FF),
    ]
    bench = ActivationBenchmark(configs)
    outputs = bench.compare_outputs(tiny_x)
    for name, tensor in outputs.items():
        assert torch.isfinite(tensor).all(), f"{name} produced non-finite output"


# ---------------------------------------------------------------------------
# 17. sparsity is in [0, 1]
# ---------------------------------------------------------------------------


def test_sparsity_range(tiny_x):
    factory = FFNFactory()
    cfg = ActivationConfig("reglu", D_MODEL, D_FF)
    ffn = factory.create(cfg)
    bench = ActivationBenchmark([cfg])
    s = bench.sparsity(tiny_x, ffn)
    assert 0.0 <= s <= 1.0, f"Sparsity out of range: {s}"


# ---------------------------------------------------------------------------
# 18. GLU variants have more parameters than a standard 2-layer MLP
#     of the same d_model and d_ff (two input projections vs one)
# ---------------------------------------------------------------------------


def test_glu_more_params_than_standard(factory):
    cfg_glu = ActivationConfig("swiglu", D_MODEL, D_FF)
    cfg_std = ActivationConfig("gelu", D_MODEL, D_FF)
    glu_params = factory.parameter_count(factory.create(cfg_glu))
    std_params = factory.parameter_count(factory.create(cfg_std))
    assert glu_params > std_params, (
        f"Expected GLU params ({glu_params}) > standard params ({std_params})"
    )


# ---------------------------------------------------------------------------
# 19. All modules work at d_model=16, d_ff=32 (the tiny fixture dimensions)
# ---------------------------------------------------------------------------


def test_all_variants_tiny_dims(tiny_x):
    factory = FFNFactory()
    for act in ["swiglu", "geglu", "reglu", "squared_relu", "gelu", "silu"]:
        cfg = ActivationConfig(act, D_MODEL, D_FF)
        ffn = factory.create(cfg)
        out = ffn(tiny_x)
        assert out.shape == (BATCH, SEQ, D_MODEL), f"{act}: unexpected output shape {out.shape}"


# ---------------------------------------------------------------------------
# 20. FFNFactory raises ValueError for unknown activation
# ---------------------------------------------------------------------------


def test_factory_unknown_activation_raises(factory):
    cfg = ActivationConfig(activation="unknown_act", d_model=D_MODEL, d_ff=D_FF)
    with pytest.raises(ValueError):
        factory.create(cfg)
