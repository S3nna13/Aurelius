import torch.nn as nn
from src.profiling.flops_counter import (
    FLOPsConfig,
    FLOPsCounter,
    ModuleFLOPs,
    FLOPS_REGISTRY,
)


def make_counter(include_bias=True):
    return FLOPsCounter(FLOPsConfig(include_bias=include_bias))


def test_count_linear_no_bias():
    c = make_counter(include_bias=False)
    assert c.count_linear(4, 8, batch_size=1, seq_len=1) == 2 * 1 * 1 * 4 * 8


def test_count_linear_with_bias():
    c = make_counter(include_bias=True)
    expected = 2 * 1 * 1 * 4 * 8 + 8
    assert c.count_linear(4, 8, batch_size=1, seq_len=1) == expected


def test_count_linear_batch_seq():
    c = make_counter(include_bias=False)
    assert c.count_linear(4, 8, batch_size=2, seq_len=3) == 2 * 2 * 3 * 4 * 8


def test_count_attention_formula():
    c = make_counter()
    result = c.count_attention(seq_len=16, d_model=64, n_heads=4, batch_size=2)
    assert result == 4 * 2 * 2 * 16 * 64 * 64


def test_count_module_sequential_linear():
    model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 8))
    c = make_counter(include_bias=False)
    result = c.count_module(model, input_shape=(1, 1))
    assert len(result) == 2
    assert all(isinstance(r, ModuleFLOPs) for r in result)
    assert result[0].module_type == "Linear"
    assert result[1].module_type == "Linear"


def test_count_module_flops_values():
    model = nn.Linear(4, 8, bias=False)
    c = make_counter(include_bias=False)
    result = c.count_module(model, input_shape=(1, 1))
    assert len(result) == 1
    assert result[0].flops == 2 * 1 * 1 * 4 * 8


def test_count_module_params():
    model = nn.Linear(4, 8, bias=False)
    c = make_counter()
    result = c.count_module(model, input_shape=(1, 1))
    assert result[0].params == 4 * 8


def test_total_flops_sum():
    flops = [
        ModuleFLOPs("a", "Linear", flops=100, params=10),
        ModuleFLOPs("b", "Linear", flops=200, params=20),
    ]
    c = make_counter()
    assert c.total_flops(flops) == 300


def test_total_flops_empty():
    c = make_counter()
    assert c.total_flops([]) == 0


def test_summary_keys():
    model = nn.Linear(8, 16)
    c = make_counter()
    s = c.summary(model, input_shape=(1, 1))
    assert "total_flops" in s
    assert "total_params" in s
    assert "by_layer" in s


def test_summary_by_layer_structure():
    model = nn.Linear(8, 16)
    c = make_counter()
    s = c.summary(model, input_shape=(1, 1))
    assert len(s["by_layer"]) >= 1
    layer = s["by_layer"][0]
    assert "module_name" in layer
    assert "module_type" in layer
    assert "flops" in layer
    assert "params" in layer


def test_registry_key():
    assert "default" in FLOPS_REGISTRY
    assert FLOPS_REGISTRY["default"] is FLOPsCounter
