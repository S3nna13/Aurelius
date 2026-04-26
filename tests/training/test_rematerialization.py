"""Tests for src/training/rematerialization.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.training.rematerialization import (
    MemoryProfiler,
    RematerializationConfig,
    SelectiveCheckpointWrapper,
    checkpoint_forward,
    estimate_activation_memory,
    should_checkpoint_layer,
    wrap_model_layers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class LinearLayer(nn.Module):
    """Simple stand-in for a transformer layer: nn.Linear that returns a Tensor."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return self.linear(x)


def make_layers(n: int = 4, dim: int = 64) -> nn.ModuleList:
    return nn.ModuleList([LinearLayer(dim) for _ in range(n)])


# ---------------------------------------------------------------------------
# 1. RematerializationConfig defaults
# ---------------------------------------------------------------------------


def test_rematerialization_config_defaults() -> None:
    cfg = RematerializationConfig()
    assert cfg.checkpoint_every_n == 1
    assert cfg.checkpoint_layers is None
    assert cfg.use_reentrant is False
    assert cfg.offload_to_cpu is False


# ---------------------------------------------------------------------------
# 2. should_checkpoint_layer — every_n=1 (all layers)
# ---------------------------------------------------------------------------


def test_should_checkpoint_layer_every_1_all_layers() -> None:
    cfg = RematerializationConfig(checkpoint_every_n=1)
    for i in range(10):
        assert should_checkpoint_layer(i, cfg) is True


# ---------------------------------------------------------------------------
# 3. should_checkpoint_layer — every_n=2 (every other)
# ---------------------------------------------------------------------------


def test_should_checkpoint_layer_every_2() -> None:
    cfg = RematerializationConfig(checkpoint_every_n=2)
    assert should_checkpoint_layer(0, cfg) is True
    assert should_checkpoint_layer(1, cfg) is False
    assert should_checkpoint_layer(2, cfg) is True
    assert should_checkpoint_layer(3, cfg) is False
    assert should_checkpoint_layer(4, cfg) is True


# ---------------------------------------------------------------------------
# 4. should_checkpoint_layer — explicit checkpoint_layers list
# ---------------------------------------------------------------------------


def test_should_checkpoint_layer_explicit_list() -> None:
    cfg = RematerializationConfig(checkpoint_layers=[0, 2, 5])
    assert should_checkpoint_layer(0, cfg) is True
    assert should_checkpoint_layer(1, cfg) is False
    assert should_checkpoint_layer(2, cfg) is True
    assert should_checkpoint_layer(3, cfg) is False
    assert should_checkpoint_layer(5, cfg) is True
    assert should_checkpoint_layer(4, cfg) is False


# ---------------------------------------------------------------------------
# 5. checkpoint_forward produces same output as direct call
# ---------------------------------------------------------------------------


def test_checkpoint_forward_same_output() -> None:
    torch.manual_seed(0)
    layer = LinearLayer(64)
    x = torch.randn(2, 10, 64)

    with torch.no_grad():
        direct_out = layer(x)
        ckpt_out = checkpoint_forward(layer, x, use_reentrant=False)

    assert direct_out.shape == ckpt_out.shape
    assert torch.allclose(direct_out, ckpt_out, atol=1e-5), (
        f"Max diff: {(direct_out - ckpt_out).abs().max()}"
    )


# ---------------------------------------------------------------------------
# 6. checkpoint_forward preserves gradients
# ---------------------------------------------------------------------------


def test_checkpoint_forward_preserves_gradients() -> None:
    torch.manual_seed(1)
    layer = LinearLayer(64)
    x = torch.randn(2, 10, 64, requires_grad=True)

    out = checkpoint_forward(layer, x, use_reentrant=False)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "x.grad should not be None after backward"
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# 7. SelectiveCheckpointWrapper — forward output shape
# ---------------------------------------------------------------------------


def test_selective_checkpoint_wrapper_output_shape() -> None:
    layers = make_layers(n=4, dim=64)
    cfg = RematerializationConfig(checkpoint_every_n=2)
    wrapper = SelectiveCheckpointWrapper(layers, cfg)

    x = torch.randn(2, 10, 64)
    out = wrapper(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 8. SelectiveCheckpointWrapper — checkpoint_every_n=1 (all checkpointed)
# ---------------------------------------------------------------------------


def test_selective_checkpoint_wrapper_all_checkpointed() -> None:
    torch.manual_seed(2)
    layers = make_layers(n=3, dim=64)
    cfg = RematerializationConfig(checkpoint_every_n=1)
    wrapper = SelectiveCheckpointWrapper(layers, cfg)

    x = torch.randn(1, 5, 64, requires_grad=True)
    out = wrapper(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None


# ---------------------------------------------------------------------------
# 9. SelectiveCheckpointWrapper — checkpoint_every_n=999 (none checkpointed)
# ---------------------------------------------------------------------------


def test_selective_checkpoint_wrapper_none_checkpointed() -> None:
    torch.manual_seed(3)
    layers = make_layers(n=4, dim=64)
    cfg = RematerializationConfig(checkpoint_every_n=999)
    wrapper = SelectiveCheckpointWrapper(layers, cfg)

    x = torch.randn(1, 5, 64, requires_grad=True)
    out = wrapper(x)
    loss = out.sum()
    loss.backward()

    # Layer 0 is still checkpointed (0 % 999 == 0), but that's fine —
    # the test checks the wrapper runs successfully with mostly-direct calls.
    assert x.grad is not None
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 10. estimate_activation_memory — keys present
# ---------------------------------------------------------------------------


def test_estimate_activation_memory_keys_present() -> None:
    result = estimate_activation_memory(batch_size=2, seq_len=512, d_model=64, n_layers=4)
    assert "full_mb" in result
    assert "checkpointed_mb" in result
    assert "savings_factor" in result


# ---------------------------------------------------------------------------
# 11. estimate_activation_memory — savings_factor == n_layers
# ---------------------------------------------------------------------------


def test_estimate_activation_memory_savings_factor() -> None:
    n_layers = 8
    result = estimate_activation_memory(batch_size=2, seq_len=512, d_model=128, n_layers=n_layers)
    assert result["savings_factor"] == pytest.approx(float(n_layers), rel=1e-6)


# ---------------------------------------------------------------------------
# 12. MemoryProfiler.measure — correct tuple structure
# ---------------------------------------------------------------------------


def test_memory_profiler_measure_tuple_structure() -> None:
    profiler = MemoryProfiler()

    def simple_fn(a: int, b: int) -> int:
        return a + b

    result, stats = profiler.measure(simple_fn, 3, 4)

    assert result == 7
    assert isinstance(stats, dict)
    assert "peak_memory_delta_mb" in stats
    assert "elapsed_ms" in stats
    assert isinstance(stats["peak_memory_delta_mb"], float)
    assert isinstance(stats["elapsed_ms"], float)
    assert stats["elapsed_ms"] >= 0.0
    assert stats["peak_memory_delta_mb"] >= 0.0


# ---------------------------------------------------------------------------
# 13. wrap_model_layers convenience function
# ---------------------------------------------------------------------------


def test_wrap_model_layers_returns_wrapper() -> None:
    layers = make_layers(n=3, dim=64)
    cfg = RematerializationConfig(checkpoint_every_n=1)
    wrapper = wrap_model_layers(layers, cfg)

    assert isinstance(wrapper, SelectiveCheckpointWrapper)
    x = torch.randn(1, 8, 64)
    out = wrapper(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 14. RematerializationConfig with custom values
# ---------------------------------------------------------------------------


def test_rematerialization_config_custom() -> None:
    cfg = RematerializationConfig(
        checkpoint_every_n=3,
        checkpoint_layers=[1, 3, 5],
        use_reentrant=True,
        offload_to_cpu=True,
    )
    assert cfg.checkpoint_every_n == 3
    assert cfg.checkpoint_layers == [1, 3, 5]
    assert cfg.use_reentrant is True
    assert cfg.offload_to_cpu is True


# ---------------------------------------------------------------------------
# 15. estimate_activation_memory values are positive and ordered correctly
# ---------------------------------------------------------------------------


def test_estimate_activation_memory_ordering() -> None:
    result = estimate_activation_memory(
        batch_size=4, seq_len=1024, d_model=256, n_layers=12, dtype_bytes=4
    )
    assert result["full_mb"] > 0.0
    assert result["checkpointed_mb"] > 0.0
    assert result["full_mb"] > result["checkpointed_mb"]
