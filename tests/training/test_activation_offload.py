"""Tests for src/training/activation_offload.py"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.activation_offload import (
    ActivationOffloadHook,
    OffloadFunction,
    OffloadedActivation,
    OffloadingStats,
    SelectiveOffloadWrapper,
    offload_activation,
    wrap_model_with_offloading,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

SMALL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=32,
)


@pytest.fixture
def small_model() -> AureliusTransformer:
    return AureliusTransformer(SMALL_CFG)


# ---------------------------------------------------------------------------
# 1. OffloadedActivation — stores on CPU
# ---------------------------------------------------------------------------

def test_offloaded_activation_cpu():
    """OffloadedActivation should store the tensor on CPU regardless of origin."""
    t = torch.randn(4, 8)
    oa = OffloadedActivation(t)
    assert oa.cpu_data.device.type == "cpu", "cpu_data must live on CPU"
    assert oa.shape == t.shape
    assert oa.dtype == t.dtype


# ---------------------------------------------------------------------------
# 2. OffloadedActivation.restore() — returns tensor on original device
# ---------------------------------------------------------------------------

def test_offloaded_activation_restore():
    """restore() should return a tensor on the original device with correct shape/dtype."""
    t = torch.randn(3, 5, dtype=torch.float32)
    oa = OffloadedActivation(t)
    restored = oa.restore()
    assert restored.device.type == t.device.type
    assert restored.shape == t.shape
    assert restored.dtype == t.dtype


# ---------------------------------------------------------------------------
# 3. offload_activation — moves tensor to CPU
# ---------------------------------------------------------------------------

def test_offload_function_forward_cpu():
    """offload_activation should return a CPU tensor."""
    x = torch.randn(2, 4, requires_grad=True)
    y = offload_activation(x)
    assert y.device.type == "cpu", f"Expected CPU, got {y.device}"


# ---------------------------------------------------------------------------
# 4. Gradients flow through offload_activation
# ---------------------------------------------------------------------------

def test_offload_function_backward():
    """Gradients must flow back through OffloadFunction (non-None grad on input)."""
    x = torch.randn(2, 4, requires_grad=True)
    y = offload_activation(x)
    loss = y.to(x.device).sum()
    loss.backward()
    assert x.grad is not None, "x.grad must not be None after backward"
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# 5. Values are preserved through offload + restore
# ---------------------------------------------------------------------------

def test_offload_function_preserves_values():
    """Values should be numerically identical after offload and restore."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = offload_activation(x)
    restored = y.to(x.device)
    assert torch.allclose(x.detach(), restored.detach()), (
        "Values changed during offload/restore"
    )


# ---------------------------------------------------------------------------
# 6. SelectiveOffloadWrapper offloads in training mode
# ---------------------------------------------------------------------------

def test_selective_wrapper_offloads_training():
    """In training mode the wrapper's output tensor should be on CPU."""
    inner = nn.Linear(8, 8)
    wrapper = SelectiveOffloadWrapper(inner, offload=True)
    wrapper.train()

    x = torch.randn(2, 8)
    out = wrapper(x)
    assert out.device.type == "cpu", (
        f"Expected output on CPU in training mode, got {out.device}"
    )


# ---------------------------------------------------------------------------
# 7. SelectiveOffloadWrapper does NOT offload in eval mode
# ---------------------------------------------------------------------------

def test_selective_wrapper_no_offload_eval():
    """In eval mode the wrapper should pass tensors through without offloading."""
    inner = nn.Linear(8, 8)
    wrapper = SelectiveOffloadWrapper(inner, offload=True)
    wrapper.eval()

    x = torch.randn(2, 8)
    with torch.no_grad():
        out = wrapper(x)
    assert not wrapper.training, "Wrapper should be in eval mode"
    assert out.device == x.device


# ---------------------------------------------------------------------------
# 8. SelectiveOffloadWrapper handles tuple output
# ---------------------------------------------------------------------------

def test_selective_wrapper_tuple_output():
    """When the inner module returns a tuple, only the first element is offloaded."""

    class TupleModule(nn.Module):
        def forward(self, x):
            return x * 2.0, x * 3.0

    wrapper = SelectiveOffloadWrapper(TupleModule(), offload=True)
    wrapper.train()

    x = torch.randn(2, 4)
    out = wrapper(x)

    assert isinstance(out, tuple), "Output must remain a tuple"
    assert len(out) == 2, "Tuple length must be preserved"
    assert out[0].device.type == "cpu", "First element should be on CPU"
    assert out[1].device == x.device, "Second element device must be unchanged"


# ---------------------------------------------------------------------------
# 9. wrap_model_with_offloading — wraps all layers
# ---------------------------------------------------------------------------

def test_wrap_model_all_layers(small_model):
    """wrap_model_with_offloading with no layer_indices should wrap every layer."""
    wrap_model_with_offloading(small_model)
    for i, layer in enumerate(small_model.layers):
        assert isinstance(layer, SelectiveOffloadWrapper), (
            f"Layer {i} should be a SelectiveOffloadWrapper, got {type(layer)}"
        )


# ---------------------------------------------------------------------------
# 10. wrap_model_with_offloading — wraps only selected layers
# ---------------------------------------------------------------------------

def test_wrap_model_selective_layers(small_model):
    """wrap_model_with_offloading with layer_indices=[0] should wrap only layer 0."""
    n_layers = len(small_model.layers)
    assert n_layers >= 2, "Need at least 2 layers for this test"

    wrap_model_with_offloading(small_model, layer_indices=[0])

    assert isinstance(small_model.layers[0], SelectiveOffloadWrapper), (
        "Layer 0 should be wrapped"
    )
    for i in range(1, n_layers):
        assert not isinstance(small_model.layers[i], SelectiveOffloadWrapper), (
            f"Layer {i} should NOT be wrapped"
        )


# ---------------------------------------------------------------------------
# 11. OffloadingStats — tracks bytes correctly
# ---------------------------------------------------------------------------

def test_offloading_stats_tracking():
    """OffloadingStats.record_offload should accumulate byte counts correctly."""
    stats = OffloadingStats()
    assert stats.offloaded_bytes == 0

    t1 = torch.randn(4, 4, dtype=torch.float32)  # 16 elements x 4 bytes = 64
    stats.record_offload(t1)
    expected = t1.numel() * t1.element_size()
    assert stats.offloaded_bytes == expected, (
        f"Expected {expected} bytes, got {stats.offloaded_bytes}"
    )

    t2 = torch.randn(8, dtype=torch.float16)  # 8 elements x 2 bytes = 16
    stats.record_offload(t2)
    expected += t2.numel() * t2.element_size()
    assert stats.offloaded_bytes == expected

    assert stats.offloaded_mb() == pytest.approx(expected / (1024 ** 2))
