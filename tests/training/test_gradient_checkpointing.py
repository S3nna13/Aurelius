"""Tests for gradient checkpointing utilities.

Covers: CheckpointConfig, checkpoint_forward, CheckpointedLayer,
CheckpointedSequential, estimate_activation_memory,
estimate_checkpointed_memory, wrap_model_with_checkpointing.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.training.gradient_checkpointing import (
    CheckpointConfig,
    CheckpointedLayer,
    CheckpointedModule,
    CheckpointedSequential,
    apply_checkpointing,
    checkpoint_forward,
    estimate_activation_memory,
    estimate_checkpointed_memory,
    wrap_model_with_checkpointing,
)

# Tiny dimensions used throughout
BATCH = 2
SEQ_LEN = 4
D_MODEL = 8
N_LAYERS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_layer(in_f: int = D_MODEL, out_f: int = D_MODEL) -> nn.Linear:
    return nn.Linear(in_f, out_f, bias=False)


def make_input(requires_grad: bool = False) -> torch.Tensor:
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    if requires_grad:
        x.requires_grad_(True)
    return x


# ---------------------------------------------------------------------------
# 1. CheckpointConfig defaults
# ---------------------------------------------------------------------------

def test_checkpoint_config_defaults():
    cfg = CheckpointConfig()
    assert cfg.enabled is True
    assert cfg.checkpoint_every_n_layers == 1
    assert cfg.use_reentrant is False


# ---------------------------------------------------------------------------
# 2. checkpoint_forward output matches direct call (enabled=True)
# ---------------------------------------------------------------------------

def test_checkpoint_forward_output_matches_direct():
    torch.manual_seed(0)
    fn = make_layer()
    fn.eval()
    x = make_input()

    with torch.no_grad():
        expected = fn(x)
        actual = checkpoint_forward(fn, x, enabled=True)

    assert torch.allclose(expected, actual, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. checkpoint_forward with enabled=False calls fn directly (same values)
# ---------------------------------------------------------------------------

def test_checkpoint_forward_disabled_same_values():
    torch.manual_seed(1)
    fn = make_layer()
    fn.eval()
    x = make_input()

    with torch.no_grad():
        expected = fn(x)
        actual = checkpoint_forward(fn, x, enabled=False)

    assert torch.allclose(expected, actual, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. CheckpointedLayer output shape preserved
# ---------------------------------------------------------------------------

def test_checkpointed_layer_output_shape():
    torch.manual_seed(2)
    layer = make_layer()
    cfg = CheckpointConfig()
    cl = CheckpointedLayer(layer, cfg)

    x = make_input()
    out = cl(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 5. CheckpointedLayer gradients flow (loss.backward() does not error)
# ---------------------------------------------------------------------------

def test_checkpointed_layer_gradients_flow():
    torch.manual_seed(3)
    layer = make_layer()
    cfg = CheckpointConfig()
    cl = CheckpointedLayer(layer, cfg)

    x = make_input(requires_grad=True)
    out = cl(x)
    out.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# 6. CheckpointedSequential output shape
# ---------------------------------------------------------------------------

def test_checkpointed_sequential_output_shape():
    torch.manual_seed(4)
    layers = [make_layer() for _ in range(N_LAYERS)]
    cfg = CheckpointConfig(checkpoint_every_n_layers=1)
    seq = CheckpointedSequential(layers, cfg)

    x = make_input()
    out = seq(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 7. CheckpointedSequential with n=2 applies layers in order
# ---------------------------------------------------------------------------

def test_checkpointed_sequential_n2_layers_in_order():
    torch.manual_seed(5)
    layers = [make_layer() for _ in range(4)]
    cfg = CheckpointConfig(checkpoint_every_n_layers=2)
    seq = CheckpointedSequential(layers, cfg)

    x = make_input()
    out_ckpt = seq(x)

    # Apply the exact same module_list manually in order
    with torch.no_grad():
        x_manual = x.clone()
        for layer in seq.module_list:
            x_manual = layer(x_manual)

    assert torch.allclose(out_ckpt, x_manual, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. estimate_activation_memory formula correct
# ---------------------------------------------------------------------------

def test_estimate_activation_memory_formula():
    result = estimate_activation_memory(
        batch_size=BATCH, seq_len=SEQ_LEN, d_model=D_MODEL, n_layers=N_LAYERS
    )
    expected = BATCH * SEQ_LEN * D_MODEL * N_LAYERS * 4  # float32
    assert result == expected


# ---------------------------------------------------------------------------
# 9. estimate_checkpointed_memory < estimate_activation_memory for n_layers > 1
# ---------------------------------------------------------------------------

def test_checkpointed_memory_less_than_full_memory():
    full = estimate_activation_memory(BATCH, SEQ_LEN, D_MODEL, N_LAYERS)
    ckpt = estimate_checkpointed_memory(BATCH, SEQ_LEN, D_MODEL, N_LAYERS, checkpoint_every=2)
    assert ckpt < full


# ---------------------------------------------------------------------------
# 10. wrap_model_with_checkpointing returns CheckpointedSequential
# ---------------------------------------------------------------------------

def test_wrap_model_returns_checkpointed_sequential():
    torch.manual_seed(6)
    model = nn.Sequential(*[make_layer() for _ in range(N_LAYERS)])
    cfg = CheckpointConfig()
    wrapped = wrap_model_with_checkpointing(model, cfg)
    assert isinstance(wrapped, CheckpointedSequential)


# ---------------------------------------------------------------------------
# 11. memory estimates scale linearly with batch_size
# ---------------------------------------------------------------------------

def test_memory_estimates_scale_linearly_with_batch_size():
    bs1 = estimate_activation_memory(1, SEQ_LEN, D_MODEL, N_LAYERS)
    bs2 = estimate_activation_memory(2, SEQ_LEN, D_MODEL, N_LAYERS)
    assert bs2 == 2 * bs1

    ckpt1 = estimate_checkpointed_memory(1, SEQ_LEN, D_MODEL, N_LAYERS, checkpoint_every=2)
    ckpt2 = estimate_checkpointed_memory(2, SEQ_LEN, D_MODEL, N_LAYERS, checkpoint_every=2)
    assert ckpt2 == 2 * ckpt1


# ---------------------------------------------------------------------------
# 12. estimate_checkpointed_memory with checkpoint_every=1 equals full memory
# ---------------------------------------------------------------------------

def test_checkpointed_memory_every1_equals_full():
    full = estimate_activation_memory(BATCH, SEQ_LEN, D_MODEL, N_LAYERS)
    ckpt = estimate_checkpointed_memory(BATCH, SEQ_LEN, D_MODEL, N_LAYERS, checkpoint_every=1)
    assert ckpt == full


# ---------------------------------------------------------------------------
# 13. CheckpointedLayer disabled (enabled=False) still produces correct output
# ---------------------------------------------------------------------------

def test_checkpointed_layer_disabled_produces_output():
    torch.manual_seed(7)
    layer = make_layer()
    cfg = CheckpointConfig(enabled=False)
    cl = CheckpointedLayer(layer, cfg)

    x = make_input()
    out = cl(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 14. wrap_model_with_checkpointing: wrapped model forward shape preserved
# ---------------------------------------------------------------------------

def test_wrap_model_forward_shape():
    torch.manual_seed(8)
    model = nn.Sequential(*[make_layer() for _ in range(N_LAYERS)])
    cfg = CheckpointConfig(checkpoint_every_n_layers=2)
    wrapped = wrap_model_with_checkpointing(model, cfg)

    x = make_input()
    out = wrapped(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 15. estimate_checkpointed_memory uses ceil for non-divisible n_layers
# ---------------------------------------------------------------------------

def test_checkpointed_memory_ceil_non_divisible():
    # n_layers=5, checkpoint_every=2 => ceil(5/2)=3 checkpoints
    result = estimate_checkpointed_memory(
        batch_size=1, seq_len=1, d_model=1, n_layers=5, checkpoint_every=2, dtype_bytes=1
    )
    expected = math.ceil(5 / 2)  # 3
    assert result == expected


# ---------------------------------------------------------------------------
# 16. CheckpointedSequential backward completes and grad is not None
# ---------------------------------------------------------------------------

def test_checkpointed_sequential_backward_completes():
    torch.manual_seed(9)
    layers = [make_layer() for _ in range(N_LAYERS)]
    cfg = CheckpointConfig(checkpoint_every_n_layers=1)
    seq = CheckpointedSequential(layers, cfg)

    x = make_input(requires_grad=True)
    out = seq(x)
    out.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# CheckpointedModule + apply_checkpointing tests (spec requirements)
# ---------------------------------------------------------------------------

def _fresh_16d_module():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 16), nn.ReLU())


def _fresh_16d_input():
    torch.manual_seed(1)
    return torch.randn(4, 16)


def test_checkpoint_config_instantiates():
    cfg = CheckpointConfig()
    assert cfg.checkpoint_every == 2
    assert cfg.enabled is True


def test_checkpointed_module_instantiates():
    cfg = CheckpointConfig()
    m = _fresh_16d_module()
    cm = CheckpointedModule(m, cfg)
    assert cm is not None


def test_checkpointed_module_forward_same_as_unwrapped():
    cfg = CheckpointConfig(enabled=False)
    m = _fresh_16d_module()
    x = _fresh_16d_input()
    cm = CheckpointedModule(m, cfg)
    assert torch.allclose(m(x), cm(x))


def test_checkpointed_module_output_is_finite():
    cfg = CheckpointConfig()
    cm = CheckpointedModule(_fresh_16d_module(), cfg)
    out = cm(_fresh_16d_input())
    assert torch.all(torch.isfinite(out))


def test_checkpointed_module_gradient_flows():
    cfg = CheckpointConfig(enabled=True)
    m = _fresh_16d_module()
    cm = CheckpointedModule(m, cfg)
    x = _fresh_16d_input().requires_grad_(True)
    cm(x).sum().backward()
    assert x.grad is not None


def test_checkpointed_module_disabled_still_works():
    cfg = CheckpointConfig(enabled=False)
    cm = CheckpointedModule(_fresh_16d_module(), cfg)
    x = _fresh_16d_input()
    out = cm(x)
    assert out.shape == x.shape


def test_checkpointed_matches_non_checkpointed_same_weights():
    torch.manual_seed(42)
    m1 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
    m2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
    m2.load_state_dict(m1.state_dict())
    x = torch.randn(4, 16)
    cm_on = CheckpointedModule(m1, CheckpointConfig(enabled=True))
    cm_off = CheckpointedModule(m2, CheckpointConfig(enabled=False))
    assert torch.allclose(cm_on(x), cm_off(x), atol=1e-6)


def test_apply_checkpointing_returns_nn_module():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    cfg = CheckpointConfig()
    result = apply_checkpointing(SimpleModel(), cfg)
    assert isinstance(result, nn.Module)


def test_apply_checkpointing_works_with_simple_linear():
    cfg = CheckpointConfig()
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
    result = apply_checkpointing(model, cfg, layer_names=["0"])
    assert isinstance(result, nn.Module)
    out = result(torch.randn(2, 16))
    assert out.shape == (2, 16)


def test_checkpointed_module_finite_gradients():
    cfg = CheckpointConfig(enabled=True)
    m = _fresh_16d_module()
    cm = CheckpointedModule(m, cfg)
    cm(_fresh_16d_input()).sum().backward()
    for p in m.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))
