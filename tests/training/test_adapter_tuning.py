"""Tests for src/training/adapter_tuning.py."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.adapter_tuning import (
    AdaptedLayer,
    AdapterConfig,
    AdapterModel,
    BottleneckAdapter,
    ParallelAdapter,
    compute_adapter_efficiency,
    count_adapter_parameters,
    freeze_base_parameters,
)

# ---------------------------------------------------------------------------
# Tiny dimensions used throughout all tests
# ---------------------------------------------------------------------------
D = 16  # d_model
BN = 4  # bottleneck_dim
B = 2  # batch size
T = 4  # sequence length


# ---------------------------------------------------------------------------
# 1. AdapterConfig defaults
# ---------------------------------------------------------------------------
def test_adapter_config_defaults():
    cfg = AdapterConfig()
    assert cfg.d_model == 512
    assert cfg.bottleneck_dim == 64
    assert cfg.adapter_dropout == 0.0
    assert cfg.init_scale == 1e-3
    assert cfg.adapter_type == "bottleneck"


# ---------------------------------------------------------------------------
# 2. BottleneckAdapter output shape preserved
# ---------------------------------------------------------------------------
def test_bottleneck_adapter_output_shape():
    adapter = BottleneckAdapter(D, BN)
    x = torch.randn(B, T, D)
    out = adapter(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ---------------------------------------------------------------------------
# 3. BottleneckAdapter residual: output ≈ input at init (near-zero weights)
# ---------------------------------------------------------------------------
def test_bottleneck_adapter_near_identity_at_init():
    adapter = BottleneckAdapter(D, BN, init_scale=1e-3)
    x = torch.randn(B, T, D)
    with torch.no_grad():
        out = adapter(x)
    # The adapter delta (out - x) should be much smaller than x itself
    delta_norm = (out - x).norm().item()
    x_norm = x.norm().item()
    assert delta_norm < 0.05 * x_norm, (
        f"Adapter delta ({delta_norm:.4f}) not much smaller than input ({x_norm:.4f})"
    )


# ---------------------------------------------------------------------------
# 4. ParallelAdapter output shape (B, T, D)
# ---------------------------------------------------------------------------
def test_parallel_adapter_output_shape():
    adapter = ParallelAdapter(D, BN)
    x = torch.randn(B, T, D)
    out = adapter(x)
    assert out.shape == (B, T, D), f"Expected ({B},{T},{D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 5. AdaptedLayer output shape
# ---------------------------------------------------------------------------
def test_adapted_layer_output_shape():
    base = nn.Linear(D, D)
    adapter = BottleneckAdapter(D, BN)
    layer = AdaptedLayer(base, adapter)
    x = torch.randn(B, T, D)
    out = layer(x)
    assert out.shape == (B, T, D), f"Expected ({B},{T},{D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 6. AdaptedLayer: base Linear weight has no grad after freeze
# ---------------------------------------------------------------------------
def test_adapted_layer_frozen_base_grad():
    base = nn.Linear(D, D)
    adapter = BottleneckAdapter(D, BN)
    layer = AdaptedLayer(base, adapter)

    # Freeze only the base linear (not the adapter)
    for p in base.parameters():
        p.requires_grad = False

    x = torch.randn(B, T, D)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    # Base weight grad should be None because requires_grad=False
    assert base.weight.grad is None, "Base weight should not accumulate grad when frozen"


# ---------------------------------------------------------------------------
# 7. count_adapter_parameters > 0
# ---------------------------------------------------------------------------
def test_count_adapter_parameters_positive():
    model = nn.Sequential(
        nn.Linear(D, D),
        BottleneckAdapter(D, BN),
    )
    count = count_adapter_parameters(model)
    assert count > 0, "Expected at least one adapter parameter"


# ---------------------------------------------------------------------------
# 8. freeze_base_parameters makes non-adapter params frozen
# ---------------------------------------------------------------------------
def test_freeze_base_parameters():
    linear = nn.Linear(D, D)
    adapter = BottleneckAdapter(D, BN)
    model = nn.Sequential(linear, adapter)

    freeze_base_parameters(model)

    # linear parameters should be frozen
    assert not linear.weight.requires_grad, "Base linear weight should be frozen"
    assert not linear.bias.requires_grad, "Base linear bias should be frozen"

    # adapter parameters should still be trainable
    for p in adapter.parameters():
        assert p.requires_grad, "Adapter parameters should remain trainable"


# ---------------------------------------------------------------------------
# 9. AdapterModel forward works end-to-end
# ---------------------------------------------------------------------------
def test_adapter_model_forward():
    # Use Sequential so the Linear is a child that _replace_linears can wrap
    base = nn.Sequential(nn.Linear(D, D))
    cfg = AdapterConfig(d_model=D, bottleneck_dim=BN, init_scale=1e-3)
    model = AdapterModel(base, cfg, n_layers=1)

    x = torch.randn(B, T, D)
    out = model(x)
    assert out.shape == (B, T, D), f"Expected ({B},{T},{D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 10. get_adapter_state_dict keys not empty
# ---------------------------------------------------------------------------
def test_get_adapter_state_dict_nonempty():
    base = nn.Sequential(nn.Linear(D, D))
    cfg = AdapterConfig(d_model=D, bottleneck_dim=BN, init_scale=1e-3)
    model = AdapterModel(base, cfg, n_layers=1)

    sd = model.get_adapter_state_dict()
    assert len(sd) > 0, "Adapter state dict should not be empty"
    # All keys should correspond to tensor values
    for k, v in sd.items():
        assert isinstance(v, torch.Tensor), f"State dict value for '{k}' is not a Tensor"


# ---------------------------------------------------------------------------
# 11. compute_adapter_efficiency in (0, 1)
# ---------------------------------------------------------------------------
def test_compute_adapter_efficiency_range():
    eff = compute_adapter_efficiency(base_params=1000, adapter_params=10)
    assert 0.0 < eff < 1.0, f"Efficiency {eff} not in (0, 1)"


# ---------------------------------------------------------------------------
# 12. BottleneckAdapter output is finite
# ---------------------------------------------------------------------------
def test_bottleneck_adapter_output_finite():
    adapter = BottleneckAdapter(D, BN)
    x = torch.randn(B, T, D)
    out = adapter(x)
    assert torch.isfinite(out).all(), "BottleneckAdapter output contains non-finite values"


# ---------------------------------------------------------------------------
# 13. Near-zero init: |adapter(x)| << |x|  (explicit delta magnitude check)
# ---------------------------------------------------------------------------
def test_near_zero_init_delta_magnitude():
    adapter = BottleneckAdapter(D, BN, init_scale=1e-3)
    x = torch.ones(B, T, D)  # unit input for reproducibility
    with torch.no_grad():
        out = adapter(x)
    delta = (out - x).abs().mean().item()
    x_mean = x.abs().mean().item()
    assert delta < 0.1 * x_mean, (
        f"Mean absolute adapter delta ({delta:.6f}) should be << mean |x| ({x_mean:.6f})"
    )


# ---------------------------------------------------------------------------
# 14. AdapterModel: only adapter params have requires_grad=True
# ---------------------------------------------------------------------------
def test_adapter_model_only_adapter_trainable():
    # Two-layer Sequential so both Linears get wrapped
    base = nn.Sequential(nn.Linear(D, D), nn.Linear(D, D))
    cfg = AdapterConfig(d_model=D, bottleneck_dim=BN, init_scale=1e-3)
    model = AdapterModel(base, cfg, n_layers=2)

    for name, p in model.named_parameters():
        if p.requires_grad:
            # Trainable params must belong to an adapter
            assert "adapter" in name, f"Non-adapter param '{name}' is trainable after freeze"


# ---------------------------------------------------------------------------
# 15. ParallelAdapter near-zero init output magnitude
# ---------------------------------------------------------------------------
def test_parallel_adapter_near_zero_output():
    adapter = ParallelAdapter(D, BN, init_scale=1e-3)
    x = torch.ones(B, T, D)
    with torch.no_grad():
        out = adapter(x)
    out_norm = out.norm().item()
    x_norm = x.norm().item()
    assert out_norm < 0.1 * x_norm, (
        f"ParallelAdapter output norm ({out_norm:.6f}) should be << input norm ({x_norm:.6f})"
    )
