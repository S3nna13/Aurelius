"""Tests for IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.alignment.ia3 import (
    IA3Config,
    IA3Layer,
    IA3ScaledLinear,
    IA3Trainer,
    count_total_parameters,
    count_trainable_parameters,
    get_ia3_parameters,
    inject_ia3_layers,
    load_ia3_weights,
    save_ia3_weights,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared tiny model fixture
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


@pytest.fixture()
def tiny_model() -> AureliusTransformer:
    """Fresh tiny model for each test."""
    return AureliusTransformer(TINY_CFG)


@pytest.fixture()
def injected(tiny_model):
    """Tiny model with IA3 already injected."""
    cfg = IA3Config()
    ia3_layers = inject_ia3_layers(tiny_model, cfg)
    return tiny_model, ia3_layers


# ---------------------------------------------------------------------------
# 1. IA3Config defaults
# ---------------------------------------------------------------------------


def test_ia3_config_defaults():
    cfg = IA3Config()
    assert cfg.target_modules == ["k_proj", "v_proj", "down_proj"]
    assert cfg.init_ia3_weights is True
    assert cfg.trainable_only is True


# ---------------------------------------------------------------------------
# 2. IA3Layer initialised to ones
# ---------------------------------------------------------------------------


def test_ia3_layer_init_ones():
    layer = IA3Layer(64)
    assert layer.scale.shape == (64,)
    assert torch.all(layer.scale == 1.0)


# ---------------------------------------------------------------------------
# 3. IA3Layer forward preserves shape
# ---------------------------------------------------------------------------


def test_ia3_layer_forward_shape():
    layer = IA3Layer(64)
    x = torch.randn(2, 10, 64)
    out = layer(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 4. IA3Layer forward — known scale vector scales output correctly
# ---------------------------------------------------------------------------


def test_ia3_layer_forward_scales():
    d = 8
    layer = IA3Layer(d)
    scale_vals = torch.arange(1, d + 1, dtype=torch.float32)
    layer.scale.data.copy_(scale_vals)

    x = torch.ones(1, 4, d)
    out = layer(x)
    # Each position should equal scale_vals
    expected = scale_vals.unsqueeze(0).unsqueeze(0).expand(1, 4, d)
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# 5. IA3ScaledLinear forward preserves shape
# ---------------------------------------------------------------------------


def test_ia3_scaled_linear_shape():
    linear = nn.Linear(128, 64, bias=False)
    ia3_layer = IA3Layer(64)
    scaled = IA3ScaledLinear(linear, ia3_layer)

    x = torch.randn(2, 5, 128)
    out = scaled(x)
    assert out.shape == (2, 5, 64)


# ---------------------------------------------------------------------------
# 6. inject_ia3_layers returns dict with n_layers entries
# ---------------------------------------------------------------------------


def test_inject_ia3_layers_count(tiny_model):
    cfg = IA3Config()
    ia3_layers = inject_ia3_layers(tiny_model, cfg)
    assert len(ia3_layers) == TINY_CFG.n_layers


# ---------------------------------------------------------------------------
# 7. inject_ia3_layers freezes base model params
# ---------------------------------------------------------------------------


def test_inject_ia3_layers_freezes_base(tiny_model):
    cfg = IA3Config()
    inject_ia3_layers(tiny_model, cfg)
    trainable = count_trainable_parameters(tiny_model)
    total = count_total_parameters(tiny_model)
    # IA3 params are tiny fraction — much less than total
    assert trainable < total
    # Trainable should be only the IA3 scales
    expected_ia3 = TINY_CFG.n_layers * TINY_CFG.d_model
    assert trainable == expected_ia3


# ---------------------------------------------------------------------------
# 8. inject replaces down_proj with IA3ScaledLinear
# ---------------------------------------------------------------------------


def test_inject_ia3_layers_replaces_down_proj(tiny_model):
    cfg = IA3Config()
    inject_ia3_layers(tiny_model, cfg)
    assert isinstance(tiny_model.layers[0].ffn.down_proj, IA3ScaledLinear)
    assert isinstance(tiny_model.layers[1].ffn.down_proj, IA3ScaledLinear)


# ---------------------------------------------------------------------------
# 9. count_trainable_vs_total
# ---------------------------------------------------------------------------


def test_count_trainable_vs_total(tiny_model):
    cfg = IA3Config()
    inject_ia3_layers(tiny_model, cfg)
    trainable = count_trainable_parameters(tiny_model)
    total = count_total_parameters(tiny_model)
    assert trainable < total


# ---------------------------------------------------------------------------
# 10. get_ia3_parameters length
# ---------------------------------------------------------------------------


def test_get_ia3_parameters_length(injected):
    _, ia3_layers = injected
    params = get_ia3_parameters(ia3_layers)
    assert len(params) == TINY_CFG.n_layers


# ---------------------------------------------------------------------------
# 11. IA3Trainer.train_step returns correct keys
# ---------------------------------------------------------------------------


def test_ia3_trainer_train_step_keys(injected):
    model, ia3_layers = injected
    optimizer = torch.optim.Adam(get_ia3_parameters(ia3_layers), lr=1e-3)
    trainer = IA3Trainer(model, ia3_layers, optimizer)

    input_ids = torch.randint(0, TINY_CFG.vocab_size, (2, 16))
    labels = torch.randint(0, TINY_CFG.vocab_size, (2, 16))
    result = trainer.train_step(input_ids, labels)

    assert "loss" in result
    assert "n_ia3_params" in result


# ---------------------------------------------------------------------------
# 12. IA3Trainer loss is positive
# ---------------------------------------------------------------------------


def test_ia3_trainer_loss_positive(injected):
    model, ia3_layers = injected
    optimizer = torch.optim.Adam(get_ia3_parameters(ia3_layers), lr=1e-3)
    trainer = IA3Trainer(model, ia3_layers, optimizer)

    input_ids = torch.randint(0, TINY_CFG.vocab_size, (2, 16))
    labels = torch.randint(0, TINY_CFG.vocab_size, (2, 16))
    result = trainer.train_step(input_ids, labels)

    assert result["loss"] > 0.0


# ---------------------------------------------------------------------------
# 13. save and load ia3 weights
# ---------------------------------------------------------------------------


def test_save_load_ia3_weights(injected, tmp_path):
    _, ia3_layers = injected
    # Give each scale a known non-one value so we can verify round-trip
    for layer in ia3_layers.values():
        nn.init.uniform_(layer.scale, 0.5, 1.5)

    original_values = {name: layer.scale.data.clone() for name, layer in ia3_layers.items()}

    path = str(tmp_path / "ia3_weights.pt")
    save_ia3_weights(ia3_layers, path)

    # Overwrite scales with zeros
    for layer in ia3_layers.values():
        layer.scale.data.zero_()

    # Load back
    load_ia3_weights(ia3_layers, path)

    for name, layer in ia3_layers.items():
        assert torch.allclose(layer.scale.data, original_values[name], atol=1e-6), (
            f"Scale mismatch for {name} after save/load"
        )


# ---------------------------------------------------------------------------
# 14. merge_into_model doesn't crash
# ---------------------------------------------------------------------------


def test_merge_into_model_runs(injected):
    model, ia3_layers = injected
    optimizer = torch.optim.Adam(get_ia3_parameters(ia3_layers), lr=1e-3)
    trainer = IA3Trainer(model, ia3_layers, optimizer)
    trainer.merge_into_model()  # should not raise


# ---------------------------------------------------------------------------
# 15. IA3 parameter efficiency
# ---------------------------------------------------------------------------


def test_ia3_parameter_efficiency(injected):
    model, ia3_layers = injected
    n_ia3 = sum(p.numel() for p in get_ia3_parameters(ia3_layers))
    n_total = count_total_parameters(model)
    ratio = n_ia3 / n_total
    assert ratio < 0.01, f"IA3 ratio {ratio:.4f} >= 0.01 — not parameter-efficient enough"
