"""Tests for src/training/mixed_precision.py."""

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.mixed_precision import (
    DynamicLossScaler,
    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    autocast_forward,
    count_overflow_params,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SMALL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

_INPUT_SHAPE = (1, 8)


def _make_model():
    return AureliusTransformer(_SMALL_CFG)


def _make_trainer(dtype="bfloat16"):
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    cfg = MixedPrecisionConfig(enabled=True, dtype=dtype)
    trainer = MixedPrecisionTrainer(model, optimizer, cfg)
    return trainer, model, optimizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mp_config_defaults():
    cfg = MixedPrecisionConfig()
    assert cfg.enabled is True
    assert cfg.dtype == "float16"
    assert cfg.initial_scale == 2**16
    assert cfg.growth_factor == 2.0
    assert cfg.backoff_factor == 0.5
    assert cfg.growth_interval == 2000
    assert cfg.min_scale == 1.0


def test_dynamic_loss_scaler_scale_loss():
    cfg = MixedPrecisionConfig(initial_scale=128.0)
    scaler = DynamicLossScaler(cfg)
    loss = torch.tensor(3.0)
    scaled = scaler.scale_loss(loss)
    assert scaled.item() == pytest.approx(3.0 * 128.0)


def test_dynamic_loss_scaler_no_overflow_grows():
    cfg = MixedPrecisionConfig(
        initial_scale=256.0,
        growth_factor=2.0,
        growth_interval=5,
        min_scale=1.0,
    )
    scaler = DynamicLossScaler(cfg)
    initial = scaler.scale

    for _ in range(cfg.growth_interval):
        scaler.update(overflow=False)

    assert scaler.scale == pytest.approx(initial * cfg.growth_factor)


def test_dynamic_loss_scaler_overflow_shrinks():
    cfg = MixedPrecisionConfig(
        initial_scale=256.0,
        backoff_factor=0.5,
        min_scale=1.0,
    )
    scaler = DynamicLossScaler(cfg)
    initial = scaler.scale

    scaler.update(overflow=True)

    assert scaler.scale == pytest.approx(initial * cfg.backoff_factor)
    assert scaler.overflow_count == 1


def test_dynamic_loss_scaler_min_scale():
    cfg = MixedPrecisionConfig(
        initial_scale=2.0,
        backoff_factor=0.5,
        min_scale=1.0,
    )
    scaler = DynamicLossScaler(cfg)

    for _ in range(10):
        scaler.update(overflow=True)

    assert scaler.scale >= cfg.min_scale


def test_unscale_gradients_divides():
    model = nn.Linear(4, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight) * 4.0

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = MixedPrecisionConfig(initial_scale=2.0)
    scaler = DynamicLossScaler(cfg)

    finite = scaler.unscale_gradients(optimizer)

    assert finite is True
    assert model.weight.grad.allclose(torch.full_like(model.weight.grad, 2.0))


def test_unscale_gradients_detects_nan():
    model = nn.Linear(4, 1, bias=False)
    grad = torch.ones_like(model.weight)
    grad[0, 0] = float("nan")
    model.weight.grad = grad

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = MixedPrecisionConfig(initial_scale=1.0)
    scaler = DynamicLossScaler(cfg)

    finite = scaler.unscale_gradients(optimizer)

    assert finite is False


def test_mp_trainer_forward_backward_keys():
    trainer, model, _ = _make_trainer(dtype="bfloat16")
    model.train()

    input_ids = torch.randint(0, _SMALL_CFG.vocab_size, _INPUT_SHAPE)
    labels = torch.randint(0, _SMALL_CFG.vocab_size, _INPUT_SHAPE)

    result = trainer.forward_backward(input_ids, labels)

    assert "loss" in result
    assert "scale" in result
    assert "overflow" in result
    assert isinstance(result["loss"], float)
    assert isinstance(result["scale"], float)
    assert isinstance(result["overflow"], bool)


def test_mp_trainer_cast_inputs_dtype():
    cfg = MixedPrecisionConfig(enabled=True, dtype="bfloat16")
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = MixedPrecisionTrainer(model, optimizer, cfg)

    x = torch.randn(2, 4)
    (x_cast,) = trainer.cast_inputs(x)

    assert x_cast.dtype == torch.bfloat16


def test_count_overflow_params_zero():
    model = nn.Linear(8, 4, bias=True)
    for p in model.parameters():
        p.grad = torch.ones_like(p)

    assert count_overflow_params(model) == 0


def test_autocast_forward_output_shape():
    model = _make_model()
    model.eval()

    B, T = _INPUT_SHAPE
    input_ids = torch.randint(0, _SMALL_CFG.vocab_size, _INPUT_SHAPE)

    with torch.no_grad():
        loss, logits, pkv = autocast_forward(model, input_ids, dtype=torch.bfloat16)

    assert logits.shape == (B, T, _SMALL_CFG.vocab_size)
    for p in model.parameters():
        assert p.dtype == torch.float32
