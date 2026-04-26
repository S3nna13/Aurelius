"""Tests for progressive layer dropping."""

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.layer_drop import (
    LayerDropConfig,
    LayerDropTrainer,
    apply_layer_drop,
    compute_drop_rate,
    estimate_speedup,
    layer_drop_mask,
)


def _tiny_config():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


def _make_model():
    cfg = _tiny_config()
    return AureliusTransformer(cfg)


# ── LayerDropConfig tests ──


def test_config_defaults():
    cfg = LayerDropConfig()
    assert cfg.drop_rate == 0.2
    assert cfg.schedule == "linear"
    assert cfg.warmup_steps == 100


def test_config_custom():
    cfg = LayerDropConfig(drop_rate=0.5, schedule="cosine", warmup_steps=50)
    assert cfg.drop_rate == 0.5
    assert cfg.schedule == "cosine"
    assert cfg.warmup_steps == 50


# ── compute_drop_rate tests ──


def test_constant_schedule():
    cfg = LayerDropConfig(drop_rate=0.3, schedule="constant", warmup_steps=0)
    # After warmup, constant should always return drop_rate
    assert compute_drop_rate(0, 100, cfg) == 0.3
    assert compute_drop_rate(50, 100, cfg) == 0.3
    assert compute_drop_rate(99, 100, cfg) == 0.3


def test_linear_schedule_increases():
    cfg = LayerDropConfig(drop_rate=0.4, schedule="linear", warmup_steps=0)
    r0 = compute_drop_rate(0, 100, cfg)
    r50 = compute_drop_rate(50, 100, cfg)
    r99 = compute_drop_rate(99, 100, cfg)
    assert r0 == pytest.approx(0.0, abs=1e-6)
    assert r50 > r0
    assert r99 > r50
    assert r99 == pytest.approx(0.4 * 99 / 100, abs=1e-6)


def test_cosine_schedule():
    cfg = LayerDropConfig(drop_rate=0.2, schedule="cosine", warmup_steps=0)
    r0 = compute_drop_rate(0, 100, cfg)
    r100 = compute_drop_rate(100, 100, cfg)
    assert r0 == pytest.approx(0.0, abs=1e-6)
    assert r100 == pytest.approx(0.2, abs=1e-6)


def test_warmup_returns_zero():
    cfg = LayerDropConfig(drop_rate=0.5, schedule="constant", warmup_steps=100)
    assert compute_drop_rate(0, 1000, cfg) == 0.0
    assert compute_drop_rate(50, 1000, cfg) == 0.0
    assert compute_drop_rate(99, 1000, cfg) == 0.0
    # After warmup
    assert compute_drop_rate(100, 1000, cfg) == 0.5


def test_invalid_schedule():
    cfg = LayerDropConfig(schedule="banana", warmup_steps=0)
    with pytest.raises(ValueError, match="Unknown schedule"):
        compute_drop_rate(50, 100, cfg)


def test_zero_total_steps():
    cfg = LayerDropConfig(drop_rate=0.3, schedule="linear", warmup_steps=0)
    assert compute_drop_rate(0, 0, cfg) == 0.0


# ── layer_drop_mask tests ──


def test_mask_keeps_all_in_mode():
    mask = layer_drop_mask(10, 0.5, training=False)
    assert all(mask)
    assert len(mask) == 10


def test_mask_keeps_first_and_last():
    torch.manual_seed(0)
    # With very high drop rate, first and last should still be kept
    mask = layer_drop_mask(10, 0.99, training=True)
    assert mask[0] is True
    assert mask[-1] is True


def test_mask_zero_drop_rate():
    mask = layer_drop_mask(6, 0.0, training=True)
    assert all(mask)


def test_mask_empty():
    mask = layer_drop_mask(0, 0.5, training=True)
    assert mask == []


def test_mask_length():
    mask = layer_drop_mask(8, 0.3, training=True)
    assert len(mask) == 8


def test_mask_drops_some_layers():
    """With high drop rate and many layers, statistically some should be dropped."""
    torch.manual_seed(42)
    dropped_any = False
    for _ in range(20):
        mask = layer_drop_mask(20, 0.9, training=True)
        if not all(mask):
            dropped_any = True
            break
    assert dropped_any, "With 0.9 drop rate over 20 layers, expected some drops"


# ── apply_layer_drop tests ──


def test_apply_layer_drop_full_mask():
    model = _make_model()
    model.train(False)
    input_ids = torch.randint(0, 256, (1, 8))
    mask = [True] * len(model.layers)
    model._layer_drop_input_ids = input_ids
    logits = apply_layer_drop(model, mask)
    assert logits.shape == (1, 8, 256)


def test_apply_layer_drop_output_shape():
    model = _make_model()
    input_ids = torch.randint(0, 256, (2, 16))
    # Drop nothing
    mask = [True, True]
    model._layer_drop_input_ids = input_ids
    logits = apply_layer_drop(model, mask)
    assert logits.shape == (2, 16, 256)


# ── LayerDropTrainer tests ──


def test_trainer_train_step():
    model = _make_model()
    cfg = LayerDropConfig(drop_rate=0.0, schedule="constant", warmup_steps=0)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = LayerDropTrainer(model, cfg, opt)
    trainer.total_steps = 100

    input_ids = torch.randint(0, 256, (2, 16))
    result = trainer.train_step(input_ids)

    assert "loss" in result
    assert "n_active_layers" in result
    assert "drop_rate" in result
    assert isinstance(result["loss"], float)
    assert result["n_active_layers"] == 2  # tiny config has 2 layers
    assert result["drop_rate"] == 0.0


def test_trainer_step_count_increments():
    model = _make_model()
    cfg = LayerDropConfig(drop_rate=0.0, schedule="constant", warmup_steps=0)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = LayerDropTrainer(model, cfg, opt)

    input_ids = torch.randint(0, 256, (1, 8))
    trainer.train_step(input_ids)
    assert trainer.step_count == 1
    trainer.train_step(input_ids)
    assert trainer.step_count == 2


def test_trainer_loss_decreases():
    """Loss should decrease over multiple steps with enough learning rate."""
    torch.manual_seed(0)
    model = _make_model()
    cfg = LayerDropConfig(drop_rate=0.0, schedule="constant", warmup_steps=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = LayerDropTrainer(model, cfg, opt)
    trainer.total_steps = 50

    input_ids = torch.randint(0, 256, (2, 32))
    first_loss = trainer.train_step(input_ids)["loss"]
    for _ in range(19):
        result = trainer.train_step(input_ids)
    last_loss = result["loss"]
    assert last_loss < first_loss


# ── estimate_speedup tests ──


def test_estimate_speedup_no_drop():
    assert estimate_speedup(24, 0.0) == pytest.approx(1.0)


def test_estimate_speedup_half_drop():
    assert estimate_speedup(24, 0.5) == pytest.approx(2.0)


def test_estimate_speedup_formula():
    # n_layers / (n_layers * (1 - drop_rate)) = 1 / (1 - drop_rate)
    assert estimate_speedup(10, 0.2) == pytest.approx(1.25)


def test_estimate_speedup_full_drop():
    assert estimate_speedup(10, 1.0) == float("inf")
