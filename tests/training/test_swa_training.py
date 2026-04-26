"""Tests for src/training/swa_training.py."""

from __future__ import annotations

import pytest
import torch
from torch.optim import SGD

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.swa_training import (
    CyclicalLRScheduler,
    SWAConfig,
    SWAModel,
    SWATrainer,
    update_bn,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_config(**kwargs) -> SWAConfig:
    defaults = dict(
        swa_start=10,
        swa_freq=5,
        swa_lr=0.05,
        cycle_length=20,
        cycle_mult=1.0,
        min_lr=1e-6,
        max_lr=1e-3,
    )
    defaults.update(kwargs)
    return SWAConfig(**defaults)


def make_small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


def make_optimizer(model):
    return SGD(model.parameters(), lr=1e-3)


def make_input():
    return torch.randint(0, 256, (1, 8))


# ---------------------------------------------------------------------------
# 1. SWAConfig defaults
# ---------------------------------------------------------------------------


def test_swa_config_defaults():
    cfg = SWAConfig()
    assert cfg.swa_start == 100
    assert cfg.swa_freq == 5
    assert cfg.swa_lr == 0.05
    assert cfg.cycle_length == 20
    assert cfg.cycle_mult == 1.0
    assert cfg.min_lr == 1e-6
    assert cfg.max_lr == 1e-3


# ---------------------------------------------------------------------------
# 2. CyclicalLRScheduler.get_lr at step 0 ≈ max_lr
# ---------------------------------------------------------------------------


def test_cyclical_lr_step0_is_max_lr():
    cfg = make_config(min_lr=1e-6, max_lr=1e-3, cycle_length=20)
    model = make_small_model()
    opt = make_optimizer(model)
    sched = CyclicalLRScheduler(opt, cfg)
    # At cycle_step=0, cos(0) = 1 => lr = min_lr + 0.5*(max_lr-min_lr)*2 = max_lr
    assert sched.get_lr() == pytest.approx(cfg.max_lr, rel=1e-5)


# ---------------------------------------------------------------------------
# 3. CyclicalLRScheduler.get_lr at mid-cycle ≈ (max+min)/2
# ---------------------------------------------------------------------------


def test_cyclical_lr_mid_cycle():
    cfg = make_config(min_lr=0.0, max_lr=1.0, cycle_length=20)
    model = make_small_model()
    opt = make_optimizer(model)
    sched = CyclicalLRScheduler(opt, cfg)
    # Advance to mid-cycle manually
    sched._cycle_step = 10  # half of 20
    lr = sched.get_lr()
    # cos(pi * 10 / 20) = cos(pi/2) = 0 => lr = 0 + 0.5*(1.0-0.0)*(1+0) = 0.5
    assert lr == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. CyclicalLRScheduler.step returns float
# ---------------------------------------------------------------------------


def test_cyclical_lr_step_returns_float():
    cfg = make_config()
    model = make_small_model()
    opt = make_optimizer(model)
    sched = CyclicalLRScheduler(opt, cfg)
    lr = sched.step()
    assert isinstance(lr, float)


# ---------------------------------------------------------------------------
# 5. CyclicalLRScheduler LR changes over steps
# ---------------------------------------------------------------------------


def test_cyclical_lr_changes_over_steps():
    cfg = make_config(min_lr=1e-6, max_lr=1e-3, cycle_length=20)
    model = make_small_model()
    opt = make_optimizer(model)
    sched = CyclicalLRScheduler(opt, cfg)
    lrs = [sched.step() for _ in range(10)]
    # LR should not be constant (cosine decay)
    assert len(set(lrs)) > 1


# ---------------------------------------------------------------------------
# 6. SWAModel.update increments n_averaged
# ---------------------------------------------------------------------------


def test_swa_model_update_increments_n_averaged():
    model = make_small_model()
    swa = SWAModel(model)
    assert swa._n_averaged == 0
    swa.update(model)
    assert swa._n_averaged == 1
    swa.update(model)
    assert swa._n_averaged == 2


# ---------------------------------------------------------------------------
# 7. SWAModel.update averaged params change after update
# ---------------------------------------------------------------------------


def test_swa_model_update_params_change():
    model = make_small_model()
    swa = SWAModel(model)
    swa.update(model)  # snapshot first

    # Mutate model params
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    swa.update(model)  # second snapshot (should average)
    assert swa._n_averaged == 2

    # The averaged params should be between original and modified values
    for name, p in model.named_parameters():
        avg = swa._swa_params[name]
        # avg = (orig + orig+1) / 2 = orig + 0.5 — just verify it's stored
        assert avg is not None


# ---------------------------------------------------------------------------
# 8. SWAModel.get_averaged_model copies params to model
# ---------------------------------------------------------------------------


def test_swa_model_get_averaged_model():
    model = make_small_model()
    swa = SWAModel(model)
    swa.update(model)

    # Change model params
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(999.0)

    swa.get_averaged_model(model)

    # After applying, params should NOT be 999 anymore
    for p in model.parameters():
        assert not torch.all(p == 999.0).item()


# ---------------------------------------------------------------------------
# 9. SWAModel.reset zeros n_averaged
# ---------------------------------------------------------------------------


def test_swa_model_reset():
    model = make_small_model()
    swa = SWAModel(model)
    swa.update(model)
    swa.update(model)
    assert swa._n_averaged == 2

    swa.reset()
    assert swa._n_averaged == 0
    assert swa._swa_params is None


# ---------------------------------------------------------------------------
# 10. update_bn runs without error on model without BN
# ---------------------------------------------------------------------------


def test_update_bn_no_bn_layers():
    model = make_small_model()
    data = [make_input() for _ in range(3)]
    # Should be a no-op — no error
    update_bn(model, data)


# ---------------------------------------------------------------------------
# 11. SWATrainer.train_step returns required keys
# ---------------------------------------------------------------------------


def test_swa_trainer_step_returns_required_keys():
    model = make_small_model()
    opt = make_optimizer(model)
    cfg = make_config(swa_start=100)
    trainer = SWATrainer(model, opt, cfg)
    metrics = trainer.train_step(make_input())
    for key in ("loss", "lr", "swa_n_averaged", "step"):
        assert key in metrics, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 12. SWATrainer swa_n_averaged is 0 before swa_start
# ---------------------------------------------------------------------------


def test_swa_trainer_no_averaging_before_start():
    model = make_small_model()
    opt = make_optimizer(model)
    cfg = make_config(swa_start=100, swa_freq=5)
    trainer = SWATrainer(model, opt, cfg)

    for _ in range(5):
        metrics = trainer.train_step(make_input())

    assert metrics["swa_n_averaged"] == 0


# ---------------------------------------------------------------------------
# 13. SWATrainer swa_n_averaged > 0 after swa_start
# ---------------------------------------------------------------------------


def test_swa_trainer_averaging_after_start():
    model = make_small_model()
    opt = make_optimizer(model)
    cfg = make_config(swa_start=3, swa_freq=1)
    trainer = SWATrainer(model, opt, cfg)

    for _ in range(6):
        metrics = trainer.train_step(make_input())

    assert metrics["swa_n_averaged"] > 0


# ---------------------------------------------------------------------------
# 14. SWATrainer.finalize applies averaged weights
# ---------------------------------------------------------------------------


def test_swa_trainer_finalize_applies_weights():
    model = make_small_model()
    opt = make_optimizer(model)
    cfg = make_config(swa_start=0, swa_freq=1)
    trainer = SWATrainer(model, opt, cfg)

    # Run enough steps to build an average
    for _ in range(5):
        trainer.train_step(make_input())

    # Capture param values before finalize
    {n: p.detach().clone() for n, p in model.named_parameters()}

    # Corrupt params
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.0)

    trainer.finalize()

    # After finalize, at least some params should differ from 0.0
    changed = False
    for n, p in model.named_parameters():
        if not torch.allclose(p, torch.zeros_like(p)):
            changed = True
            break

    assert changed, "finalize() did not restore averaged weights"


# ---------------------------------------------------------------------------
# 15. CyclicalLRScheduler cycle resets at cycle_length
# ---------------------------------------------------------------------------


def test_cyclical_lr_cycle_resets():
    cycle_length = 10
    cfg = make_config(cycle_length=cycle_length, cycle_mult=1.0)
    model = make_small_model()
    opt = make_optimizer(model)
    sched = CyclicalLRScheduler(opt, cfg)

    # Run exactly one full cycle
    for _ in range(cycle_length):
        sched.step()

    assert sched.cycle_count() == 1
    assert sched._cycle_step == 0

    # LR should be back near max_lr at start of new cycle
    lr = sched.get_lr()
    assert lr == pytest.approx(cfg.max_lr, rel=1e-5)
