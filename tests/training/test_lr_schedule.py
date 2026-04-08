import pytest
import math
from src.training.lr_schedule import SGDRConfig, SGDRScheduler

def test_warmup_starts_at_zero():
    cfg = SGDRConfig(lr_max=1e-3, warmup_steps=10, T0=100)
    s = SGDRScheduler(cfg)
    assert s.get_lr(0) == pytest.approx(0.0)

def test_warmup_ends_at_lr_max():
    cfg = SGDRConfig(lr_max=1e-3, warmup_steps=10, T0=100)
    s = SGDRScheduler(cfg)
    # At step warmup_steps - 1 should be close to lr_max
    # At step warmup_steps exactly, cosine phase begins (not warmup)
    lr = s.get_lr(9)
    assert lr == pytest.approx(1e-3 * 9 / 10)

def test_cosine_at_cycle_start_is_lr_max():
    cfg = SGDRConfig(lr_max=1e-3, lr_min=1e-5, warmup_steps=0, T0=100)
    s = SGDRScheduler(cfg)
    lr = s.get_lr(0)
    assert lr == pytest.approx(1e-3, rel=1e-4)

def test_cosine_at_cycle_end_is_lr_min():
    cfg = SGDRConfig(lr_max=1e-3, lr_min=1e-5, warmup_steps=0, T0=100)
    s = SGDRScheduler(cfg)
    lr = s.get_lr(99)  # last step of first cycle (t ≈ 1.0)
    # cos(pi * 99/100) ≈ -1 → lr_min
    assert lr == pytest.approx(1e-5, rel=0.05)

def test_restart_resets_to_lr_max():
    cfg = SGDRConfig(lr_max=1e-3, lr_min=1e-5, warmup_steps=0, T0=100, T_mult=1.0)
    s = SGDRScheduler(cfg)
    lr_after_restart = s.get_lr(100)  # start of second cycle
    assert lr_after_restart == pytest.approx(1e-3, rel=1e-4)

def test_decay_factor_reduces_peak():
    cfg = SGDRConfig(lr_max=1e-3, lr_min=1e-5, warmup_steps=0, T0=100, T_mult=1.0, decay_factor=0.5)
    s = SGDRScheduler(cfg)
    lr_cycle2_start = s.get_lr(100)
    assert lr_cycle2_start == pytest.approx(5e-4, rel=1e-3)

def test_lr_is_monotone_in_warmup():
    cfg = SGDRConfig(lr_max=1e-3, warmup_steps=50, T0=200)
    s = SGDRScheduler(cfg)
    lrs = [s.get_lr(i) for i in range(50)]
    assert all(lrs[i] <= lrs[i+1] for i in range(len(lrs)-1))

def test_get_cycle_info():
    cfg = SGDRConfig(lr_max=1e-3, warmup_steps=0, T0=100, T_mult=2.0)
    s = SGDRScheduler(cfg)
    info = s.get_cycle_info(0)
    assert info["cycle"] == 0
    assert info["cycle_len"] == 100
    info2 = s.get_cycle_info(150)  # into second cycle (len=200)
    assert info2["cycle"] == 1
    assert info2["cycle_len"] == 200
