from __future__ import annotations

import math

import pytest

from src.training.privacy_accountant import PrivacyAccountant, PrivacyBudget


# --- PrivacyBudget dataclass ---

def test_privacy_budget_fields():
    pb = PrivacyBudget(epsilon=1.0, delta=1e-5, mechanism="gaussian")
    assert pb.epsilon == 1.0
    assert pb.delta == 1e-5
    assert pb.mechanism == "gaussian"


# --- PrivacyAccountant construction ---

def test_default_construction():
    pa = PrivacyAccountant()
    assert pa.delta == 1e-5
    assert pa.mechanism == "gaussian"


def test_invalid_delta_zero():
    with pytest.raises(ValueError, match="delta"):
        PrivacyAccountant(delta=0.0)


def test_invalid_delta_one():
    with pytest.raises(ValueError, match="delta"):
        PrivacyAccountant(delta=1.0)


def test_invalid_mechanism():
    with pytest.raises(ValueError, match="mechanism"):
        PrivacyAccountant(mechanism="unknown")


def test_valid_mechanisms():
    for m in ("gaussian", "laplace", "rdp_gaussian"):
        pa = PrivacyAccountant(mechanism=m)
        assert pa.mechanism == m


# --- accumulate ---

def test_accumulate_returns_budget():
    pa = PrivacyAccountant()
    budget = pa.accumulate(noise_multiplier=1.1, sample_rate=0.01)
    assert isinstance(budget, PrivacyBudget)


def test_accumulate_increases_epsilon():
    pa = PrivacyAccountant()
    b1 = pa.accumulate(noise_multiplier=1.1, sample_rate=0.01, steps=10)
    b2 = pa.accumulate(noise_multiplier=1.1, sample_rate=0.01, steps=10)
    assert b2.epsilon > b1.epsilon


def test_accumulate_invalid_noise_multiplier():
    pa = PrivacyAccountant()
    with pytest.raises(ValueError, match="noise_multiplier"):
        pa.accumulate(noise_multiplier=0.0, sample_rate=0.01)


def test_accumulate_invalid_sample_rate_zero():
    pa = PrivacyAccountant()
    with pytest.raises(ValueError, match="sample_rate"):
        pa.accumulate(noise_multiplier=1.1, sample_rate=0.0)


def test_accumulate_invalid_sample_rate_above_one():
    pa = PrivacyAccountant()
    with pytest.raises(ValueError, match="sample_rate"):
        pa.accumulate(noise_multiplier=1.1, sample_rate=1.5)


def test_accumulate_invalid_steps():
    pa = PrivacyAccountant()
    with pytest.raises(ValueError, match="steps"):
        pa.accumulate(noise_multiplier=1.1, sample_rate=0.01, steps=0)


def test_accumulate_epsilon_formula():
    pa = PrivacyAccountant(delta=1e-5)
    steps = 100
    sample_rate = 0.01
    noise_multiplier = 1.5
    pa.accumulate(noise_multiplier=noise_multiplier, sample_rate=sample_rate, steps=steps)
    per_step = math.sqrt(2.0 * math.log(1.25 / 1e-5)) / noise_multiplier
    expected = per_step * math.sqrt(steps * sample_rate)
    assert abs(pa.get_budget().epsilon - expected) < 1e-9


# --- get_budget ---

def test_get_budget_zero_initially():
    pa = PrivacyAccountant()
    assert pa.get_budget().epsilon == 0.0


def test_get_budget_delta_preserved():
    pa = PrivacyAccountant(delta=1e-6)
    pa.accumulate(1.0, 0.1)
    assert pa.get_budget().delta == 1e-6


# --- reset ---

def test_reset_clears_epsilon():
    pa = PrivacyAccountant()
    pa.accumulate(1.1, 0.01, steps=100)
    pa.reset()
    assert pa.get_budget().epsilon == 0.0


def test_reset_clears_steps():
    pa = PrivacyAccountant()
    pa.accumulate(1.1, 0.01, steps=50)
    pa.reset()
    assert pa.report()["total_steps"] == 0


# --- steps_for_epsilon ---

def test_steps_for_epsilon_positive():
    pa = PrivacyAccountant()
    n = pa.steps_for_epsilon(target_epsilon=1.0, noise_multiplier=1.1, sample_rate=0.01)
    assert n > 0


def test_steps_for_epsilon_respects_budget():
    pa = PrivacyAccountant()
    target = 2.0
    noise_multiplier = 1.1
    sample_rate = 0.01
    n = pa.steps_for_epsilon(target, noise_multiplier, sample_rate)
    fresh = PrivacyAccountant()
    fresh.accumulate(noise_multiplier, sample_rate, steps=n)
    assert fresh.get_budget().epsilon <= target + 1e-6


def test_steps_for_epsilon_invalid_target():
    pa = PrivacyAccountant()
    with pytest.raises(ValueError, match="target_epsilon"):
        pa.steps_for_epsilon(target_epsilon=0.0, noise_multiplier=1.1, sample_rate=0.01)


def test_steps_for_epsilon_invalid_noise():
    pa = PrivacyAccountant()
    with pytest.raises(ValueError, match="noise_multiplier"):
        pa.steps_for_epsilon(target_epsilon=1.0, noise_multiplier=0.0, sample_rate=0.01)


def test_steps_for_epsilon_invalid_sample_rate():
    pa = PrivacyAccountant()
    with pytest.raises(ValueError, match="sample_rate"):
        pa.steps_for_epsilon(target_epsilon=1.0, noise_multiplier=1.1, sample_rate=0.0)


# --- report ---

def test_report_keys():
    pa = PrivacyAccountant()
    r = pa.report()
    assert set(r.keys()) == {"epsilon", "delta", "mechanism", "total_steps"}


def test_report_total_steps_accumulates():
    pa = PrivacyAccountant()
    pa.accumulate(1.1, 0.01, steps=10)
    pa.accumulate(1.1, 0.01, steps=20)
    assert pa.report()["total_steps"] == 30


def test_report_mechanism_echoed():
    pa = PrivacyAccountant(mechanism="rdp_gaussian")
    assert pa.report()["mechanism"] == "rdp_gaussian"
