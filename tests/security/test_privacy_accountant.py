"""Tests for privacy_accountant.py."""

from __future__ import annotations

import math

import pytest

from src.security.privacy_accountant import Mechanism, PrivacyAccountant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def accountant() -> PrivacyAccountant:
    return PrivacyAccountant(delta=1e-5)


@pytest.fixture()
def gaussian_step() -> Mechanism:
    return Mechanism(name="gaussian", sigma=1.0, sensitivity=1.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_instantiates_with_default_delta():
    """PrivacyAccountant can be created with default delta."""
    acc = PrivacyAccountant()
    assert acc.delta == 1e-5


def test_n_steps_zero_after_init(accountant):
    """n_steps is 0 immediately after construction."""
    assert accountant.n_steps == 0


def test_add_step_increases_n_steps(accountant, gaussian_step):
    """add_step increments n_steps by one per call."""
    accountant.add_step(gaussian_step)
    assert accountant.n_steps == 1
    accountant.add_step(gaussian_step)
    assert accountant.n_steps == 2


def test_compute_rdp_zero_for_no_steps(accountant):
    """compute_rdp returns 0.0 when no steps have been recorded."""
    assert accountant.compute_rdp(alpha=8.0) == 0.0


def test_compute_rdp_positive_after_step(accountant, gaussian_step):
    """compute_rdp is positive after at least one step."""
    accountant.add_step(gaussian_step)
    assert accountant.compute_rdp(alpha=8.0) > 0.0


def test_compute_rdp_scales_linearly(accountant, gaussian_step):
    """compute_rdp scales linearly with the number of identical steps."""
    alpha = 4.0
    accountant.add_step(gaussian_step)
    single = accountant.compute_rdp(alpha)

    for _ in range(4):
        accountant.add_step(gaussian_step)
    # Now 5 steps total
    assert math.isclose(accountant.compute_rdp(alpha), 5 * single, rel_tol=1e-9)


def test_rdp_gaussian_increases_with_smaller_sigma(accountant):
    """Smaller sigma (less noise) yields higher RDP cost."""
    high_noise = accountant._rdp_gaussian(alpha=4.0, sigma=2.0, sensitivity=1.0)
    low_noise = accountant._rdp_gaussian(alpha=4.0, sigma=0.5, sensitivity=1.0)
    assert low_noise > high_noise


def test_rdp_gaussian_increases_with_larger_alpha(accountant):
    """RDP cost increases with larger alpha for a fixed Gaussian mechanism."""
    low_alpha = accountant._rdp_gaussian(alpha=2.0, sigma=1.0, sensitivity=1.0)
    high_alpha = accountant._rdp_gaussian(alpha=16.0, sigma=1.0, sensitivity=1.0)
    assert high_alpha > low_alpha


def test_rdp_to_dp_returns_finite_positive(accountant, gaussian_step):
    """rdp_to_dp returns a finite positive float."""
    accountant.add_step(gaussian_step)
    rdp_eps = accountant.compute_rdp(alpha=8.0)
    eps = accountant.rdp_to_dp(rdp_eps, alpha=8.0)
    assert math.isfinite(eps)
    assert eps > 0.0


def test_get_epsilon_positive_after_steps(accountant, gaussian_step):
    """get_epsilon returns a positive float after steps are added."""
    accountant.add_step(gaussian_step)
    eps = accountant.get_epsilon(alpha=8.0)
    assert eps > 0.0


def test_reset_clears_all_steps(accountant, gaussian_step):
    """reset() brings n_steps back to 0."""
    for _ in range(5):
        accountant.add_step(gaussian_step)
    assert accountant.n_steps == 5
    accountant.reset()
    assert accountant.n_steps == 0


def test_privacy_spent_returns_correct_keys(accountant, gaussian_step):
    """privacy_spent returns a dict whose keys match the supplied alphas list."""
    accountant.add_step(gaussian_step)
    alphas = [2.0, 4.0, 8.0, 16.0]
    result = accountant.privacy_spent(alphas)
    assert set(result.keys()) == set(alphas)


def test_privacy_spent_multiple_values(accountant, gaussian_step):
    """privacy_spent produces distinct epsilon values for different alphas."""
    accountant.add_step(Mechanism(name="tight", sigma=0.5, sensitivity=1.0))
    alphas = [2.0, 4.0, 8.0, 16.0, 32.0]
    result = accountant.privacy_spent(alphas)
    values = list(result.values())
    # Not all values should be identical — different orders give different epsilons
    assert len(set(values)) > 1


def test_mechanism_dataclass_stores_fields():
    """Mechanism dataclass correctly stores name, sigma, and sensitivity."""
    m = Mechanism(name="test_mech", sigma=0.3, sensitivity=2.0)
    assert m.name == "test_mech"
    assert m.sigma == 0.3
    assert m.sensitivity == 2.0


def test_mechanism_default_sensitivity():
    """Mechanism dataclass defaults sensitivity to 1.0."""
    m = Mechanism(name="default_sens", sigma=1.0)
    assert m.sensitivity == 1.0
