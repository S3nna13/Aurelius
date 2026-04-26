"""
Tests for src/alignment/double_sided_is.py — 10 tests.
Pure PyTorch, tiny tensors.
"""

import pytest
import torch

from src.alignment.double_sided_is import DDISConfig, DoubleSidedISLoss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
N = 32  # number of tokens


def make_loss(**kwargs) -> DoubleSidedISLoss:
    return DoubleSidedISLoss(DDISConfig(**kwargs))


def random_log_probs(n: int = N) -> torch.Tensor:
    return torch.randn(n)


def random_advantages(n: int = N) -> torch.Tensor:
    return torch.randn(n)


# ---------------------------------------------------------------------------
# 1. DDISConfig instantiates with defaults
# ---------------------------------------------------------------------------


def test_ddisconfig_defaults():
    cfg = DDISConfig()
    assert cfg.eps_low == pytest.approx(0.2)
    assert cfg.eps_high == pytest.approx(0.2)
    assert cfg.clip_outside is True


# ---------------------------------------------------------------------------
# 2. DoubleSidedISLoss instantiates
# ---------------------------------------------------------------------------


def test_double_sided_is_loss_instantiates():
    loss_fn = make_loss()
    assert isinstance(loss_fn, DoubleSidedISLoss)


# ---------------------------------------------------------------------------
# 3. Forward returns scalar tensor
# ---------------------------------------------------------------------------


def test_forward_returns_scalar():
    loss_fn = make_loss()
    log_theta = random_log_probs()
    log_old = random_log_probs()
    adv = random_advantages()
    loss = loss_fn(log_theta, log_old, adv)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 4. Loss is finite
# ---------------------------------------------------------------------------


def test_forward_loss_is_finite():
    loss_fn = make_loss()
    log_theta = random_log_probs()
    log_old = random_log_probs()
    adv = random_advantages()
    loss = loss_fn(log_theta, log_old, adv)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# 5. Identical log probs -> ratio = 1.0 (via ratio_stats mean_ratio ~ 1.0)
# ---------------------------------------------------------------------------


def test_identical_log_probs_ratio_is_one():
    loss_fn = make_loss()
    log_probs = random_log_probs()
    stats = loss_fn.ratio_stats(log_probs, log_probs)
    assert stats["mean_ratio"] == pytest.approx(1.0, abs=1e-5), (
        f"Expected mean_ratio ~ 1.0 for identical probs, got {stats['mean_ratio']}"
    )


# ---------------------------------------------------------------------------
# 6. ratio_stats returns dict with all 3 keys
# ---------------------------------------------------------------------------


def test_ratio_stats_returns_all_keys():
    loss_fn = make_loss()
    stats = loss_fn.ratio_stats(random_log_probs(), random_log_probs())
    assert "mean_ratio" in stats
    assert "frac_clipped_low" in stats
    assert "frac_clipped_high" in stats


# ---------------------------------------------------------------------------
# 7. frac_clipped_low in [0, 1]
# ---------------------------------------------------------------------------


def test_frac_clipped_low_in_range():
    loss_fn = make_loss()
    stats = loss_fn.ratio_stats(random_log_probs(), random_log_probs())
    assert 0.0 <= stats["frac_clipped_low"] <= 1.0


# ---------------------------------------------------------------------------
# 8. frac_clipped_high in [0, 1]
# ---------------------------------------------------------------------------


def test_frac_clipped_high_in_range():
    loss_fn = make_loss()
    stats = loss_fn.ratio_stats(random_log_probs(), random_log_probs())
    assert 0.0 <= stats["frac_clipped_high"] <= 1.0


# ---------------------------------------------------------------------------
# 9. Gradient flows through loss
# ---------------------------------------------------------------------------


def test_gradient_flows_through_loss():
    loss_fn = make_loss()
    log_theta = random_log_probs().requires_grad_(True)
    log_old = random_log_probs().detach()
    adv = random_advantages().detach()
    loss = loss_fn(log_theta, log_old, adv)
    loss.backward()
    assert log_theta.grad is not None, "log_theta should have gradient after backward"
    assert torch.isfinite(log_theta.grad).all(), "Gradient must be finite"


# ---------------------------------------------------------------------------
# 10. Loss changes when advantages change
# ---------------------------------------------------------------------------


def test_loss_changes_with_different_advantages():
    loss_fn = make_loss()
    log_theta = random_log_probs()
    log_old = random_log_probs()
    adv1 = torch.ones(N)
    adv2 = -torch.ones(N)
    loss1 = loss_fn(log_theta, log_old, adv1).item()
    loss2 = loss_fn(log_theta, log_old, adv2).item()
    assert loss1 != loss2, (
        f"Loss should differ for opposite advantages, got loss1={loss1}, loss2={loss2}"
    )
