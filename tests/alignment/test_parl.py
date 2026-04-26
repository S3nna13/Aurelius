"""Unit tests for PARL reward (Kimi K2.5 §3.3, arXiv:2602.02276)."""

from __future__ import annotations

import pytest
import torch

from src.alignment.parl import AnnealedLambda, PARLReward

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(B: int = 4, val: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (r_perf, r_parallel, r_finish) tensors of shape (B,)."""
    return (
        torch.full((B,), val),
        torch.full((B,), val),
        torch.full((B,), val),
    )


# ---------------------------------------------------------------------------
# Test 1 — output shape equals input shape [B]
# ---------------------------------------------------------------------------


def test_output_shape():
    B = 8
    reward_fn = PARLReward()
    r_perf, r_par, r_fin = _make_batch(B)
    out = reward_fn(r_perf, r_par, r_fin, step=0)
    assert out.shape == (B,), f"Expected shape ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2 — serial_collapse: r_parallel=0 → reward < parallel case (step=0)
# ---------------------------------------------------------------------------


def test_serial_collapse():
    B = 4
    reward_fn = PARLReward(lambda1=1.0, lambda2=1.0)
    r_perf = torch.ones(B)
    r_finish = torch.ones(B)

    # Parallel case: r_parallel=1
    parallel_reward = reward_fn(r_perf, torch.ones(B), r_finish, step=0)
    # Serial case: r_parallel=0
    serial_reward = reward_fn(r_perf, torch.zeros(B), r_finish, step=0)

    assert (serial_reward < parallel_reward).all(), (
        "serial (r_parallel=0) reward should be strictly less than parallel case at step=0"
    )


# ---------------------------------------------------------------------------
# Test 3 — AnnealedLambda: value at step=0 == start; at step=total_steps == 0
# ---------------------------------------------------------------------------


def test_annealed_lambda_bounds():
    total = 5_000
    lam = AnnealedLambda(start=1.0, total_steps=total)
    assert lam(0) == pytest.approx(1.0), "AnnealedLambda(step=0) should equal start"
    assert lam(total) == pytest.approx(0.0), "AnnealedLambda(step=total_steps) should equal 0"


# ---------------------------------------------------------------------------
# Test 4 — perf_only: lambda1=0, lambda2=0 → output == r_perf exactly
# ---------------------------------------------------------------------------


def test_perf_only():
    B = 6
    reward_fn = PARLReward(lambda1=0.0, lambda2=0.0)
    r_perf, r_par, r_fin = _make_batch(B, val=0.7)
    out = reward_fn(r_perf, r_par, r_fin, step=0)
    assert torch.allclose(out, r_perf), "With lambda1=lambda2=0, output must equal r_perf"


# ---------------------------------------------------------------------------
# Test 5 — no NaN / Inf on zero inputs
# ---------------------------------------------------------------------------


def test_no_nan_inf_on_zeros():
    B = 4
    reward_fn = PARLReward()
    zeros = torch.zeros(B)
    out = reward_fn(zeros, zeros, zeros, step=0)
    assert torch.isfinite(out).all(), "Output must be finite for all-zero inputs"


# ---------------------------------------------------------------------------
# Test 6 — determinism: same inputs → same output
# ---------------------------------------------------------------------------


def test_determinism():
    B = 4
    reward_fn = PARLReward(lambda1=0.5, lambda2=0.3, total_steps=8_000)
    r_perf, r_par, r_fin = _make_batch(B, val=0.3)
    out1 = reward_fn(r_perf, r_par, r_fin, step=1_000)
    out2 = reward_fn(r_perf, r_par, r_fin, step=1_000)
    assert torch.equal(out1, out2), "Same inputs must produce identical outputs"


# ---------------------------------------------------------------------------
# Test 7 — batch_size_one: shape (1,)
# ---------------------------------------------------------------------------


def test_batch_size_one():
    reward_fn = PARLReward()
    r_perf, r_par, r_fin = _make_batch(B=1)
    out = reward_fn(r_perf, r_par, r_fin, step=0)
    assert out.shape == (1,), f"Expected shape (1,), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 8 — empty_subagents proxy: r_parallel=0, r_finish=0 → finite output
# ---------------------------------------------------------------------------


def test_empty_subagents_finite():
    B = 4
    reward_fn = PARLReward()
    r_perf = torch.rand(B)
    zeros = torch.zeros(B)
    out = reward_fn(r_perf, zeros, zeros, step=0)
    assert torch.isfinite(out).all(), "Output must be finite when r_parallel=0 and r_finish=0"
    # Should equal r_perf + 0 + 0
    assert torch.allclose(out, r_perf), "r_parallel=r_finish=0 at step=0 → output equals r_perf"


# ---------------------------------------------------------------------------
# Test 9 — step_beyond_total: lambdas clamp to 0 (not negative)
# ---------------------------------------------------------------------------


def test_step_beyond_total_clamped():
    B = 4
    total = 1_000
    reward_fn = PARLReward(lambda1=1.0, lambda2=1.0, total_steps=total)
    r_perf = torch.ones(B)
    r_par = torch.ones(B) * 10.0  # large value — would produce negative if not clamped
    r_fin = torch.ones(B) * 10.0

    out = reward_fn(r_perf, r_par, r_fin, step=total * 10)  # far past total_steps

    # λ₁ and λ₂ must be exactly 0 → output == r_perf
    assert torch.allclose(out, r_perf), (
        "step > total_steps should clamp lambdas to 0, output must equal r_perf"
    )

    # Also verify AnnealedLambda directly
    lam = AnnealedLambda(start=2.0, total_steps=total)
    assert lam(total * 5) == pytest.approx(0.0), "AnnealedLambda past total_steps must return 0"
    assert lam(total * 5) >= 0.0, "AnnealedLambda must never return a negative value"


# ---------------------------------------------------------------------------
# Test 10 — all_ones: known result check
# ---------------------------------------------------------------------------


def test_all_ones_known_result():
    """With r_perf=r_parallel=r_finish=1, step=0, lambda1=lambda2=1:
    result = 1 + 1*1 + 1*1 = 3  for every element.
    """
    B = 5
    reward_fn = PARLReward(lambda1=1.0, lambda2=1.0, total_steps=10_000)
    ones = torch.ones(B)
    out = reward_fn(ones, ones, ones, step=0)
    expected = torch.full((B,), 3.0)
    assert torch.allclose(out, expected), (
        f"all-ones at step=0 should give 3.0 per element, got {out}"
    )


# ---------------------------------------------------------------------------
# Bonus — AnnealedLambda mid-point interpolation
# ---------------------------------------------------------------------------


def test_annealed_lambda_midpoint():
    total = 10_000
    lam = AnnealedLambda(start=1.0, total_steps=total)
    mid = lam(total // 2)
    assert mid == pytest.approx(0.5, abs=1e-6), "Half-way through should give 0.5"
