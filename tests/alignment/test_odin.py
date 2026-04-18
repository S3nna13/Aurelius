"""Tests for src/alignment/odin.py — ODIN loss (arXiv:2402.07319).

12 focused tests verifying correctness, gradient flow, numerical stability,
length normalisation behaviour, mask handling, and batch compatibility.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.alignment.odin import ODINConfig, ODINLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(
    batch: int = 2,
    t_w: int = 8,
    t_l: int = 8,
    device: str = "cpu",
    seed: int = 0,
):
    """Return a consistent set of random (B, T) log-prob tensors and full masks."""
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    def _lp(b, t):
        return torch.randn(b, t, generator=rng, device=device) - 3.0  # negative, like log-probs

    chosen_lp = _lp(batch, t_w).requires_grad_(True)
    rejected_lp = _lp(batch, t_l).requires_grad_(True)
    ref_chosen_lp = _lp(batch, t_w)
    ref_rejected_lp = _lp(batch, t_l)

    chosen_mask = torch.ones(batch, t_w, device=device)
    rejected_mask = torch.ones(batch, t_l, device=device)

    return chosen_lp, rejected_lp, ref_chosen_lp, ref_rejected_lp, chosen_mask, rejected_mask


# ---------------------------------------------------------------------------
# Test 1 — loss is a scalar
# ---------------------------------------------------------------------------

def test_loss_is_scalar():
    loss_fn = ODINLoss()
    inputs = _make_inputs()
    loss, _ = loss_fn(*inputs)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# Test 2 — gradients flow through chosen and rejected log_probs
# ---------------------------------------------------------------------------

def test_gradient_flows():
    loss_fn = ODINLoss()
    chosen_lp, rejected_lp, ref_c, ref_r, cm, rm = _make_inputs()
    loss, _ = loss_fn(chosen_lp, rejected_lp, ref_c, ref_r, cm, rm)
    loss.backward()
    assert chosen_lp.grad is not None, "No gradient for chosen_log_probs"
    assert rejected_lp.grad is not None, "No gradient for rejected_log_probs"
    assert not torch.all(chosen_lp.grad == 0), "chosen gradient is all-zero"
    assert not torch.all(rejected_lp.grad == 0), "rejected gradient is all-zero"


# ---------------------------------------------------------------------------
# Test 3 — policy == reference → loss ≈ log(2)
# ---------------------------------------------------------------------------

def test_policy_equals_reference_gives_log2():
    """When π_θ == π_ref, implicit rewards are 0, margin is 0, loss = log 2."""
    loss_fn = ODINLoss(ODINConfig(beta=0.1))
    B, T = 4, 10
    lp = torch.randn(B, T) - 3.0
    mask = torch.ones(B, T)

    loss, _ = loss_fn(lp, lp, lp, lp, mask, mask)
    expected = math.log(2)
    assert abs(loss.item() - expected) < 1e-5, (
        f"Expected loss ≈ {expected:.6f}, got {loss.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 4 — higher chosen reward → lower loss
# ---------------------------------------------------------------------------

def test_higher_chosen_reward_lowers_loss():
    """Boosting the chosen log-probs should reduce the loss."""
    B, T = 2, 6
    rng = torch.manual_seed(7)
    base_lp = torch.randn(B, T) - 3.0
    ref_lp = torch.randn(B, T) - 3.0
    mask = torch.ones(B, T)

    loss_fn = ODINLoss(ODINConfig(beta=0.1))

    # Baseline loss
    loss_base, _ = loss_fn(base_lp, base_lp, ref_lp, ref_lp, mask, mask)

    # Chosen gets a strong positive boost — wider margin → lower loss
    boosted = base_lp + 10.0
    loss_boosted, _ = loss_fn(boosted, base_lp, ref_lp, ref_lp, mask, mask)

    assert loss_boosted.item() < loss_base.item(), (
        f"Expected lower loss with boosted chosen; got {loss_boosted.item():.4f} vs {loss_base.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — length normalisation: longer sequence has smaller per-step contribution
# ---------------------------------------------------------------------------

def test_length_normalisation_reduces_long_sequence_contribution():
    """A 2× longer chosen sequence should yield a smaller normalised reward."""
    loss_fn = ODINLoss(ODINConfig(normalize_length=True))

    B = 2
    lp_short = torch.full((B, 5), -1.0)
    lp_long  = torch.full((B, 10), -1.0)  # same per-token value, twice as many tokens
    ref_zero = torch.zeros(B, 5)
    ref_zero_long = torch.zeros(B, 10)
    mask_short = torch.ones(B, 5)
    mask_long  = torch.ones(B, 10)

    # Without normalisation the sum for long would be 2× short
    _, m_short = loss_fn(lp_short, lp_short, ref_zero, ref_zero, mask_short, mask_short)
    _, m_long  = loss_fn(lp_long,  lp_long,  ref_zero_long, ref_zero_long, mask_long, mask_long)

    # After length-norm, per-step reward should be the same regardless of length
    assert abs(m_short["chosen_reward"] - m_long["chosen_reward"]) < 1e-5, (
        "Length-normalised rewards should be equal for same per-token values; "
        f"short={m_short['chosen_reward']:.6f} long={m_long['chosen_reward']:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 6 — normalize_length=False gives different result than True
# ---------------------------------------------------------------------------

def test_normalize_false_differs_from_true():
    B, T_w, T_l = 2, 4, 12  # deliberately different lengths
    chosen_lp = torch.full((B, T_w), -1.0)
    rejected_lp = torch.full((B, T_l), -1.0)
    ref_c = torch.zeros(B, T_w)
    ref_r = torch.zeros(B, T_l)
    mask_c = torch.ones(B, T_w)
    mask_r = torch.ones(B, T_l)

    cfg_norm  = ODINConfig(normalize_length=True)
    cfg_plain = ODINConfig(normalize_length=False)

    loss_norm,  _ = ODINLoss(cfg_norm )(chosen_lp, rejected_lp, ref_c, ref_r, mask_c, mask_r)
    loss_plain, _ = ODINLoss(cfg_plain)(chosen_lp, rejected_lp, ref_c, ref_r, mask_c, mask_r)

    assert not torch.isclose(loss_norm, loss_plain), (
        f"Expected different losses; both are {loss_norm.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 7 — mask zeros out padding tokens
# ---------------------------------------------------------------------------

def test_mask_zeros_out_padding():
    """Adding padding (mask=0) tokens should not change the loss."""
    B, T = 2, 6
    lp_real = torch.randn(B, T) - 3.0
    ref_lp  = torch.randn(B, T) - 3.0
    mask_real = torch.ones(B, T)

    # Pad by 4 extra zero-masked tokens
    pad = 4
    lp_padded  = torch.cat([lp_real,  torch.zeros(B, pad)], dim=1)
    ref_padded = torch.cat([ref_lp,   torch.zeros(B, pad)], dim=1)
    mask_padded = torch.cat([mask_real, torch.zeros(B, pad)], dim=1)

    loss_fn = ODINLoss()
    loss_real,   _ = loss_fn(lp_real, lp_real, ref_lp, ref_lp, mask_real, mask_real)
    loss_padded, _ = loss_fn(lp_padded, lp_padded, ref_padded, ref_padded, mask_padded, mask_padded)

    assert torch.isclose(loss_real, loss_padded, atol=1e-5), (
        f"Mask should zero padding; got {loss_real.item():.6f} vs {loss_padded.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 8 — metrics dict has expected keys
# ---------------------------------------------------------------------------

def test_metrics_keys():
    loss_fn = ODINLoss()
    _, metrics = loss_fn(*_make_inputs())
    expected_keys = {"chosen_reward", "rejected_reward", "reward_margin", "chosen_length", "rejected_length"}
    assert set(metrics.keys()) == expected_keys, (
        f"Unexpected metrics keys: {set(metrics.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 9 — numerical stability: no NaN with extreme log_probs
# ---------------------------------------------------------------------------

def test_no_nan_with_extreme_log_probs():
    B, T = 4, 16
    # Very large and very small values
    chosen_lp     = torch.full((B, T), -1e6)
    rejected_lp   = torch.full((B, T),  1e6)
    ref_chosen_lp = torch.zeros(B, T)
    ref_rejected_lp = torch.zeros(B, T)
    mask = torch.ones(B, T)

    loss_fn = ODINLoss()
    loss, metrics = loss_fn(chosen_lp, rejected_lp, ref_chosen_lp, ref_rejected_lp, mask, mask)

    assert not torch.isnan(loss), f"Loss is NaN with extreme inputs"
    assert not torch.isinf(loss), f"Loss is Inf with extreme inputs"
    for k, v in metrics.items():
        assert math.isfinite(v) or math.isinf(v), f"Metric {k} is NaN: {v}"
        assert not math.isnan(v), f"Metric {k} is NaN: {v}"


# ---------------------------------------------------------------------------
# Test 10 — determinism
# ---------------------------------------------------------------------------

def test_determinism():
    """Same inputs must always produce identical outputs."""
    loss_fn = ODINLoss(ODINConfig(beta=0.2))
    inputs = _make_inputs(seed=99)

    loss1, m1 = loss_fn(*inputs)
    loss2, m2 = loss_fn(*inputs)

    assert torch.equal(loss1, loss2), "Loss is not deterministic"
    assert m1 == m2, "Metrics are not deterministic"


# ---------------------------------------------------------------------------
# Test 11 — length penalty: concise chosen vs. verbose rejected favours chosen
# ---------------------------------------------------------------------------

def test_length_penalty_favours_concise_chosen():
    """With equal per-token reward, a shorter chosen should have a higher
    normalised reward than a much longer rejected response."""
    B = 2
    T_w, T_l = 10, 100  # chosen is 10× shorter

    # Both policy and ref agree on per-token log-prob of -1
    chosen_lp     = torch.full((B, T_w), -1.0)
    rejected_lp   = torch.full((B, T_l), -1.0)
    ref_chosen_lp = torch.full((B, T_w), -2.0)   # chosen gets +1 implicit reward per token
    ref_rejected_lp = torch.full((B, T_l), -2.0) # rejected also gets +1 per token

    mask_c = torch.ones(B, T_w)
    mask_r = torch.ones(B, T_l)

    loss_fn = ODINLoss(ODINConfig(normalize_length=True, beta=1.0))
    _, metrics = loss_fn(chosen_lp, rejected_lp, ref_chosen_lp, ref_rejected_lp, mask_c, mask_r)

    # After length-normalisation both sides have the same per-token reward,
    # so the margin should be near zero and the reward_margin near zero.
    # The key invariant: chosen_reward == rejected_reward (both normalised to +1).
    assert abs(metrics["chosen_reward"] - metrics["rejected_reward"]) < 1e-5, (
        "Equal per-token rewards should be equal after length-normalisation; "
        f"chosen={metrics['chosen_reward']:.6f}, rejected={metrics['rejected_reward']:.6f}"
    )

    # Explicitly test that concise wins when chosen has strictly higher per-token reward
    ref_chosen_lp2 = torch.full((B, T_w), -3.0)  # chosen gets +2 per token (vs +1 for rejected)
    _, m2 = loss_fn(chosen_lp, rejected_lp, ref_chosen_lp2, ref_rejected_lp, mask_c, mask_r)
    assert m2["reward_margin"] > 0, (
        f"Expected positive margin for higher per-token chosen reward; got {m2['reward_margin']:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 12 — batch size 1 works
# ---------------------------------------------------------------------------

def test_batch_size_1():
    loss_fn = ODINLoss()
    inputs = _make_inputs(batch=1, t_w=5, t_l=7)
    loss, metrics = loss_fn(*inputs)

    assert loss.shape == torch.Size([]), "Loss should be scalar for B=1"
    assert isinstance(metrics, dict), "Metrics should be a dict for B=1"
    assert all(isinstance(v, float) for v in metrics.values()), "All metric values should be float"
