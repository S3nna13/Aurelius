"""Tests for TokenCreditAssigner — token credit assignment from response-level reward.

Covers:
  1.  test_config_defaults
  2.  test_uniform_shape
  3.  test_uniform_sum
  4.  test_uniform_masked
  5.  test_discounted_shape
  6.  test_discounted_monotone
  7.  test_end_decay_last_token_highest
  8.  test_end_decay_masked
  9.  test_gae_shape
  10. test_gae_no_value
  11. test_gae_backward_scan
  12. test_assign_uniform
  13. test_assign_gae
  14. test_normalize
  15. test_statistics_keys
  16. test_statistics_values

  Integration: test_integration_all_methods
"""

from __future__ import annotations

import torch

from src.training.token_credit_assignment import TokenCreditAssigner, TokenCreditConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mask(B: int, T: int, lengths=None) -> torch.Tensor:
    """Build a [B, T] mask from per-row lengths (default: all valid)."""
    if lengths is None:
        lengths = [T] * B
    mask = torch.zeros(B, T)
    for i, lvl in enumerate(lengths):
        mask[i, :lvl] = 1.0
    return mask


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = TokenCreditConfig()
    assert cfg.method == "gae"
    assert cfg.gamma == 0.99
    assert cfg.lam == 0.95
    assert cfg.end_decay == 0.9
    assert cfg.normalize is True
    assert cfg.eps == 1e-8


# ---------------------------------------------------------------------------
# 2. Uniform — shape
# ---------------------------------------------------------------------------


def test_uniform_shape():
    cfg = TokenCreditConfig(method="uniform", normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 3, 10
    rewards = torch.ones(B)
    mask = make_mask(B, T)
    out = assigner.uniform(rewards, mask)
    assert out.shape == (B, T)


# ---------------------------------------------------------------------------
# 3. Uniform — sum ≈ reward per sequence
# ---------------------------------------------------------------------------


def test_uniform_sum():
    cfg = TokenCreditConfig(method="uniform", normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 4, 8
    rewards = torch.tensor([1.0, 2.0, -1.0, 0.5])
    mask = make_mask(B, T)
    out = assigner.uniform(rewards, mask)
    sums = out.sum(dim=1)
    assert torch.allclose(sums, rewards, atol=1e-5), f"sums={sums}, rewards={rewards}"


# ---------------------------------------------------------------------------
# 4. Uniform — masked positions are zero
# ---------------------------------------------------------------------------


def test_uniform_masked():
    cfg = TokenCreditConfig(method="uniform", normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 2, 6
    rewards = torch.ones(B)
    lengths = [3, 5]
    mask = make_mask(B, T, lengths)
    out = assigner.uniform(rewards, mask)
    for i, lvl in enumerate(lengths):
        assert torch.all(out[i, lvl:] == 0.0), f"row {i}: expected zeros after position {lvl}"


# ---------------------------------------------------------------------------
# 5. Discounted — shape
# ---------------------------------------------------------------------------


def test_discounted_shape():
    cfg = TokenCreditConfig(method="discounted", gamma=0.99, normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 5, 12
    rewards = torch.ones(B)
    mask = make_mask(B, T)
    out = assigner.discounted(rewards, mask)
    assert out.shape == (B, T)


# ---------------------------------------------------------------------------
# 6. Discounted — earlier tokens get strictly less credit (gamma < 1)
# ---------------------------------------------------------------------------


def test_discounted_monotone():
    cfg = TokenCreditConfig(method="discounted", gamma=0.9, normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 1, 8
    rewards = torch.ones(B)  # positive reward → all credits positive
    mask = make_mask(B, T)
    out = assigner.discounted(rewards, mask)  # [1, 8]
    credits = out[0]
    # credits should be strictly increasing (earlier = less)
    for t in range(T - 1):
        assert credits[t] < credits[t + 1], (
            f"credit[{t}]={credits[t]:.6f} not < credit[{t + 1}]={credits[t + 1]:.6f}"
        )


# ---------------------------------------------------------------------------
# 7. End-decay — last valid token gets the highest credit
# ---------------------------------------------------------------------------


def test_end_decay_last_token_highest():
    cfg = TokenCreditConfig(method="end_decay", end_decay=0.8, normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 2, 7
    rewards = torch.ones(B)
    lengths = [5, 7]
    mask = make_mask(B, T, lengths)
    out = assigner.end_decay(rewards, mask)
    for i, lvl in enumerate(lengths):
        last_credit = out[i, lvl - 1].item()
        for t in range(lvl - 1):
            assert out[i, t].item() <= last_credit + 1e-6, (
                f"row {i}, token {t}: credit {out[i, t].item():.6f} > last {last_credit:.6f}"
            )


# ---------------------------------------------------------------------------
# 8. End-decay — masked positions are zero
# ---------------------------------------------------------------------------


def test_end_decay_masked():
    cfg = TokenCreditConfig(method="end_decay", end_decay=0.9, normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 3, 10
    rewards = torch.ones(B)
    lengths = [4, 7, 10]
    mask = make_mask(B, T, lengths)
    out = assigner.end_decay(rewards, mask)
    for i, lvl in enumerate(lengths):
        if lvl < T:
            assert torch.all(out[i, lvl:] == 0.0)


# ---------------------------------------------------------------------------
# 9. GAE — shape
# ---------------------------------------------------------------------------


def test_gae_shape():
    cfg = TokenCreditConfig(method="gae", gamma=0.99, lam=0.95, normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 3, 10
    rewards = torch.ones(B)
    values = torch.zeros(B, T)
    mask = make_mask(B, T)
    out = assigner.gae(rewards, values, mask)
    assert out.shape == (B, T)


# ---------------------------------------------------------------------------
# 10. GAE with zero values reduces to discounted return
#     A_t = Σ_{k≥0} γ^k δ_{t+k}  with δ_k = r_k  (V≡0)
#     = γ^(T-1-t) * reward  for the last token   (discounted)
# ---------------------------------------------------------------------------


def test_gae_no_value():
    gamma = 0.9
    cfg = TokenCreditConfig(method="gae", gamma=gamma, lam=1.0, normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 1, 5
    reward_val = 2.0
    rewards = torch.tensor([reward_val])
    values = torch.zeros(B, T)
    mask = make_mask(B, T)
    out = assigner.gae(rewards, values, mask)  # [1, 5]
    # With V=0 and lam=1, A_t = Σ γ^k r_{t+k}
    # Only r_{T-1} = reward_val is nonzero.
    # A_t = γ^(T-1-t) * reward_val
    for t in range(T):
        expected = (gamma ** (T - 1 - t)) * reward_val
        got = out[0, t].item()
        assert abs(got - expected) < 1e-4, f"t={t}: expected {expected:.6f}, got {got:.6f}"


# ---------------------------------------------------------------------------
# 11. GAE backward scan — advantages should generally decrease for earlier
#     tokens with a positive reward and reasonable gamma/lam
# ---------------------------------------------------------------------------


def test_gae_backward_scan():
    cfg = TokenCreditConfig(method="gae", gamma=0.99, lam=0.95, normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 1, 6
    rewards = torch.tensor([1.0])
    values = torch.zeros(B, T)
    mask = make_mask(B, T)
    out = assigner.gae(rewards, values, mask)
    credits = out[0]  # [T]
    # For positive reward, last token should have higher advantage than first
    assert credits[-1].item() > credits[0].item(), (
        f"Expected credits[-1] > credits[0], got {credits[-1]:.6f} vs {credits[0]:.6f}"
    )


# ---------------------------------------------------------------------------
# 12. assign — dispatches to uniform correctly
# ---------------------------------------------------------------------------


def test_assign_uniform():
    cfg = TokenCreditConfig(method="uniform", normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 2, 5
    rewards = torch.tensor([4.0, 8.0])
    mask = make_mask(B, T)
    out = assigner.assign(rewards, mask)
    assert out.shape == (B, T)
    # Each token should be reward / T
    for i, r in enumerate([4.0, 8.0]):
        expected = r / T
        assert torch.allclose(out[i, :T], torch.full((T,), expected), atol=1e-5)


# ---------------------------------------------------------------------------
# 13. assign — dispatches to gae correctly with values
# ---------------------------------------------------------------------------


def test_assign_gae():
    cfg = TokenCreditConfig(method="gae", gamma=0.99, lam=0.95, normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 2, 8
    rewards = torch.ones(B)
    values = torch.rand(B, T)
    mask = make_mask(B, T)
    out = assigner.assign(rewards, mask, values=values)
    assert out.shape == (B, T)
    assert not torch.any(torch.isnan(out))


# ---------------------------------------------------------------------------
# 14. Normalize — output has approximately zero mean and unit std
# ---------------------------------------------------------------------------


def test_normalize():
    cfg = TokenCreditConfig(method="uniform", normalize=True)
    assigner = TokenCreditAssigner(cfg)
    B, T = 8, 20
    rewards = torch.randn(B)
    mask = make_mask(B, T)
    out = assigner.assign(rewards, mask)
    # Gather valid values
    valid = out[mask.bool()]
    assert abs(valid.mean().item()) < 0.1, f"mean={valid.mean().item()}"
    # std should be close to 1 (within noise for finite samples)
    assert abs(valid.std().item() - 1.0) < 0.1, f"std={valid.std().item()}"


# ---------------------------------------------------------------------------
# 15. statistics — correct keys are returned
# ---------------------------------------------------------------------------


def test_statistics_keys():
    cfg = TokenCreditConfig(method="uniform", normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 2, 5
    credits = torch.ones(B, T)
    mask = make_mask(B, T)
    stats = assigner.statistics(credits, mask)
    assert set(stats.keys()) == {"mean", "std", "min", "max"}


# ---------------------------------------------------------------------------
# 16. statistics — values are correct for known input
# ---------------------------------------------------------------------------


def test_statistics_values():
    cfg = TokenCreditConfig(method="uniform", normalize=False)
    assigner = TokenCreditAssigner(cfg)
    B, T = 1, 4
    credits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.ones(B, T)
    stats = assigner.statistics(credits, mask)
    assert abs(stats["mean"] - 2.5) < 1e-5
    assert abs(stats["min"] - 1.0) < 1e-5
    assert abs(stats["max"] - 4.0) < 1e-5


# ---------------------------------------------------------------------------
# Integration: B=4, T=16, all 4 methods — shapes and no NaN
# ---------------------------------------------------------------------------


def test_integration_all_methods():
    B, T = 4, 16
    rewards = torch.tensor([1.0, -0.5, 0.0, 2.0])
    lengths = [16, 12, 8, 10]
    mask = make_mask(B, T, lengths)
    values = torch.rand(B, T)

    methods_and_kwargs = [
        ("uniform", {}),
        ("discounted", {}),
        ("end_decay", {}),
        ("gae", {"values": values}),
    ]

    for method, extra_kwargs in methods_and_kwargs:
        cfg = TokenCreditConfig(method=method, normalize=True)
        assigner = TokenCreditAssigner(cfg)
        out = assigner.assign(rewards, mask, **extra_kwargs)

        assert out.shape == (B, T), f"{method}: shape mismatch {out.shape}"
        assert not torch.any(torch.isnan(out)), f"{method}: NaN in output"
        assert not torch.any(torch.isinf(out)), f"{method}: Inf in output"

        # Masked-out positions must remain zero after normalization
        for i, lvl in enumerate(lengths):
            if lvl < T:
                assert torch.all(out[i, lvl:] == 0.0), (
                    f"{method}: row {i} has non-zero credits beyond mask length {lvl}"
                )

        # Statistics should run without error
        stats = assigner.statistics(out, mask)
        assert all(k in stats for k in ("mean", "std", "min", "max"))
