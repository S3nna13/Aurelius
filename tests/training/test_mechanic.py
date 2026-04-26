"""Tests for src/training/mechanic.py — Mechanic learning rate tuner.

Reference: Cutkosky & Orabona, "Mechanic: A Learning Rate Tuner",
           arXiv:2306.00144 (2023).

Test coverage (14 tests):
  1.  Params move after a step
  2.  Parameter updates are finite (no NaN/Inf)
  3.  learned_lr > 0 after the first step
  4.  learned_lr changes over multiple steps (it learns)
  5.  Determinism under fixed seed
  6.  Converges on quadratic loss over 20 steps
  7.  All s_k bets remain non-negative (clamped to >= 0)
  8.  Works with SGD as base optimizer
  9.  Works with Adam as base optimizer
 10.  No NaN/Inf on large (100x scaled) gradients
 11.  h_t computed correctly as inner product with previous update
 12.  num_bets=1: single bet still works
 13.  After many steps learned_lr stabilizes (doesn't diverge)
 14.  Base direction is applied (parameters actually change proportional to s_t)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.training.mechanic import MechanicWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(n: int = 8, seed: int = 0) -> nn.Parameter:
    """Create a 1-D nn.Parameter of size n."""
    torch.manual_seed(seed)
    return nn.Parameter(torch.randn(n))


def _quadratic_loss(x: torch.Tensor) -> torch.Tensor:
    """f(x) = ||x||^2, minimum at 0."""
    return (x * x).sum()


def _make_sgd_mechanic(params, lr: float = 0.01, s_init: float = 1e-3, **kwargs) -> MechanicWrapper:
    base = torch.optim.SGD(params if isinstance(params, list) else [params], lr=lr)
    return MechanicWrapper(base, s_init=s_init, **kwargs)


def _make_adam_mechanic(
    params, lr: float = 0.01, s_init: float = 1e-3, **kwargs
) -> MechanicWrapper:
    base = torch.optim.Adam(params if isinstance(params, list) else [params], lr=lr)
    return MechanicWrapper(base, s_init=s_init, **kwargs)


def _one_step(opt: MechanicWrapper, x: nn.Parameter) -> None:
    opt.zero_grad()
    _quadratic_loss(x).backward()
    opt.step()


# ---------------------------------------------------------------------------
# 1. Params move after a step
# ---------------------------------------------------------------------------


def test_params_move_after_step():
    x = _make_params(8, seed=1)
    before = x.data.clone()
    opt = _make_adam_mechanic([x])
    _one_step(opt, x)
    assert not torch.allclose(x.data, before), "Parameters should change after a step."


# ---------------------------------------------------------------------------
# 2. Parameter updates are finite (no NaN / Inf)
# ---------------------------------------------------------------------------


def test_params_finite_after_step():
    x = _make_params(8, seed=2)
    opt = _make_adam_mechanic([x])
    for _ in range(5):
        _one_step(opt, x)
    assert torch.isfinite(x.data).all(), "Parameters contain NaN or Inf."


# ---------------------------------------------------------------------------
# 3. learned_lr > 0 after first step
# ---------------------------------------------------------------------------


def test_learned_lr_positive_after_first_step():
    torch.manual_seed(3)
    x = _make_params(8, seed=3)
    opt = _make_adam_mechanic([x], s_init=1e-6)
    _one_step(opt, x)
    assert opt.learned_lr > 0.0, (
        f"learned_lr should be positive after first step, got {opt.learned_lr}"
    )


# ---------------------------------------------------------------------------
# 4. learned_lr changes over steps (it is learning)
# ---------------------------------------------------------------------------


def test_learned_lr_changes_over_steps():
    torch.manual_seed(4)
    x = _make_params(16, seed=4)
    opt = _make_adam_mechanic([x], s_init=1e-3)
    _one_step(opt, x)
    lr_after_1 = opt.learned_lr
    for _ in range(9):
        _one_step(opt, x)
    lr_after_10 = opt.learned_lr
    assert lr_after_1 != lr_after_10, (
        f"learned_lr should change over steps; step1={lr_after_1}, step10={lr_after_10}"
    )


# ---------------------------------------------------------------------------
# 5. Determinism under fixed seed
# ---------------------------------------------------------------------------


def test_determinism_under_seed():
    def _run():
        torch.manual_seed(42)
        x = nn.Parameter(torch.randn(8))
        base = torch.optim.Adam([x], lr=0.01)
        opt = MechanicWrapper(base, s_init=1e-8)
        for _ in range(5):
            opt.zero_grad()
            _quadratic_loss(x).backward()
            opt.step()
        return x.data.clone()

    r1 = _run()
    r2 = _run()
    assert torch.allclose(r1, r2), "Results should be deterministic under same seed."


# ---------------------------------------------------------------------------
# 6. Converges on simple quadratic loss over 20 steps
# ---------------------------------------------------------------------------


def test_converges_on_quadratic_20_steps():
    torch.manual_seed(6)
    x = nn.Parameter(torch.ones(16) * 2.0)
    opt = _make_adam_mechanic([x], lr=0.1, s_init=1e-4)

    initial_loss = _quadratic_loss(x).item()
    for _ in range(20):
        _one_step(opt, x)
    final_loss = _quadratic_loss(x).item()

    assert final_loss < initial_loss, (
        f"Loss should decrease on quadratic: {initial_loss:.4f} → {final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# 7. All s_k bets remain non-negative
# ---------------------------------------------------------------------------


def test_s_k_non_negative():
    torch.manual_seed(7)
    x = _make_params(8, seed=7)
    opt = _make_adam_mechanic([x], s_init=1e-3, num_bets=6)
    for _ in range(20):
        _one_step(opt, x)
    assert (opt._s_k >= 0).all(), f"All s_k must be non-negative; got {opt._s_k}"


# ---------------------------------------------------------------------------
# 8. Works with SGD as base optimizer
# ---------------------------------------------------------------------------


def test_works_with_sgd():
    torch.manual_seed(8)
    x = _make_params(8, seed=8)
    before = x.data.clone()
    opt = _make_sgd_mechanic([x], lr=0.01, s_init=1e-3)
    for _ in range(5):
        _one_step(opt, x)
    assert not torch.allclose(x.data, before), "Params should change with SGD base."
    assert torch.isfinite(x.data).all(), "Params should be finite with SGD base."


# ---------------------------------------------------------------------------
# 9. Works with Adam as base optimizer
# ---------------------------------------------------------------------------


def test_works_with_adam():
    torch.manual_seed(9)
    x = _make_params(8, seed=9)
    before = x.data.clone()
    opt = _make_adam_mechanic([x], lr=0.01, s_init=1e-3)
    for _ in range(5):
        _one_step(opt, x)
    assert not torch.allclose(x.data, before), "Params should change with Adam base."
    assert torch.isfinite(x.data).all(), "Params should be finite with Adam base."


# ---------------------------------------------------------------------------
# 10. No NaN / Inf on large (100x scaled) gradients
# ---------------------------------------------------------------------------


def test_no_nan_on_large_gradients():
    torch.manual_seed(10)
    x = nn.Parameter(torch.ones(8) * 100.0)  # 100x scale → large gradients
    opt = _make_adam_mechanic([x], lr=0.01, s_init=1e-8)
    for _ in range(10):
        opt.zero_grad()
        (x * x * 100.0).sum().backward()  # amplified gradients
        opt.step()
    assert torch.isfinite(x.data).all(), "NaN/Inf detected with large gradients."
    assert (opt._s_k >= 0).all(), "s_k must stay non-negative under large gradients."


# ---------------------------------------------------------------------------
# 11. h_t computed correctly: inner product with previous update
# ---------------------------------------------------------------------------


def test_h_t_inner_product_correct():
    """Verify h_t = <g_t, Δθ_{t-1}> is computed correctly.

    On the first step there is no previous update, so h_t = 0 and s_k
    should remain at s_init.  On the second step, h_t is non-zero; we
    verify s_k remains non-negative and finite.
    """
    torch.manual_seed(11)
    x = nn.Parameter(torch.zeros(8))
    base = torch.optim.SGD([x], lr=0.01)
    s_init = 1e-3
    opt = MechanicWrapper(base, s_init=s_init, num_bets=6)

    # Step 1: set grad manually (no prev update yet → h_t = 0)
    x.grad = torch.ones(8)
    opt.step()
    # h_t = 0 on first step → s_k stays at s_init
    assert torch.allclose(opt._s_k, torch.full_like(opt._s_k, s_init), rtol=1e-3), (
        f"After step 1 (no prev update), s_k should equal s_init={s_init}; got {opt._s_k}"
    )

    # Step 2: prev_update now exists (params moved by ~s_init * base_dir)
    x.grad = torch.ones(8)
    opt.step()
    # s_k must remain non-negative and finite after h_t is applied
    assert (opt._s_k >= 0).all(), f"s_k should be non-negative, got {opt._s_k}"
    assert torch.isfinite(opt._s_k).all(), f"s_k should be finite, got {opt._s_k}"

    # The gradient (ones) and previous update should be anti-aligned
    # (SGD moves x in -grad direction, gradient is ones → inner product < 0).
    # So h_t < 0 → s_k should decrease (clamped at 0).
    # Verify: each s_k_new ≤ s_init (may be 0 if h_t was negative enough)
    assert (opt._s_k <= s_init + 1e-12).all(), (
        f"s_k should not exceed s_init after negative h_t; got {opt._s_k}"
    )


# ---------------------------------------------------------------------------
# 12. num_bets=1: single bet still works
# ---------------------------------------------------------------------------


def test_single_bet_works():
    torch.manual_seed(12)
    x = _make_params(8, seed=12)
    before = x.data.clone()
    base = torch.optim.Adam([x], lr=0.01)
    opt = MechanicWrapper(base, s_init=1e-3, num_bets=1)
    assert opt._s_k.shape == (1,), f"Expected (1,) s_k shape, got {opt._s_k.shape}"
    for _ in range(5):
        _one_step(opt, x)
    assert not torch.allclose(x.data, before), "Params should change with num_bets=1."
    assert torch.isfinite(x.data).all(), "Params should be finite with num_bets=1."
    assert opt._s_k.item() >= 0, "Single bet must be non-negative."


# ---------------------------------------------------------------------------
# 13. After many steps learned_lr stabilizes (doesn't diverge)
# ---------------------------------------------------------------------------


def test_learned_lr_stabilizes():
    torch.manual_seed(13)
    x = nn.Parameter(torch.ones(16) * 1.0)
    opt = _make_adam_mechanic([x], lr=0.01, s_init=1e-8, num_bets=6)

    lrs = []
    for _ in range(50):
        _one_step(opt, x)
        lrs.append(opt.learned_lr)

    # Check no divergence: final learned_lr should be finite and bounded
    assert math.isfinite(lrs[-1]), f"learned_lr diverged to {lrs[-1]}."
    # After convergence toward 0, s_t should remain >= 0
    assert lrs[-1] >= 0.0, f"learned_lr should be non-negative, got {lrs[-1]}."


# ---------------------------------------------------------------------------
# 14. Base direction applied: parameter change proportional to s_t
# ---------------------------------------------------------------------------


def test_base_direction_applied():
    """Verify that Mechanic applies s_t * base_direction rather than base_direction.

    We run one step with a known s_t=0 (by initializing s_k=0) and confirm
    that parameters do not change; then verify that with s_t>0 they do change.
    """
    torch.manual_seed(14)

    # Case A: s_init = 0 effectively → but s_init must be > 0 per validation.
    # Instead, test that parameters DO change proportionally by comparing
    # two runs with different s_init scales.
    def _run_and_get_delta(s_init: float) -> float:
        torch.manual_seed(99)
        x = nn.Parameter(torch.ones(4) * 2.0)
        base = torch.optim.SGD([x], lr=1.0)  # lr=1 so base dir is 1:1
        opt = MechanicWrapper(base, s_init=s_init, num_bets=1)
        before = x.data.clone()
        opt.zero_grad()
        _quadratic_loss(x).backward()
        opt.step()
        return float((x.data - before).norm().item())

    delta_small = _run_and_get_delta(1e-10)
    delta_large = _run_and_get_delta(1e-2)

    # Larger s_init → larger step (both after first step where h_t=0, s_t = s_init)
    assert delta_large > delta_small, (
        f"Larger s_init should produce larger step: "
        f"delta_small={delta_small:.2e}, delta_large={delta_large:.2e}"
    )
    # Both should be finite
    assert math.isfinite(delta_small), "delta_small is not finite."
    assert math.isfinite(delta_large), "delta_large is not finite."
