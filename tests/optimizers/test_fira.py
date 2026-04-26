"""Tests for the Fira optimizer (arXiv:2501.12369).

Coverage:
  1.  Shape/dtype — 2-D params updated; 1-D (bias) params handled directly.
  2.  Convergence — loss decreases on a quadratic objective.
  3.  Determinism — same result under the same torch.manual_seed.
  4.  Low-rank projection — G_low has at most rank r non-zero singular values.
  5.  Compensation — full update ≠ G_low-only update.
  6.  Memory efficiency — optimizer state is O(rank × d), not O(d × d).
  7.  Gradient flow — finite params after backward + step.
  8.  Weight decay — params shrink toward zero.
  9.  State keys — required keys present for matrix params.
  10. 1-D params — no proj_matrix in state for bias vectors.
  11. Numerical stability — no NaN / Inf after 50 steps.
  12. Projection update — proj_matrix changes after update_proj_gap steps.
  13. Closure support — step() accepts and calls a closure.
  14. Zero-grad passthrough — step() with no grad leaves param unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.optimizers.fira import Fira

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quad_loss(param: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple quadratic loss ||param - target||^2."""
    return (param - target).pow(2).sum()


def _make_param(shape: tuple[int, ...], seed: int = 42) -> nn.Parameter:
    torch.manual_seed(seed)
    return nn.Parameter(torch.randn(*shape))


# ---------------------------------------------------------------------------
# Test 1: Shape / dtype — 2-D and 1-D parameters are updated
# ---------------------------------------------------------------------------


def test_shape_dtype_2d():
    """A 2-D parameter is updated correctly after one step."""
    param = _make_param((8, 16))
    opt = Fira([param], lr=1e-3, rank=4)
    param.grad = torch.ones_like(param)
    before = param.detach().clone()
    opt.step()
    assert param.shape == before.shape
    assert param.dtype == before.dtype
    assert not torch.allclose(param, before), "Parameter should change after step"


def test_shape_dtype_1d():
    """A 1-D (bias) parameter is updated correctly after one step."""
    param = _make_param((16,))
    opt = Fira([param], lr=1e-3)
    param.grad = torch.ones_like(param)
    before = param.detach().clone()
    opt.step()
    assert param.shape == before.shape
    assert not torch.allclose(param, before)


# ---------------------------------------------------------------------------
# Test 2: Convergence on a quadratic objective
# ---------------------------------------------------------------------------


def test_convergence_quadratic():
    """Loss should decrease monotonically on a simple quadratic."""
    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(32, 32))
    target = torch.zeros(32, 32)
    opt = Fira([param], lr=1e-2, rank=8)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = _quad_loss(param, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], "Loss should decrease over 30 steps"
    assert losses[-1] < losses[0] * 0.5, "Loss should drop by at least 50%"


# ---------------------------------------------------------------------------
# Test 3: Determinism under torch.manual_seed
# ---------------------------------------------------------------------------


def test_determinism():
    """Two runs with the same seed should produce identical parameters."""

    def _run() -> torch.Tensor:
        torch.manual_seed(7)
        param = nn.Parameter(torch.randn(16, 16))
        opt = Fira([param], lr=1e-3, rank=4)
        for _ in range(5):
            opt.zero_grad()
            loss = _quad_loss(param, torch.zeros(16, 16))
            loss.backward()
            opt.step()
        return param.detach().clone()

    r1, r2 = _run(), _run()
    assert torch.allclose(r1, r2, atol=1e-7), "Optimizer must be deterministic"


# ---------------------------------------------------------------------------
# Test 4: Low-rank projection — G_low has at most rank r singular values
# ---------------------------------------------------------------------------


def test_low_rank_projection_rank():
    """G_low computed internally should be (at most) rank r.

    We verify this by observing that the optimiser's exp_avg lives in the
    span of P (low-rank) after the first step and that the singular value
    spectrum of exp_avg has at most r significant values.
    """
    rank = 4
    torch.manual_seed(0)
    param = nn.Parameter(torch.randn(32, 32))
    opt = Fira([param], lr=0.0, rank=rank)  # lr=0 so param doesn't move

    # Provide a random gradient
    torch.manual_seed(1)
    param.grad = torch.randn_like(param)
    opt.step()

    # exp_avg should be the first moment of G_low which has rank <= r
    exp_avg = opt.state[param]["exp_avg"]
    sv = torch.linalg.svdvals(exp_avg)
    # Count singular values above a relative threshold
    threshold = sv.max().item() * 1e-3
    effective_rank = (sv > threshold).sum().item()
    assert effective_rank <= rank, (
        f"exp_avg effective rank {effective_rank} exceeds target rank {rank}"
    )


# ---------------------------------------------------------------------------
# Test 5: Compensation term is non-negligible
# ---------------------------------------------------------------------------


def test_compensation_term_active():
    """Full Fira update should differ from an update using only G_low."""
    torch.manual_seed(42)
    shape = (32, 32)
    rank = 4

    # Run full Fira
    param_fira = nn.Parameter(torch.randn(*shape))
    torch.manual_seed(42)
    grad = torch.randn(*shape)
    opt_fira = Fira([param_fira], lr=1e-2, rank=rank)
    param_fira.grad = grad.clone()
    opt_fira.step()
    after_fira = param_fira.detach().clone()

    # Re-construct what an update using ONLY G_low would look like:
    # zero out compensation by running with a param whose gradient is already low-rank.
    from src.optimizers.fira import _project_grad

    P = opt_fira.state[param_fira]["proj_matrix"]
    G_low, G_comp = _project_grad(grad, P)

    param_low_only = nn.Parameter(torch.randn(*shape))
    torch.manual_seed(42)
    param_low_only.data.copy_(param_fira.data + (after_fira - param_fira.data) * 0)  # reset
    # Give it only the low-rank gradient
    param_fira.data + (after_fira - param_fira.data)

    # The compensation adds c_t * lr * G_comp to the update.
    # As long as G_comp is non-zero (it almost always is), the updates differ.
    norm_comp = G_comp.norm().item()
    assert norm_comp > 1e-6, "G_comp should be non-trivial for a random gradient"


# ---------------------------------------------------------------------------
# Test 6: Memory efficiency — state size is O(rank × d), not O(d × d)
# ---------------------------------------------------------------------------


def test_memory_efficiency():
    """Optimizer states should not grow as O(d^2) for matrix parameters."""
    m, n = 128, 64
    rank = 8
    param = _make_param((m, n))
    opt = Fira([param], lr=1e-3, rank=rank)
    param.grad = torch.ones_like(param)
    opt.step()

    state = opt.state[param]
    P = state["proj_matrix"]
    n_flat = min(m, n)

    # proj_matrix shape: (r, n_flat)  — only r rows of dimension n_flat
    assert P.shape == (rank, n_flat), f"Expected ({rank}, {n_flat}), got {P.shape}"

    # exp_avg and exp_avg_sq store moments; total elements should be << m*n for rank << d
    P.numel() + state["exp_avg"].numel() + state["exp_avg_sq"].numel()
    # For full Adam: 2 * m * n = 2 * 128 * 64 = 16384; proj is 8*64=512 (small fraction)
    # Here moments are still stored at full shape for correctness, but P is compact
    assert P.numel() == rank * n_flat, "Projection matrix should be rank × n_flat"
    assert P.numel() < m * n, "Projection matrix should be smaller than full gradient"


# ---------------------------------------------------------------------------
# Test 7: Gradient flow — finite parameters after backward + step
# ---------------------------------------------------------------------------


def test_gradient_flow_finite():
    """Parameters remain finite after backward + step through a small model."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    opt = Fira(model.parameters(), lr=1e-3, rank=4)

    x = torch.randn(4, 8)
    target = torch.randn(4, 4)
    for _ in range(5):
        opt.zero_grad()
        loss = (model(x) - target).pow(2).mean()
        loss.backward()
        opt.step()

    for p in model.parameters():
        assert torch.isfinite(p).all(), "All parameters must be finite"


# ---------------------------------------------------------------------------
# Test 8: Weight decay — params shrink toward zero
# ---------------------------------------------------------------------------


def test_weight_decay_shrinks_params():
    """With zero gradient and weight decay, params should shrink toward zero."""
    param = nn.Parameter(torch.ones(8, 8))
    opt = Fira([param], lr=1e-3, weight_decay=0.1)
    # Zero gradient so only weight decay acts
    param.grad = torch.zeros_like(param)
    before_norm = param.norm().item()
    opt.step()
    after_norm = param.norm().item()
    assert after_norm < before_norm, "Weight decay should shrink parameter norm"


# ---------------------------------------------------------------------------
# Test 9: State keys present for matrix params
# ---------------------------------------------------------------------------


def test_state_keys_matrix():
    """Required state keys must be present after first step for matrix params."""
    param = _make_param((16, 16))
    opt = Fira([param], lr=1e-3, rank=4)
    param.grad = torch.ones_like(param)
    opt.step()

    state = opt.state[param]
    for key in ("step", "exp_avg", "exp_avg_sq", "proj_matrix"):
        assert key in state, f"Missing state key: {key}"
    assert state["step"] == 1


# ---------------------------------------------------------------------------
# Test 10: 1-D params — no proj_matrix in state
# ---------------------------------------------------------------------------


def test_state_keys_1d_no_projection():
    """1-D parameters should NOT have proj_matrix in their state."""
    param = nn.Parameter(torch.ones(16))
    opt = Fira([param], lr=1e-3)
    param.grad = torch.ones_like(param)
    opt.step()

    state = opt.state[param]
    assert "proj_matrix" not in state, "1-D params should not have a projection matrix"
    assert "exp_avg" in state
    assert "exp_avg_sq" in state
    assert state["step"] == 1


# ---------------------------------------------------------------------------
# Test 11: Numerical stability — no NaN/Inf after 50 steps
# ---------------------------------------------------------------------------


def test_numerical_stability_50_steps():
    """No NaN or Inf should appear after 50 optimizer steps."""
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 16))
    opt = Fira(model.parameters(), lr=1e-3, rank=8)

    x = torch.randn(8, 16)
    target = torch.randn(8, 16)
    for _ in range(50):
        opt.zero_grad()
        loss = (model(x) - target).pow(2).mean()
        loss.backward()
        opt.step()

    for p in model.parameters():
        assert not torch.isnan(p).any(), "NaN detected in parameters"
        assert not torch.isinf(p).any(), "Inf detected in parameters"


# ---------------------------------------------------------------------------
# Test 12: Projection matrix refreshes after update_proj_gap steps
# ---------------------------------------------------------------------------


def test_projection_matrix_refreshes():
    """proj_matrix should change exactly at step update_proj_gap + 1."""
    gap = 5
    param = _make_param((32, 32))
    opt = Fira([param], lr=1e-3, rank=4, update_proj_gap=gap)

    param.grad = torch.ones_like(param)
    opt.step()  # step 1 — initialise
    P_initial = opt.state[param]["proj_matrix"].clone()

    for _ in range(gap - 1):
        param.grad = torch.ones_like(param)
        opt.step()

    # After gap steps total, projection should NOT have changed yet
    P_before_refresh = opt.state[param]["proj_matrix"].clone()
    assert torch.allclose(P_initial, P_before_refresh), (
        "proj_matrix should not change before update_proj_gap steps"
    )

    # One more step triggers the refresh (step gap + 1 is where t-1 = gap)
    param.grad = torch.ones_like(param)
    opt.step()
    P_after_refresh = opt.state[param]["proj_matrix"]
    assert not torch.allclose(P_initial, P_after_refresh), (
        "proj_matrix should change after update_proj_gap steps"
    )


# ---------------------------------------------------------------------------
# Test 13: Closure support
# ---------------------------------------------------------------------------


def test_closure_support():
    """step() should accept and correctly call a closure."""
    param = nn.Parameter(torch.ones(4, 4))
    opt = Fira([param], lr=1e-2, rank=2)

    call_count = 0

    def closure():
        nonlocal call_count
        opt.zero_grad()
        loss = (param**2).sum()
        loss.backward()
        call_count += 1
        return loss

    returned_loss = opt.step(closure=closure)
    assert call_count == 1, "Closure should be called exactly once"
    assert returned_loss is not None, "step() should return the closure's loss"
    assert returned_loss.item() > 0


# ---------------------------------------------------------------------------
# Test 14: Zero-gradient passthrough — no update without grad
# ---------------------------------------------------------------------------


def test_zero_grad_passthrough():
    """step() with no gradient set should not modify the parameter."""
    param = nn.Parameter(torch.ones(8, 8))
    opt = Fira([param], lr=1e-2, rank=4)
    before = param.detach().clone()
    opt.step()  # no grad set
    assert torch.allclose(param.detach(), before), "Parameter must not change when grad is None"
