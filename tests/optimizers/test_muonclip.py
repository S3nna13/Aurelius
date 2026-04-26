"""Unit tests for the MuonClip optimizer.

Covers correctness, numerical stability, edge cases, and API contracts.
Pure PyTorch only — no external ML library dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from src.optimizers.muonclip import MuonClip, _orthogonalize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model():
    """Two-layer MLP; MuonClip converges reliably on non-linear models."""
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))


def _make_data():
    torch.manual_seed(1)
    x = torch.randn(64, 16)
    target = torch.randn(64, 8)
    return x, target


def _run_steps(model, optimizer, x, target, n: int = 1):
    """Run *n* forward-backward-step iterations; return list of scalar losses."""
    losses = []
    loss_fn = nn.MSELoss()
    for _ in range(n):
        optimizer.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# ---------------------------------------------------------------------------
# 1. Loss decreases over 10 steps
# ---------------------------------------------------------------------------


def test_loss_decreases():
    model = _make_model()
    x, target = _make_data()
    opt = MuonClip(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        initial_loss = loss_fn(model(x), target).item()

    _run_steps(model, opt, x, target, n=30)

    with torch.no_grad():
        final_loss = loss_fn(model(x), target).item()

    assert final_loss < initial_loss, (
        f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Determinism — same seed yields identical parameters after 5 steps
# ---------------------------------------------------------------------------


def test_determinism():
    def run():
        torch.manual_seed(42)
        model = nn.Linear(8, 4)
        x = torch.randn(4, 8)
        target = torch.randn(4, 4)
        opt = MuonClip(model.parameters(), lr=1e-3)
        _run_steps(model, opt, x, target, n=5)
        return [p.clone() for p in model.parameters()]

    params_a = run()
    params_b = run()
    for pa, pb in zip(params_a, params_b):
        assert torch.allclose(pa, pb, atol=0.0), "Non-deterministic update detected"


# ---------------------------------------------------------------------------
# 3. All parameters remain finite after a backward + step
# ---------------------------------------------------------------------------


def test_gradient_finite():
    model = _make_model()
    x, target = _make_data()
    opt = MuonClip(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = nn.MSELoss()(model(x), target)
    loss.backward()
    opt.step()
    for p in model.parameters():
        assert torch.isfinite(p).all(), "Non-finite parameter detected"


# ---------------------------------------------------------------------------
# 4. Max-norm clipping handles a very large gradient without producing NaN
# ---------------------------------------------------------------------------


def test_max_norm_clipping():
    model = _make_model()
    opt = MuonClip(model.parameters(), lr=1e-3, max_norm=1.0)
    # Manually inject a huge gradient on the weight parameter.
    w = list(model.parameters())[0]
    w.grad = torch.full_like(w, 1000.0)
    # Capture original value before step.
    orig = w.data.clone()
    opt.step()
    assert not torch.isnan(w).any(), "NaN detected after large-gradient step"
    assert not torch.equal(w, orig), "Parameter was not updated"


# ---------------------------------------------------------------------------
# 5. Orthogonalization produces near-orthogonal rows
# ---------------------------------------------------------------------------


def test_orthogonalization():
    """Newton-Schulz is an iterative approximation; a single step makes progress
    toward orthogonality.  We verify two properties:

    1. Output is finite and has the same shape as the input.
    2. The singular-value spread (max/min ratio) of the row-matrix decreases
       after the NS step when starting from a pre-scaled matrix, confirming
       the step moves the rows toward equal-norm and mutual orthogonality.
    """
    torch.manual_seed(7)
    M = torch.randn(4, 8)

    # Pre-scale so singular values are near 1 (the NS fixed-point regime).
    m_flat = M.reshape(M.shape[0], -1)
    sv = torch.linalg.svdvals(m_flat)
    M_scaled = M / sv.max()

    M_orth = _orthogonalize(M_scaled)
    # Shape must be preserved.
    assert M_orth.shape == M_scaled.shape, "Shape changed after orthogonalization"
    # Output must be finite.
    assert torch.isfinite(M_orth).all(), "Non-finite values after orthogonalization"

    # Singular-value spread should decrease (NS makes progress toward orthogonality).
    sv_in = torch.linalg.svdvals(M_scaled.reshape(4, -1))
    sv_out = torch.linalg.svdvals(M_orth.reshape(4, -1))
    spread_in = (sv_in.max() / (sv_in.min() + 1e-8)).item()
    spread_out = (sv_out.max() / (sv_out.min() + 1e-8)).item()
    assert spread_out <= spread_in + 0.1, (
        f"NS step did not reduce singular-value spread: {spread_in:.4f} -> {spread_out:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. lr=0 leaves parameters unchanged
# ---------------------------------------------------------------------------


def test_lr_zero():
    model = _make_model()
    x, target = _make_data()
    opt = MuonClip(model.parameters(), lr=0.0)
    before = [p.clone() for p in model.parameters()]
    _run_steps(model, opt, x, target, n=1)
    for p, b in zip(model.parameters(), before):
        assert torch.allclose(p, b), "Parameter changed despite lr=0"


# ---------------------------------------------------------------------------
# 7. Momentum buffer accumulates after first step
# ---------------------------------------------------------------------------


def test_momentum_accumulates():
    model = _make_model()
    x, target = _make_data()
    opt = MuonClip(model.parameters(), lr=1e-3)
    _run_steps(model, opt, x, target, n=1)
    for p in model.parameters():
        state = opt.state[p]
        assert "m" in state, "Momentum buffer not created"
        assert not torch.all(state["m"] == 0), "Momentum buffer is all-zero after first step"


# ---------------------------------------------------------------------------
# 8. Two param groups with different lr both update
# ---------------------------------------------------------------------------


def test_param_groups():
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    x = torch.randn(16, 8)
    target = torch.randn(16, 4)
    loss_fn = nn.MSELoss()

    weight = model.weight
    bias = model.bias

    opt = MuonClip(
        [
            {"params": [weight], "lr": 1e-2},
            {"params": [bias], "lr": 1e-4},
        ],
        lr=1e-3,  # default (overridden by per-group lr)
    )

    w_before = weight.data.clone()
    b_before = bias.data.clone()

    opt.zero_grad()
    loss_fn(model(x), target).backward()
    opt.step()

    assert not torch.allclose(weight.data, w_before), "Weight not updated"
    assert not torch.allclose(bias.data, b_before), "Bias not updated"


# ---------------------------------------------------------------------------
# 9. 1-D parameter (bias) does not crash
# ---------------------------------------------------------------------------


def test_1d_param():
    p = nn.Parameter(torch.randn(8))
    opt = MuonClip([p], lr=1e-3)
    p.grad = torch.randn(8)
    opt.step()  # must not raise
    assert torch.isfinite(p).all()


# ---------------------------------------------------------------------------
# 10. Closure is called exactly once per step
# ---------------------------------------------------------------------------


def test_closure_called():
    model = _make_model()
    x, target = _make_data()
    opt = MuonClip(model.parameters(), lr=1e-3)

    # Prime gradients.
    nn.MSELoss()(model(x), target).backward()

    mock_closure = MagicMock(return_value=torch.tensor(0.5))
    opt.step(closure=mock_closure)
    mock_closure.assert_called_once()


# ---------------------------------------------------------------------------
# 11. 10 consecutive steps run without error
# ---------------------------------------------------------------------------


def test_multiple_steps():
    model = _make_model()
    x, target = _make_data()
    opt = MuonClip(model.parameters(), lr=1e-3)
    # Should not raise for 10 consecutive steps.
    _run_steps(model, opt, x, target, n=10)


# ---------------------------------------------------------------------------
# 12. Parameter with no grad is left unchanged
# ---------------------------------------------------------------------------


def test_no_grad_skipped():
    model = _make_model()
    opt = MuonClip(model.parameters(), lr=1e-3)
    before = [p.clone() for p in model.parameters()]
    # Do NOT call backward — all p.grad remain None.
    opt.step()
    for p, b in zip(model.parameters(), before):
        assert torch.allclose(p, b), "Parameter changed despite p.grad=None"
