"""Tests for ReLoRA (arXiv:2307.05695) — src/training/relora.py.

Tiny config: in_features=32, out_features=64, rank=4.
Pure PyTorch only — no scipy / sklearn / HuggingFace / peft.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.relora import (
    ReLoRALinear,
    ReLoRAScheduler,
    ReLoRAWrapper,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IN = 32
OUT = 64
RANK = 4
BATCH = 8


@pytest.fixture
def base_linear() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(IN, OUT, bias=True)


@pytest.fixture
def relora_layer(base_linear: nn.Linear) -> ReLoRALinear:
    torch.manual_seed(1)
    return ReLoRALinear.from_linear(base_linear, r=RANK)


@pytest.fixture
def x() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(BATCH, IN)


# ---------------------------------------------------------------------------
# Test 1: Forward output shape matches nn.Linear
# ---------------------------------------------------------------------------


def test_forward_output_shape(relora_layer: ReLoRALinear, x: torch.Tensor) -> None:
    """ReLoRALinear forward must produce same shape as the wrapped nn.Linear."""
    out = relora_layer(x)
    assert out.shape == (BATCH, OUT), f"Expected shape ({BATCH}, {OUT}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: Initial output ≈ base weight output (B=0 ⇒ zero delta)
# ---------------------------------------------------------------------------


def test_initial_output_equals_base(
    base_linear: nn.Linear, relora_layer: ReLoRALinear, x: torch.Tensor
) -> None:
    """With B=0 at init the LoRA delta is zero; output should match base linear."""
    with torch.no_grad():
        base_out = base_linear(x)
        relora_out = relora_layer(x)
    torch.testing.assert_close(
        relora_out,
        base_out,
        atol=1e-5,
        rtol=1e-5,
        msg="Initial ReLoRA output should equal base linear output (B=0)",
    )


# ---------------------------------------------------------------------------
# Test 3: merge() bakes B·A into W_0 correctly
# ---------------------------------------------------------------------------


def test_merge_updates_weight_correctly(relora_layer: ReLoRALinear) -> None:
    """After merge, W_0 should equal old_W_0 + B @ A."""
    W0_before = relora_layer.W_0.data.clone()
    relora_layer.B.data.clone()
    relora_layer.A.data.clone()

    # Give B a non-zero value so the update is non-trivial
    with torch.no_grad():
        relora_layer.B.data.normal_()

    B_nonzero = relora_layer.B.data.clone()
    A_snapshot = relora_layer.A.data.clone()

    relora_layer.merge()

    expected_W0 = W0_before + B_nonzero @ A_snapshot
    torch.testing.assert_close(
        relora_layer.W_0.data,
        expected_W0,
        atol=1e-5,
        rtol=1e-5,
        msg="After merge, W_0 should equal old_W_0 + B @ A",
    )


# ---------------------------------------------------------------------------
# Test 4: merge() resets B to zeros
# ---------------------------------------------------------------------------


def test_merge_resets_B_to_zero(relora_layer: ReLoRALinear) -> None:
    """After merge, B must be all zeros."""
    with torch.no_grad():
        relora_layer.B.data.normal_()  # make non-zero

    relora_layer.merge()

    assert relora_layer.B.data.abs().max().item() == 0.0, "B should be exactly zero after merge()"


# ---------------------------------------------------------------------------
# Test 5: merge() re-initialises A (A changes after merge)
# ---------------------------------------------------------------------------


def test_merge_reinitialises_A(relora_layer: ReLoRALinear) -> None:
    """After merge, A should be a fresh random init (different from before)."""
    A_before = relora_layer.A.data.clone()
    relora_layer.merge()
    A_after = relora_layer.A.data.clone()

    # The probability that two independent N(0,1) matrices of shape (4,32)
    # are identical is astronomically small.
    assert not torch.equal(A_before, A_after), (
        "A should be re-initialised after merge(), but it is unchanged"
    )


# ---------------------------------------------------------------------------
# Test 6: Gradients flow through A and B
# ---------------------------------------------------------------------------


def test_gradients_flow_through_A_and_B(relora_layer: ReLoRALinear, x: torch.Tensor) -> None:
    """Backward pass should populate gradients in both A and B.

    Note: at init B=0, so ∂loss/∂A = Bᵀ · upstream = 0 (correct autograd).
    We give B a non-zero value (as happens after the first optimizer step)
    before verifying that gradients flow through both factors.
    """
    # Simulate state after at least one optimizer update: B is non-zero
    with torch.no_grad():
        relora_layer.B.data.normal_()

    out = relora_layer(x)
    loss = out.sum()
    loss.backward()

    assert relora_layer.A.grad is not None, "No gradient for A"
    assert relora_layer.B.grad is not None, "No gradient for B"
    assert relora_layer.A.grad.abs().sum() > 0, "Zero gradient for A"
    assert relora_layer.B.grad.abs().sum() > 0, "Zero gradient for B"


# ---------------------------------------------------------------------------
# Test 7: After merge, new forward is sensible (not NaN / all-zero)
# ---------------------------------------------------------------------------


def test_forward_after_merge_is_sensible(relora_layer: ReLoRALinear, x: torch.Tensor) -> None:
    """After merge, forward pass should return finite, non-zero output."""
    with torch.no_grad():
        relora_layer.B.data.normal_()

    relora_layer.merge()

    with torch.no_grad():
        out = relora_layer(x)

    assert torch.isfinite(out).all(), "Output contains non-finite values after merge"
    assert out.abs().max().item() > 0.0, "Output is all-zero after merge"


# ---------------------------------------------------------------------------
# Test 8: ReLoRAScheduler — False before warmup, True at restart points
# ---------------------------------------------------------------------------


def test_scheduler_false_before_warmup() -> None:
    """should_restart should return False for steps < warmup_steps."""
    sched = ReLoRAScheduler(restart_every=100, warmup_steps=200)
    for step in range(200):
        assert not sched.should_restart(step), (
            f"should_restart returned True at step {step} (before warmup)"
        )


def test_scheduler_true_at_restart_point() -> None:
    """should_restart should return True at exact multiples of restart_every
    once past warmup."""
    sched = ReLoRAScheduler(restart_every=100, warmup_steps=50)
    # First valid restart: step=100
    assert sched.should_restart(100), "Expected True at step=100"
    assert sched.should_restart(200), "Expected True at step=200"
    assert not sched.should_restart(150), "Expected False at step=150"


# ---------------------------------------------------------------------------
# Test 9: ReLoRAScheduler — fires every restart_every steps (no off-by-one)
# ---------------------------------------------------------------------------


def test_scheduler_period_exact() -> None:
    """Scheduler fires exactly at multiples of restart_every >= warmup_steps."""
    restart_every = 50
    warmup = 100
    sched = ReLoRAScheduler(restart_every=restart_every, warmup_steps=warmup)

    fired = [s for s in range(1, 500) if sched.should_restart(s)]
    expected = [s for s in range(1, 500) if s >= warmup and s % restart_every == 0]
    assert fired == expected, f"Scheduler fired at {fired}, expected {expected}"


# ---------------------------------------------------------------------------
# Test 10: ReLoRAWrapper — only LoRA params (B, A) are trainable
# ---------------------------------------------------------------------------


def test_wrapper_only_lora_params_trainable() -> None:
    """After wrapping, only B and A should have requires_grad=True."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(IN, OUT, bias=True),
        nn.ReLU(),
        nn.Linear(OUT, IN, bias=True),
    )
    wrapper = ReLoRAWrapper(
        model, target_modules=["0", "2"], rank=RANK, restart_every=100, warmup_steps=10
    )

    for name, param in wrapper.model.named_parameters():
        is_lora = "B" in name or "A" in name
        # bias may remain trainable or not — check only weight-like params
        if "W_0" in name:
            assert not param.requires_grad, f"W_0 param '{name}' should be frozen"
        if is_lora:
            assert param.requires_grad, f"LoRA param '{name}' should be trainable"


# ---------------------------------------------------------------------------
# Test 11: ReLoRAWrapper.restart calls merge on all layers
# ---------------------------------------------------------------------------


def test_wrapper_restart_merges_all_layers() -> None:
    """wrapper.restart() must call merge() on every ReLoRALinear."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(IN, OUT, bias=False),
        nn.Linear(OUT, IN, bias=False),
    )
    wrapper = ReLoRAWrapper(
        model, target_modules=["0", "1"], rank=RANK, restart_every=100, warmup_steps=10
    )

    # Give B non-zero values in all layers
    W0_snapshots = []
    BA_products = []
    for layer in wrapper._relora_layers:
        with torch.no_grad():
            layer.B.data.normal_()
        W0_snapshots.append(layer.W_0.data.clone())
        BA_products.append(layer.B.data.clone() @ layer.A.data.clone())

    wrapper.restart()

    for i, layer in enumerate(wrapper._relora_layers):
        expected_W0 = W0_snapshots[i] + BA_products[i]
        torch.testing.assert_close(
            layer.W_0.data,
            expected_W0,
            atol=1e-5,
            rtol=1e-5,
            msg=f"Layer {i}: W_0 not updated correctly by restart()",
        )
        assert layer.B.data.abs().max().item() == 0.0, f"Layer {i}: B not zeroed by restart()"


# ---------------------------------------------------------------------------
# Test 12: ReLoRAWrapper.restart with optimizer — resets m_1, keeps m_2
# ---------------------------------------------------------------------------


def test_wrapper_restart_resets_first_moment() -> None:
    """restart(optimizer) should zero exp_avg (m_1) for LoRA params."""
    torch.manual_seed(0)
    model = nn.Linear(IN, OUT, bias=False)
    wrapper = ReLoRAWrapper(
        model,
        target_modules=[""],  # "" matches every module name
        rank=RANK,
        restart_every=100,
        warmup_steps=10,
    )

    trainable = wrapper.trainable_params_list()
    optimizer = torch.optim.Adam(trainable, lr=1e-3)

    # Run a fake forward/backward to populate optimizer state
    x = torch.randn(BATCH, IN)
    out = wrapper._relora_layers[0](x)
    loss = out.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Verify m_1 is non-zero before restart
    for p in trainable:
        state = optimizer.state.get(p, {})
        if "exp_avg" in state:
            # At least one param should have non-zero m_1
            break

    # Capture m_2 (exp_avg_sq) for comparison
    m2_before = {
        id(p): optimizer.state[p]["exp_avg_sq"].clone()
        for p in trainable
        if p in optimizer.state and "exp_avg_sq" in optimizer.state[p]
    }

    wrapper.restart(optimizer)

    # m_1 should now be zero for all LoRA params
    for p in trainable:
        state = optimizer.state.get(p)
        if state is None or "exp_avg" not in state:
            continue
        assert state["exp_avg"].abs().max().item() == 0.0, (
            "exp_avg (m_1) should be zeroed after warm restart"
        )

    # m_2 should be unchanged
    for p in trainable:
        state = optimizer.state.get(p)
        if state is None or "exp_avg_sq" not in state:
            continue
        pid = id(p)
        if pid in m2_before:
            torch.testing.assert_close(
                state["exp_avg_sq"],
                m2_before[pid],
                msg="exp_avg_sq (m_2) should not change during warm restart",
            )


# ---------------------------------------------------------------------------
# Test 13: Accumulated rank across 2 restarts > single LoRA rank
# ---------------------------------------------------------------------------


def test_accumulated_rank_after_two_restarts() -> None:
    """After 2 restarts, the cumulative weight change should have higher rank
    than a single LoRA step (rank r).

    Each restart adds a rank-r update to W_0 (in general position), so the
    cumulative rank after K restarts is up to K·r.
    """
    torch.manual_seed(7)
    in_f, out_f, r = 32, 64, 4
    linear = nn.Linear(in_f, out_f, bias=False)
    layer = ReLoRALinear.from_linear(linear, r=r)

    W0_initial = layer.W_0.data.clone()

    # Restart 1: give B random values, merge
    with torch.no_grad():
        layer.B.data.normal_()
    layer.merge()

    # Restart 2: B re-init'd by merge; give it new random values, merge again
    with torch.no_grad():
        layer.B.data.normal_()
    layer.merge()

    delta = layer.W_0.data - W0_initial  # accumulated weight change

    # Rank of delta (numerical rank with tolerance)
    singular_values = torch.linalg.svdvals(delta)
    tol = singular_values[0].item() * max(delta.shape) * 1e-5
    numerical_rank = (singular_values > tol).sum().item()

    # After 2 restarts we expect rank up to 2*r = 8; must be > r = 4
    assert numerical_rank > r, (
        f"Expected numerical rank > {r} after 2 restarts, got {numerical_rank}. "
        f"Top singular values: {singular_values[:10].tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 14: Numerical stability — no NaN/Inf after merge + new forward
# ---------------------------------------------------------------------------


def test_numerical_stability_after_merge() -> None:
    """No NaN or Inf values should appear after repeated merge + forward."""
    torch.manual_seed(99)
    linear = nn.Linear(IN, OUT, bias=True)
    layer = ReLoRALinear.from_linear(linear, r=RANK)
    x = torch.randn(BATCH, IN)

    for _ in range(5):
        with torch.no_grad():
            layer.B.data.normal_()
        layer.merge()
        with torch.no_grad():
            out = layer(x)
        assert torch.isfinite(out).all(), "Non-finite values in output after merge + forward"
        assert not torch.isnan(out).any(), "NaN in output after merge + forward"
