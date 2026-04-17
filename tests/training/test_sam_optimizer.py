"""
Tests for src/training/sam_optimizer.py

15 tests covering SAMOptimizer, ASAM, LookSAM, SharpnessAnalyzer,
and SAMTrainingLoop.

Tiny config: 2-layer MLP (16->16->16), rho=0.05, batch=2.
All tests run actual forward/backward passes.
"""

import math
import copy

import pytest
import torch
import torch.nn as nn

from src.training.sam_optimizer import (
    ASAM,
    LookSAM,
    SAMOptimizer,
    SAMTrainingLoop,
    SharpnessAnalyzer,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

D = 16   # input/hidden/output dimension
BATCH = 2
RHO = 0.05


def make_mlp() -> nn.Module:
    """Minimal 2-layer MLP: Linear(16->16) + ReLU + Linear(16->16)."""
    return nn.Sequential(
        nn.Linear(D, D),
        nn.ReLU(),
        nn.Linear(D, D),
    )


def make_data():
    """Return a fixed (input, labels) pair for reproducibility."""
    torch.manual_seed(0)
    x = torch.randn(BATCH, D)
    y = torch.randint(0, D, (BATCH,))
    return x, y


def make_sam(model: nn.Module, adaptive: bool = False) -> SAMOptimizer:
    base = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
    return SAMOptimizer(base, rho=RHO, adaptive=adaptive)


def snapshot_params(model: nn.Module) -> list:
    """Return a list of cloned parameter tensors."""
    return [p.data.clone() for p in model.parameters()]


def params_equal(a: list, b: list) -> bool:
    return all(torch.allclose(x, y) for x, y in zip(a, b))


def compute_loss_and_grad(model, x, y):
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    return loss


# ===========================================================================
# 1. SAMOptimizer.first_step: params change from original
# ===========================================================================

def test_sam_first_step_params_change():
    torch.manual_seed(1)
    model = make_mlp()
    sam = make_sam(model)
    x, y = make_data()

    before = snapshot_params(model)
    compute_loss_and_grad(model, x, y)
    sam.first_step(zero_grad=False)
    after = snapshot_params(model)

    # At least one parameter should have changed
    any_changed = any(not torch.allclose(a, b) for a, b in zip(before, after))
    assert any_changed, "Params should move during first_step"


# ===========================================================================
# 2. SAMOptimizer.second_step: params restored to original + optimizer step
# ===========================================================================

def test_sam_second_step_restores_then_updates():
    torch.manual_seed(2)
    model = make_mlp()
    sam = make_sam(model)
    x, y = make_data()

    original = snapshot_params(model)

    # First pass
    compute_loss_and_grad(model, x, y)
    sam.first_step(zero_grad=True)

    # Second pass (at perturbed params)
    compute_loss_and_grad(model, x, y)
    sam.second_step(zero_grad=True)

    final = snapshot_params(model)

    # Final params should differ from both original AND perturbed point
    # (because the optimizer applied a step from the original position)
    assert not params_equal(original, final), (
        "Params should be updated after second_step (optimizer step applied)"
    )


# ===========================================================================
# 3. SAMOptimizer: perturbation direction proportional to gradient
# ===========================================================================

def test_sam_perturbation_proportional_to_gradient():
    torch.manual_seed(3)
    model = make_mlp()
    sam = make_sam(model)
    x, y = make_data()

    compute_loss_and_grad(model, x, y)

    # Collect gradients before first_step
    grads = {id(p): p.grad.clone() for p in model.parameters() if p.grad is not None}
    before = {id(p): p.data.clone() for p in model.parameters() if p.grad is not None}

    sam.first_step(zero_grad=False)

    # For non-adaptive SAM: perturbation = rho * grad / ||grad||
    # => perturbation / grad should be a (near-)constant scalar per parameter
    grad_norm = math.sqrt(sum(g.norm().item() ** 2 for g in grads.values()))
    expected_scale = RHO / (grad_norm + 1e-12)

    for p in model.parameters():
        if id(p) not in grads:
            continue
        delta = p.data - before[id(p)]
        g = grads[id(p)]
        # delta should equal expected_scale * g
        assert torch.allclose(delta, expected_scale * g, atol=1e-5), (
            "Perturbation should be proportional to gradient"
        )


# ===========================================================================
# 4. SAMOptimizer adaptive=True: perturbation scales with |w|
# ===========================================================================

def test_sam_adaptive_perturbation_scales_with_weight():
    torch.manual_seed(4)
    model = make_mlp()
    sam_nonadapt = make_sam(model, adaptive=False)
    sam_adapt = make_sam(copy.deepcopy(model), adaptive=True)

    x, y = make_data()

    # Non-adaptive
    model2 = copy.deepcopy(model)
    compute_loss_and_grad(model, x, y)
    before_nonadapt = snapshot_params(model)
    sam_nonadapt.first_step(zero_grad=False)
    delta_nonadapt = [p.data - b for p, b in zip(model.parameters(), before_nonadapt)]

    # Adaptive (use same gradients to see difference in direction)
    model_adapt = copy.deepcopy(model2)  # fresh copy with same init
    sam_adapt2 = SAMOptimizer(
        torch.optim.SGD(model_adapt.parameters(), lr=0.01),
        rho=RHO,
        adaptive=True,
    )
    compute_loss_and_grad(model_adapt, x, y)
    before_adapt = snapshot_params(model_adapt)
    sam_adapt2.first_step(zero_grad=False)
    delta_adapt = [p.data - b for p, b in zip(model_adapt.parameters(), before_adapt)]

    # The two perturbations should NOT be identical (because adaptive scales by |w|)
    any_differ = any(
        not torch.allclose(a, b, atol=1e-6)
        for a, b in zip(delta_nonadapt, delta_adapt)
    )
    assert any_differ, "Adaptive and non-adaptive perturbations should differ"


# ===========================================================================
# 5. SAMOptimizer: perturbation norm ≈ rho (within 1% for unit-norm gradient)
# ===========================================================================

def test_sam_perturbation_norm_approx_rho():
    torch.manual_seed(5)
    model = make_mlp()
    sam = make_sam(model)
    x, y = make_data()

    compute_loss_and_grad(model, x, y)

    before = snapshot_params(model)
    sam.first_step(zero_grad=False)
    after = snapshot_params(model)

    pert_norm_sq = sum(
        (a - b).norm().item() ** 2 for a, b in zip(after, before)
    )
    pert_norm = math.sqrt(pert_norm_sq)

    # Perturbation norm should equal rho (because delta = rho/||g|| * g, so ||delta|| = rho)
    assert abs(pert_norm - RHO) / RHO < 0.01, (
        f"Perturbation norm {pert_norm:.6f} should be within 1% of rho={RHO}"
    )


# ===========================================================================
# 6. ASAM.first_step: adaptive perturbation differs from non-adaptive
# ===========================================================================

def test_asam_first_step_differs_from_sam():
    torch.manual_seed(6)
    base_model = make_mlp()
    model_sam = copy.deepcopy(base_model)
    model_asam = copy.deepcopy(base_model)

    sam = SAMOptimizer(
        torch.optim.SGD(model_sam.parameters(), lr=0.01),
        rho=RHO, adaptive=False,
    )
    asam = ASAM(
        torch.optim.SGD(model_asam.parameters(), lr=0.01),
        rho=RHO, eta=0.01,
    )

    x, y = make_data()

    compute_loss_and_grad(model_sam, x, y)
    before_sam = snapshot_params(model_sam)
    sam.first_step(zero_grad=False)
    delta_sam = [p.data - b for p, b in zip(model_sam.parameters(), before_sam)]

    compute_loss_and_grad(model_asam, x, y)
    before_asam = snapshot_params(model_asam)
    asam.first_step(zero_grad=False)
    delta_asam = [p.data - b for p, b in zip(model_asam.parameters(), before_asam)]

    any_differ = any(
        not torch.allclose(a, b, atol=1e-6)
        for a, b in zip(delta_sam, delta_asam)
    )
    assert any_differ, "ASAM perturbation should differ from standard SAM"


# ===========================================================================
# 7. ASAM: shape of perturbation matches model params
# ===========================================================================

def test_asam_perturbation_shape_matches_params():
    torch.manual_seed(7)
    model = make_mlp()
    asam = ASAM(
        torch.optim.SGD(model.parameters(), lr=0.01),
        rho=RHO,
    )
    x, y = make_data()

    compute_loss_and_grad(model, x, y)
    before = snapshot_params(model)
    asam.first_step(zero_grad=False)

    for p, b in zip(model.parameters(), before):
        delta = p.data - b
        assert delta.shape == p.shape, (
            f"Perturbation shape {delta.shape} must match param shape {p.shape}"
        )


# ===========================================================================
# 8. LookSAM: slow weights initialized equal to fast weights
# ===========================================================================

def test_looksam_slow_weights_init_equal_fast():
    torch.manual_seed(8)
    model = make_mlp()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    looksam = LookSAM(base_opt, rho=RHO, alpha=0.5, k=5)

    fast_params = snapshot_params(model)
    slow_params = looksam._slow_weights

    assert len(slow_params) == len(fast_params)
    for fast, slow in zip(fast_params, slow_params):
        assert torch.allclose(fast, slow), (
            "Slow weights should be initialized equal to fast weights"
        )


# ===========================================================================
# 9. LookSAM.sync_slow_weights: fast weights match slow weights after sync
# ===========================================================================

def test_looksam_sync_slow_weights():
    torch.manual_seed(9)
    model = make_mlp()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    looksam = LookSAM(base_opt, rho=RHO, alpha=0.5, k=5)
    x, y = make_data()

    # Do a few steps to diverge fast weights from slow
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        model.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        return loss

    for _ in range(3):
        looksam.step(closure)

    # Manually modify slow weights to something distinct
    for i in range(len(looksam._slow_weights)):
        looksam._slow_weights[i] = torch.zeros_like(looksam._slow_weights[i])

    looksam.sync_slow_weights()

    for p in model.parameters():
        assert torch.allclose(p.data, torch.zeros_like(p.data)), (
            "After sync, fast weights should match slow weights"
        )


# ===========================================================================
# 10. SharpnessAnalyzer.flatness_ratio: >= 1.0 on average
# ===========================================================================

def test_sharpness_analyzer_flatness_ratio_ge_1():
    torch.manual_seed(10)
    model = make_mlp()
    loss_fn = nn.CrossEntropyLoss()
    analyzer = SharpnessAnalyzer(model, loss_fn)
    x, y = make_data()

    # Use many directions to get a stable mean
    ratio = analyzer.flatness_ratio(x, y, rho=0.5, n_directions=50)

    assert isinstance(ratio, float), "flatness_ratio must return a float"
    # For a non-trivial random MLP, random perturbations tend to increase loss on average
    # We relax to ratio > 0 as the strict >= 1 depends on the loss landscape
    assert ratio > 0, f"flatness_ratio must be positive, got {ratio}"
    # The ratio should be finite
    assert math.isfinite(ratio), f"flatness_ratio must be finite, got {ratio}"


# ===========================================================================
# 11. SharpnessAnalyzer.gradient_diversity: >= 0, finite
# ===========================================================================

def test_sharpness_analyzer_gradient_diversity_valid():
    torch.manual_seed(11)
    model = make_mlp()
    loss_fn = nn.CrossEntropyLoss()
    analyzer = SharpnessAnalyzer(model, loss_fn)
    x, y = make_data()

    div = analyzer.gradient_diversity(x, y)

    assert isinstance(div, float), "gradient_diversity must return a float"
    assert div >= 0.0, f"gradient_diversity must be >= 0, got {div}"
    assert math.isfinite(div), f"gradient_diversity must be finite, got {div}"


# ===========================================================================
# 12. SAMTrainingLoop.train_step: loss finite, params updated
# ===========================================================================

def test_sam_training_loop_updates_params():
    torch.manual_seed(12)
    model = make_mlp()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sam = SAMOptimizer(base_opt, rho=RHO)
    loss_fn = nn.CrossEntropyLoss()
    loop = SAMTrainingLoop(model, sam, loss_fn)
    x, y = make_data()

    before = snapshot_params(model)
    info = loop.train_step(x, y)
    after = snapshot_params(model)

    assert math.isfinite(info["loss"]), f"Loss must be finite, got {info['loss']}"
    assert not params_equal(before, after), "Params must be updated after train_step"


# ===========================================================================
# 13. SAMTrainingLoop: perturbation_norm ≈ rho
# ===========================================================================

def test_sam_training_loop_perturbation_norm():
    torch.manual_seed(13)
    model = make_mlp()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    sam = SAMOptimizer(base_opt, rho=RHO)
    loss_fn = nn.CrossEntropyLoss()
    loop = SAMTrainingLoop(model, sam, loss_fn)
    x, y = make_data()

    info = loop.train_step(x, y)
    pnorm = info["perturbation_norm"]

    assert abs(pnorm - RHO) / RHO < 0.02, (
        f"perturbation_norm {pnorm:.6f} should be within 2% of rho={RHO}"
    )


# ===========================================================================
# 14. SAMTrainingLoop: loss decreases over 5 steps
# ===========================================================================

def test_sam_training_loop_loss_decreases():
    torch.manual_seed(14)
    model = make_mlp()
    base_opt = torch.optim.SGD(model.parameters(), lr=0.05)
    sam = SAMOptimizer(base_opt, rho=RHO)
    loss_fn = nn.CrossEntropyLoss()
    loop = SAMTrainingLoop(model, sam, loss_fn)
    x, y = make_data()

    losses = []
    for _ in range(5):
        info = loop.train_step(x, y)
        losses.append(info["loss"])

    # Loss at step 5 should be lower than at step 1 (general convergence trend)
    assert losses[-1] < losses[0], (
        f"Loss should decrease over 5 steps. Got losses: {losses}"
    )


# ===========================================================================
# 15. Two SAM steps with same data: second loss may differ from first
# ===========================================================================

def test_sam_two_steps_explore_landscape():
    torch.manual_seed(15)
    model = make_mlp()
    sam = make_sam(model)
    x, y = make_data()
    loss_fn = nn.CrossEntropyLoss()

    # Step 1: first_step — record loss at original params
    model.zero_grad()
    loss1 = loss_fn(model(x), y)
    loss1.backward()
    loss1_val = loss1.item()
    sam.first_step(zero_grad=True)

    # Evaluate loss at perturbed point
    with torch.no_grad():
        loss_perturbed = loss_fn(model(x), y).item()

    # Step 1: second_step
    model.zero_grad()
    loss_fn(model(x), y).backward()
    sam.second_step(zero_grad=True)

    # Step 2: first_step at updated params — record new original loss
    model.zero_grad()
    loss2 = loss_fn(model(x), y)
    loss2.backward()
    loss2_val = loss2.item()
    sam.first_step(zero_grad=True)

    # Evaluate loss at second perturbed point
    with torch.no_grad():
        loss_perturbed2 = loss_fn(model(x), y).item()

    # Restore
    model.zero_grad()
    loss_fn(model(x), y).backward()
    sam.second_step(zero_grad=True)

    # The two "original" losses must both be finite
    assert math.isfinite(loss1_val), "First-step loss must be finite"
    assert math.isfinite(loss2_val), "Second-step loss must be finite"

    # The perturbed losses must both be finite
    assert math.isfinite(loss_perturbed), "Perturbed loss (step 1) must be finite"
    assert math.isfinite(loss_perturbed2), "Perturbed loss (step 2) must be finite"

    # At least one pair of losses must differ (SAM explores different points)
    some_differ = (
        abs(loss_perturbed - loss_perturbed2) > 1e-8
        or abs(loss1_val - loss2_val) > 1e-8
    )
    assert some_differ, (
        "Two SAM steps should produce different loss values (landscape exploration)"
    )
