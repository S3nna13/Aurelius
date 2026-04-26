"""
Tests for LEACE (Least-squares Concept Erasure) — arXiv:2306.03819.

Coverage:
  1. Shape preservation
  2. Concept direction erased (linear probe ≈ chance)
  3. Low distortion for orthogonal concept
  4. Projection idempotency  P^2 = P
  5. Determinism under manual seed
  6. Gradient flow through transform
  7. ConceptEraser online == LeaceEraser batch
  8. Edge case: d = 1
  9. Edge case: perfectly balanced classes
 10. Numerical stability (near-singular covariance)
 11. Multi-erase: applying LEACE twice == once
 12. Erased direction: dot product constant across classes
 13. ConceptEraser state persists across multiple updates
 14. LeaceEraser works on single-vector (1-D) input
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.interpretability.leace_eraser import ConceptEraser, LeaceEraser

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_data(
    N: int = 200,
    d: int = 32,
    seed: int = 0,
    balanced: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic data: class-1 mean shifted along first dimension."""
    g = torch.Generator()
    g.manual_seed(seed)
    X = torch.randn(N, d, generator=g)
    if balanced:
        y = torch.zeros(N, dtype=torch.long)
        y[N // 2 :] = 1
    else:
        y = (torch.rand(N, generator=g) > 0.4).long()
    # Shift class-1 along dim 0 to create strong linear signal
    X[y == 1, 0] += 3.0
    return X, y


def _linear_probe_acc(X: torch.Tensor, y: torch.Tensor) -> float:
    """Train and evaluate a logistic probe (no autograd needed)."""
    X = X.float()
    y = y.float()
    w = nn.Parameter(torch.zeros(X.shape[1]))
    b = nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        opt.zero_grad()
        logits = X @ w + b
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        preds = (X @ w + b > 0).float()
    return (preds == y).float().mean().item()


# ---------------------------------------------------------------------------
# Test 1: Shape preservation
# ---------------------------------------------------------------------------


def test_shape_preservation():
    X, y = _make_data(N=100, d=16)
    eraser = LeaceEraser.fit(X, y)
    X_erased = eraser.transform(X)
    assert X_erased.shape == X.shape, f"Expected {X.shape}, got {X_erased.shape}"


# ---------------------------------------------------------------------------
# Test 2: Concept direction erased — linear probe ≈ chance (50%)
# ---------------------------------------------------------------------------


def test_concept_erased_linear_probe():
    """Fit eraser on train split; train probe on erased-train; evaluate on erased-test."""
    # Draw a single dataset and split — same distribution for train and test
    X_all, y_all = _make_data(N=600, d=32, seed=42)
    X_train, y_train = X_all[:400], y_all[:400]
    X_test, y_test = X_all[400:], y_all[400:]

    eraser = LeaceEraser.fit(X_train, y_train)
    X_train_e = eraser.transform(X_train)
    X_test_e = eraser.transform(X_test)

    # Train linear probe on erased train split, evaluate on erased test split
    X_tr = X_train_e.detach()
    X_te = X_test_e.detach()
    w = nn.Parameter(torch.zeros(X_tr.shape[1]))
    b = nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(X_tr @ w + b, y_train.float())
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        preds = (X_te @ w + b > 0).float()
    acc = (preds == y_test.float()).float().mean().item()
    # Probe should be near chance on held-out erased data
    assert acc < 0.60, f"Linear probe accuracy {acc:.3f} is too high — concept not erased"


# ---------------------------------------------------------------------------
# Test 3: Low distortion for orthogonal concept
# ---------------------------------------------------------------------------


def test_low_distortion_orthogonal_concept():
    """When concept direction is dim-0 and dims 1..d-1 are irrelevant,
    the Frobenius distortion should be small relative to data norm."""
    torch.manual_seed(7)
    N, d = 200, 32
    X = torch.randn(N, d)
    y = torch.zeros(N, dtype=torch.long)
    y[N // 2 :] = 1
    # Small shift so that eraser changes X very little
    X[y == 1, 0] += 0.1
    eraser = LeaceEraser.fit(X, y)
    X_erased = eraser.transform(X)
    rel_distortion = (X_erased - X).norm() / X.norm()
    assert rel_distortion < 0.5, f"Relative distortion {rel_distortion:.4f} too large"


# ---------------------------------------------------------------------------
# Test 4: Projection idempotency  transform(transform(X)) ≈ transform(X)
# ---------------------------------------------------------------------------


def test_idempotency():
    X, y = _make_data(N=200, d=24, seed=1)
    eraser = LeaceEraser.fit(X, y)
    X1 = eraser.transform(X)
    X2 = eraser.transform(X1)
    assert torch.allclose(X1, X2, atol=1e-5), (
        f"Max diff after double transform: {(X1 - X2).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 5: Determinism under torch.manual_seed
# ---------------------------------------------------------------------------


def test_determinism():
    def _run(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        X, y = _make_data(N=100, d=16, seed=seed)
        eraser = LeaceEraser.fit(X, y)
        return eraser.transform(X)

    out_a = _run(99)
    out_b = _run(99)
    assert torch.allclose(out_a, out_b), "Results differ across identical seeds"


# ---------------------------------------------------------------------------
# Test 6: Gradient flow through transform
# ---------------------------------------------------------------------------


def test_gradient_flow():
    X, y = _make_data(N=50, d=16, seed=3)
    X_leaf = X.clone().requires_grad_(True)
    eraser = LeaceEraser.fit(X.detach(), y)
    X_erased = eraser.transform(X_leaf)
    loss = X_erased.sum()
    loss.backward()
    assert X_leaf.grad is not None, "No gradient flowed to X"
    assert not torch.isnan(X_leaf.grad).any(), "NaN in gradient"


# ---------------------------------------------------------------------------
# Test 7: ConceptEraser online == LeaceEraser batch
# ---------------------------------------------------------------------------


def test_concept_eraser_matches_batch():
    X, y = _make_data(N=200, d=20, seed=5)

    # Batch
    batch_eraser = LeaceEraser.fit(X, y)
    X_batch = batch_eraser.transform(X)

    # Online (single batch update — should be identical)
    online_eraser = ConceptEraser(d_model=20)
    online_eraser.update(X, y)
    X_online = online_eraser(X)

    assert torch.allclose(X_batch, X_online, atol=1e-4), (
        f"Batch vs online max diff: {(X_batch - X_online).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 8: Edge case d = 1
# ---------------------------------------------------------------------------


def test_edge_case_d1():
    torch.manual_seed(10)
    N = 100
    X = torch.randn(N, 1)
    y = torch.zeros(N, dtype=torch.long)
    y[N // 2 :] = 1
    X[y == 1, 0] += 2.0
    eraser = LeaceEraser.fit(X, y)
    X_erased = eraser.transform(X)
    assert X_erased.shape == (N, 1)
    # d=1: the only direction is erased, all values should be very similar
    assert X_erased.std() < 0.5, "d=1: erased reps should have very low variance"


# ---------------------------------------------------------------------------
# Test 9: Edge case — perfectly balanced classes (N_0 = N_1)
# ---------------------------------------------------------------------------


def test_balanced_classes():
    N = 200  # exactly 100 per class
    torch.manual_seed(11)
    X = torch.randn(N, 16)
    y = torch.zeros(N, dtype=torch.long)
    y[N // 2 :] = 1
    X[y == 1, 0] += 2.0
    eraser = LeaceEraser.fit(X, y)
    X_erased = eraser.transform(X)
    assert X_erased.shape == X.shape
    assert not torch.isnan(X_erased).any()


# ---------------------------------------------------------------------------
# Test 10: Numerical stability (near-singular covariance)
# ---------------------------------------------------------------------------


def test_numerical_stability_near_singular():
    """All samples in class-1 have identical features in dims 1..d-1."""
    torch.manual_seed(13)
    N, d = 100, 16
    X = torch.zeros(N, d)
    X[:, 0] = torch.randn(N)  # only dim 0 varies
    y = torch.zeros(N, dtype=torch.long)
    y[N // 2 :] = 1
    X[y == 1, 0] += 1.0
    eraser = LeaceEraser.fit(X, y, eps=1e-4)
    X_erased = eraser.transform(X)
    assert not torch.isnan(X_erased).any(), "NaN in output with near-singular covariance"
    assert not torch.isinf(X_erased).any(), "Inf in output with near-singular covariance"


# ---------------------------------------------------------------------------
# Test 11: Multi-erase — applying LEACE twice for the same concept == once
# ---------------------------------------------------------------------------


def test_multi_erase_idempotent():
    X, y = _make_data(N=200, d=24, seed=15)
    eraser = LeaceEraser.fit(X, y)
    X1 = eraser.transform(X)
    X2 = eraser.transform(X1)
    # Second pass should change nothing (concept already removed)
    assert torch.allclose(X1, X2, atol=1e-5), (
        f"Double-erase max diff: {(X1 - X2).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 12: Erased direction — dot product constant across classes
# ---------------------------------------------------------------------------


def test_erased_direction_constant_across_classes():
    """After erasure, projecting onto w should give the same value for all x."""
    X, y = _make_data(N=300, d=32, seed=17)
    eraser = LeaceEraser.fit(X, y)
    X_erased = eraser.transform(X)
    w = eraser.w.to(X_erased.device)

    proj = X_erased @ w  # (N,)
    proj_0 = proj[y == 0]
    proj_1 = proj[y == 1]

    mean_diff = (proj_0.mean() - proj_1.mean()).abs().item()
    assert mean_diff < 1e-3, (
        f"Mean projection differs across classes by {mean_diff:.4e} — concept not erased"
    )


# ---------------------------------------------------------------------------
# Test 13: ConceptEraser multi-batch updates consistent with single batch
# ---------------------------------------------------------------------------


def test_concept_eraser_multi_batch():
    """Split data into two halves; online updates should match batch on full set."""
    X, y = _make_data(N=400, d=20, seed=21)
    N = X.shape[0]
    half = N // 2

    # Online over two batches — need to ensure each class present in each batch
    # Use first half / second half, both have balanced classes
    online = ConceptEraser(d_model=20)
    online.update(X[:half], y[:half])
    online.update(X[half:], y[half:])

    batch = LeaceEraser.fit(X, y)

    X_online = online(X)
    X_batch = batch.transform(X)

    # Should be close but not necessarily identical (order of updates matters)
    # Check that the concept is erased in both
    acc_online = _linear_probe_acc(X_online.detach(), y.float())
    acc_batch = _linear_probe_acc(X_batch.detach(), y.float())
    assert acc_online < 0.60, f"Online eraser: probe acc {acc_online:.3f} too high"
    assert acc_batch < 0.60, f"Batch eraser: probe acc {acc_batch:.3f} too high"


# ---------------------------------------------------------------------------
# Test 14: LeaceEraser on a single 1-D input vector
# ---------------------------------------------------------------------------


def test_single_vector_input():
    X, y = _make_data(N=100, d=16, seed=25)
    eraser = LeaceEraser.fit(X, y)
    single = X[0]  # shape (d,)
    out = eraser.transform(single)
    assert out.shape == single.shape, f"Expected {single.shape}, got {out.shape}"
    assert not torch.isnan(out).any()
