"""
tests/interpretability/test_polysemanticity.py

Tests for src/interpretability/polysemanticity.py

Tests verify:
  1.  Shape: polysemanticity_index(W) returns (d_out,) for W of shape (d_in, d_out)
  2.  Monosemantic case: one-hot weight columns → PI close to 0
  3.  Polysemantic case: uniform weight column → PI close to 1
  4a. Participation ratio: identity activations → PR ≈ d
  4b. Participation ratio: rank-1 activations → PR ≈ 1
  5a. Superposition score: orthogonal features → SS ≈ 0
  5b. Superposition score: identical features → SS ≈ 1
  6a. Activation sparsity: all-zeros → sparsity = 1.0
  6b. Activation sparsity: all large → sparsity = 0.0
  7.  Determinism under torch.manual_seed
  8.  analyze() returns expected dict keys
  9.  Edge case: square weight matrix (d_in == d_out)
  10. Edge case: single neuron (d_out == 1)
  11. PI range: all PI values in [0, 1]
  12. SS range: superposition_score in [0, 1]
  13. Gradient flow: PI, PR, SS are differentiable
  14. Numerical stability: no NaN/Inf on random weight matrices
"""

import math

import pytest
import torch

from src.interpretability.polysemanticity import PolysemanticitAnalyzer


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
D_IN = 16
D_OUT = 8
N_SAMPLES = 64

ANALYZER = PolysemanticitAnalyzer(threshold=0.1)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _random_W(d_in: int = D_IN, d_out: int = D_OUT, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(d_in, d_out, generator=g)


def _random_acts(n: int = N_SAMPLES, d: int = D_OUT, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(n, d, generator=g)


# ---------------------------------------------------------------------------
# 1. Shape: polysemanticity_index returns (d_out,)
# ---------------------------------------------------------------------------

def test_pi_output_shape():
    W = _random_W()
    pi = ANALYZER.polysemanticity_index(W)
    assert pi.shape == (D_OUT,), f"Expected ({D_OUT},), got {pi.shape}"


# ---------------------------------------------------------------------------
# 2. Monosemantic case: one-hot weight columns → PI close to 0
# ---------------------------------------------------------------------------

def test_pi_monosemantic_near_zero():
    """One-hot column = all weight on one feature → minimum entropy → PI ≈ 0."""
    # Build a weight matrix where each column is a unit standard basis vector
    W = torch.zeros(D_IN, D_OUT)
    for col in range(D_OUT):
        W[col % D_IN, col] = 1.0

    pi = ANALYZER.polysemanticity_index(W)
    assert (pi < 0.05).all(), (
        f"One-hot columns should give PI ≈ 0, got max PI = {pi.max().item():.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Polysemantic case: uniform weight column → PI close to 1
# ---------------------------------------------------------------------------

def test_pi_polysemantic_near_one():
    """Uniform weights = maximum entropy → PI ≈ 1."""
    W = torch.ones(D_IN, D_OUT)  # every entry = 1 → uniform distribution
    pi = ANALYZER.polysemanticity_index(W)
    # All columns identical and uniform
    assert (pi > 0.9).all(), (
        f"Uniform columns should give PI close to 1, got min PI = {pi.min().item():.4f}"
    )


# ---------------------------------------------------------------------------
# 4a. Participation ratio: identity activations → PR ≈ d
# ---------------------------------------------------------------------------

def test_pr_identity_activations():
    """
    Identity-like activations (rows = standard basis vectors cycling) are
    equivalent to the identity covariance → PR = d.
    We use a large N so each basis direction is equally represented.
    """
    d = D_OUT
    # Build N activations that cover each basis direction equally
    repeats = N_SAMPLES // d
    acts = torch.zeros(repeats * d, d)
    for i in range(d):
        acts[i * repeats:(i + 1) * repeats, i] = 1.0

    pr = ANALYZER.participation_ratio(acts)
    # PR should be close to d (within 10%)
    assert abs(pr - d) < 0.5, (
        f"Identity activations should give PR ≈ {d}, got {pr:.4f}"
    )


# ---------------------------------------------------------------------------
# 4b. Participation ratio: rank-1 activations → PR ≈ 1
# ---------------------------------------------------------------------------

def test_pr_rank1_activations():
    """All activations along one direction → covariance has rank 1 → PR ≈ 1."""
    acts = torch.zeros(N_SAMPLES, D_OUT)
    acts[:, 0] = torch.arange(1, N_SAMPLES + 1, dtype=torch.float32)

    pr = ANALYZER.participation_ratio(acts)
    assert abs(pr - 1.0) < 0.05, (
        f"Rank-1 activations should give PR ≈ 1, got {pr:.4f}"
    )


# ---------------------------------------------------------------------------
# 5a. Superposition score: orthogonal features → SS ≈ 0
# ---------------------------------------------------------------------------

def test_ss_orthogonal_near_zero():
    """Standard basis columns are orthogonal → SS = 0."""
    d = D_OUT
    W = torch.eye(d)  # square, columns are orthonormal basis
    ss = ANALYZER.superposition_score(W)
    assert ss < 1e-5, (
        f"Orthogonal columns should give SS ≈ 0, got {ss:.6f}"
    )


# ---------------------------------------------------------------------------
# 5b. Superposition score: identical features → SS ≈ 1
# ---------------------------------------------------------------------------

def test_ss_identical_near_one():
    """All columns the same unit vector → cosine sim = 1 → SS = 1."""
    W = torch.ones(D_IN, D_OUT)  # all columns identical
    ss = ANALYZER.superposition_score(W)
    assert abs(ss - 1.0) < 1e-5, (
        f"Identical columns should give SS ≈ 1, got {ss:.6f}"
    )


# ---------------------------------------------------------------------------
# 6a. Activation sparsity: all zeros → sparsity = 1.0
# ---------------------------------------------------------------------------

def test_sparsity_all_zeros():
    acts = torch.zeros(N_SAMPLES, D_OUT)
    sparsity = ANALYZER.activation_sparsity(acts)
    assert torch.allclose(sparsity, torch.ones(D_OUT)), (
        f"All-zero activations should give sparsity 1.0, got {sparsity}"
    )


# ---------------------------------------------------------------------------
# 6b. Activation sparsity: all large values → sparsity = 0.0
# ---------------------------------------------------------------------------

def test_sparsity_all_nonzero():
    # Values well above threshold=0.1
    acts = torch.full((N_SAMPLES, D_OUT), 10.0)
    sparsity = ANALYZER.activation_sparsity(acts)
    assert torch.allclose(sparsity, torch.zeros(D_OUT)), (
        f"Large activations should give sparsity 0.0, got {sparsity}"
    )


# ---------------------------------------------------------------------------
# 7. Determinism under torch.manual_seed
# ---------------------------------------------------------------------------

def test_determinism():
    torch.manual_seed(42)
    W1 = torch.randn(D_IN, D_OUT)
    acts1 = torch.randn(N_SAMPLES, D_OUT)

    torch.manual_seed(42)
    W2 = torch.randn(D_IN, D_OUT)
    acts2 = torch.randn(N_SAMPLES, D_OUT)

    pi1 = ANALYZER.polysemanticity_index(W1)
    pi2 = ANALYZER.polysemanticity_index(W2)
    assert torch.allclose(pi1, pi2), "PI should be deterministic given same seed"

    pr1 = ANALYZER.participation_ratio(acts1)
    pr2 = ANALYZER.participation_ratio(acts2)
    assert abs(pr1 - pr2) < 1e-6, "PR should be deterministic given same seed"

    ss1 = ANALYZER.superposition_score(W1)
    ss2 = ANALYZER.superposition_score(W2)
    assert abs(ss1 - ss2) < 1e-6, "SS should be deterministic given same seed"


# ---------------------------------------------------------------------------
# 8. analyze() returns expected dict keys (without activations)
# ---------------------------------------------------------------------------

def test_analyze_keys_without_activations():
    W = _random_W()
    result = ANALYZER.analyze(W)
    required_keys = {
        "polysemanticity_index",
        "superposition_score",
        "feature_geometry",
        "mean_pi",
        "n_neurons",
        "n_features",
    }
    for k in required_keys:
        assert k in result, f"Missing key in analyze() output: {k}"


def test_analyze_keys_with_activations():
    W = _random_W()
    acts = _random_acts()
    result = ANALYZER.analyze(W, activations=acts)
    required_keys = {
        "polysemanticity_index",
        "superposition_score",
        "feature_geometry",
        "mean_pi",
        "n_neurons",
        "n_features",
        "participation_ratio",
        "activation_sparsity",
        "mean_sparsity",
    }
    for k in required_keys:
        assert k in result, f"Missing key in analyze() output: {k}"


# ---------------------------------------------------------------------------
# 9. Edge case: square weight matrix (d_in == d_out)
# ---------------------------------------------------------------------------

def test_edge_case_square_weight():
    d = D_OUT
    W = _random_W(d_in=d, d_out=d)
    pi = ANALYZER.polysemanticity_index(W)
    assert pi.shape == (d,), f"Expected ({d},), got {pi.shape}"
    ss = ANALYZER.superposition_score(W)
    assert 0.0 <= ss <= 1.0


# ---------------------------------------------------------------------------
# 10. Edge case: single neuron (d_out == 1)
# ---------------------------------------------------------------------------

def test_edge_case_single_neuron():
    W = _random_W(d_in=D_IN, d_out=1)
    pi = ANALYZER.polysemanticity_index(W)
    assert pi.shape == (1,), f"Expected (1,), got {pi.shape}"
    ss = ANALYZER.superposition_score(W)
    # Only one neuron — no pairs → SS = 0
    assert ss == 0.0, f"Single neuron should give SS=0, got {ss}"


# ---------------------------------------------------------------------------
# 11. PI range: all PI values in [0, 1]
# ---------------------------------------------------------------------------

def test_pi_range():
    torch.manual_seed(7)
    for _ in range(5):
        W = torch.randn(D_IN, D_OUT)
        pi = ANALYZER.polysemanticity_index(W)
        assert (pi >= 0.0).all() and (pi <= 1.0).all(), (
            f"PI out of [0,1]: min={pi.min():.4f}, max={pi.max():.4f}"
        )


# ---------------------------------------------------------------------------
# 12. SS range: superposition_score in [0, 1]
# ---------------------------------------------------------------------------

def test_ss_range():
    torch.manual_seed(7)
    for _ in range(5):
        W = torch.randn(D_IN, D_OUT)
        ss = ANALYZER.superposition_score(W)
        assert 0.0 <= ss <= 1.0, f"SS out of [0,1]: {ss}"


# ---------------------------------------------------------------------------
# 13. Gradient flow: PI, PR, SS differentiable
# ---------------------------------------------------------------------------

def test_gradient_flow_pi():
    W = torch.randn(D_IN, D_OUT, requires_grad=True)
    pi = ANALYZER.polysemanticity_index(W)
    pi.sum().backward()
    assert W.grad is not None, "PI should be differentiable w.r.t. W"
    assert not W.grad.isnan().any(), "PI gradient should not contain NaN"


def test_gradient_flow_pr():
    acts = torch.randn(N_SAMPLES, D_OUT, requires_grad=True)
    # participation_ratio returns a Python float; compute via covariance path
    cov = acts.T @ acts / N_SAMPLES
    eigenvalues = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    sum_lam = eigenvalues.sum()
    sum_lam2 = (eigenvalues ** 2).sum()
    pr = (sum_lam ** 2) / sum_lam2
    pr.backward()
    assert acts.grad is not None, "PR should be differentiable w.r.t. activations"


def test_gradient_flow_ss():
    W = torch.randn(D_IN, D_OUT, requires_grad=True)
    eps = 1e-8
    norms = W.norm(dim=0, keepdim=True).clamp(min=eps)
    W_norm = W / norms
    gram = W_norm.T @ W_norm
    d_out = D_OUT
    mask = ~torch.eye(d_out, dtype=torch.bool)
    ss = gram.abs()[mask].mean()
    ss.backward()
    assert W.grad is not None, "SS should be differentiable w.r.t. W"


# ---------------------------------------------------------------------------
# 14. Numerical stability: no NaN/Inf on random weight matrices
# ---------------------------------------------------------------------------

def test_numerical_stability():
    torch.manual_seed(99)
    for seed in range(10):
        W = torch.randn(D_IN, D_OUT) * (10 ** (seed - 5))  # range 1e-5 .. 1e4
        acts = torch.randn(N_SAMPLES, D_OUT) * (10 ** (seed - 5))

        pi = ANALYZER.polysemanticity_index(W)
        assert not pi.isnan().any(), f"NaN in PI for seed={seed}"
        assert not pi.isinf().any(), f"Inf in PI for seed={seed}"

        ss = ANALYZER.superposition_score(W)
        assert not math.isnan(ss), f"NaN in SS for seed={seed}"
        assert not math.isinf(ss), f"Inf in SS for seed={seed}"

        pr = ANALYZER.participation_ratio(acts)
        assert not math.isnan(pr), f"NaN in PR for seed={seed}"
        assert not math.isinf(pr), f"Inf in PR for seed={seed}"

        sparsity = ANALYZER.activation_sparsity(acts)
        assert not sparsity.isnan().any(), f"NaN in sparsity for seed={seed}"
