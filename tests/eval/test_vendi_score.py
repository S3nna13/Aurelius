"""
test_vendi_score.py -- Tests for vendi_score module.

12 focused tests covering vendi_score, embedding_vendi_score,
token_vendi_score, and VendiScorer.

Reference: "The Vendi Score: A Diversity Evaluation Metric for Machine
Learning", Friedman & Dieng, arXiv:2210.02410.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.eval.vendi_score import (
    VendiScorer,
    embedding_vendi_score,
    token_vendi_score,
    vendi_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def identity_kernel(n: int) -> torch.Tensor:
    """n×n identity matrix (maximally diverse kernel)."""
    return torch.eye(n, dtype=torch.float32)


def ones_kernel(n: int) -> torch.Tensor:
    """n×n all-ones matrix (all samples identical)."""
    return torch.ones(n, n, dtype=torch.float32)


def make_orthogonal_embeddings(n: int, d: int) -> torch.Tensor:
    """n orthogonal unit vectors in R^d (requires d >= n)."""
    assert d >= n, "Need d >= n for orthogonal embeddings"
    basis = torch.eye(d, dtype=torch.float32)
    return basis[:n]  # (n, d)


def make_identical_embeddings(n: int, d: int) -> torch.Tensor:
    """n copies of the same unit vector."""
    v = torch.randn(d)
    v = v / v.norm()
    return v.unsqueeze(0).expand(n, -1).clone()  # (n, d)


# ---------------------------------------------------------------------------
# Test 1: identity kernel → VS = n (maximally diverse)
# ---------------------------------------------------------------------------

def test_vendi_score_identity_kernel():
    """K = I_n  =>  VS = n."""
    for n in (3, 5, 8):
        K = identity_kernel(n)
        vs = vendi_score(K)
        assert abs(vs - n) < 1e-4, f"Expected VS={n}, got {vs} for n={n}"


# ---------------------------------------------------------------------------
# Test 2: all-ones kernel → VS ≈ 1 (all samples identical)
# ---------------------------------------------------------------------------

def test_vendi_score_ones_kernel():
    """K = 1·1^T  =>  VS ≈ 1 (rank-1 matrix, single non-zero eigenvalue)."""
    for n in (3, 5, 10):
        K = ones_kernel(n)
        vs = vendi_score(K)
        assert abs(vs - 1.0) < 1e-4, f"Expected VS≈1, got {vs} for n={n}"


# ---------------------------------------------------------------------------
# Test 3: VS range — 1 ≤ VS ≤ n
# ---------------------------------------------------------------------------

def test_vendi_score_range():
    """VS is always in [1, n] for a valid PSD kernel."""
    torch.manual_seed(0)
    for n in (4, 8, 16):
        # Random PSD kernel: K = A A^T / n
        A = torch.randn(n, n)
        K = A @ A.T / n
        vs = vendi_score(K)
        assert 1.0 - 1e-4 <= vs <= n + 1e-4, (
            f"VS={vs} out of range [1, {n}] for n={n}"
        )


# ---------------------------------------------------------------------------
# Test 4: identical embeddings → VS ≈ 1
# ---------------------------------------------------------------------------

def test_embedding_vendi_score_identical():
    """n copies of the same embedding → VS ≈ 1 for all kernels."""
    torch.manual_seed(1)
    n, d = 6, 16
    emb = make_identical_embeddings(n, d)
    for kernel in ("cosine", "linear", "rbf"):
        vs = embedding_vendi_score(emb, kernel=kernel)
        assert abs(vs - 1.0) < 1e-3, (
            f"kernel={kernel}: expected VS≈1 for identical embeddings, got {vs}"
        )


# ---------------------------------------------------------------------------
# Test 5: orthogonal embeddings → VS ≈ n
# ---------------------------------------------------------------------------

def test_embedding_vendi_score_orthogonal():
    """n orthogonal unit embeddings → VS ≈ n for cosine and linear kernels."""
    n, d = 5, 8
    emb = make_orthogonal_embeddings(n, d)
    for kernel in ("cosine", "linear"):
        vs = embedding_vendi_score(emb, kernel=kernel)
        assert abs(vs - n) < 1e-3, (
            f"kernel={kernel}: expected VS≈{n} for orthogonal embeddings, got {vs}"
        )


# ---------------------------------------------------------------------------
# Test 6: cosine kernel produces PSD matrix
# ---------------------------------------------------------------------------

def test_cosine_kernel_psd():
    """Cosine kernel matrix must be PSD (all eigenvalues ≥ 0)."""
    torch.manual_seed(2)
    n, d = 10, 32
    emb = torch.randn(n, d)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    K = emb @ emb.T  # cosine because emb is already normalised
    eigenvalues = torch.linalg.eigvalsh(K)
    assert (eigenvalues >= -1e-6).all(), (
        f"Cosine kernel has negative eigenvalues: {eigenvalues.min().item()}"
    )


# ---------------------------------------------------------------------------
# Test 7: token_vendi_score — all identical sequences → VS ≈ 1
# ---------------------------------------------------------------------------

def test_token_vendi_score_identical():
    """All identical token sequences → VS ≈ 1."""
    seq = [1, 2, 3, 4]
    sequences = [seq] * 8
    vs = token_vendi_score(sequences)
    assert abs(vs - 1.0) < 1e-4, f"Expected VS≈1 for identical seqs, got {vs}"


# ---------------------------------------------------------------------------
# Test 8: token_vendi_score — all unique sequences → VS ≈ n
# ---------------------------------------------------------------------------

def test_token_vendi_score_unique():
    """All distinct token sequences → VS ≈ n."""
    n = 6
    sequences = [[i, i + 100, i + 200] for i in range(n)]
    vs = token_vendi_score(sequences)
    assert abs(vs - n) < 1e-3, f"Expected VS≈{n} for unique seqs, got {vs}"


# ---------------------------------------------------------------------------
# Test 9: VendiScorer.score_batch returns list of correct length
# ---------------------------------------------------------------------------

def test_scorer_score_batch_length():
    """score_batch returns a list whose length equals the number of groups."""
    torch.manual_seed(3)
    scorer = VendiScorer(kernel="cosine")
    groups = [torch.randn(n, 16) for n in (3, 5, 7, 4)]
    results = scorer.score_batch(groups)
    assert isinstance(results, list), "score_batch must return a list"
    assert len(results) == len(groups), (
        f"Expected {len(groups)} scores, got {len(results)}"
    )
    # Each score must be a float
    for i, s in enumerate(results):
        assert isinstance(s, float), f"results[{i}] is not a float: {type(s)}"


# ---------------------------------------------------------------------------
# Test 10: numerical stability with near-zero embeddings
# ---------------------------------------------------------------------------

def test_numerical_stability_near_zero():
    """Near-zero embeddings must not produce NaN or Inf scores."""
    torch.manual_seed(4)
    n, d = 8, 16
    emb = torch.randn(n, d) * 1e-8   # very small magnitudes
    for kernel in ("cosine", "linear", "rbf"):
        vs = embedding_vendi_score(emb, kernel=kernel)
        assert math.isfinite(vs), (
            f"kernel={kernel}: VS is not finite for near-zero embeddings: {vs}"
        )


# ---------------------------------------------------------------------------
# Test 11: determinism — same input → same score
# ---------------------------------------------------------------------------

def test_determinism():
    """Identical inputs must produce identical scores (no random state)."""
    torch.manual_seed(5)
    emb = torch.randn(10, 32)
    vs1 = embedding_vendi_score(emb.clone(), kernel="cosine")
    vs2 = embedding_vendi_score(emb.clone(), kernel="cosine")
    assert vs1 == vs2, f"Non-deterministic scores: {vs1} vs {vs2}"


# ---------------------------------------------------------------------------
# Test 12: n=1 (single sample) → VS = 1
# ---------------------------------------------------------------------------

def test_single_sample():
    """A single sample always has VS = 1 regardless of kernel."""
    torch.manual_seed(6)
    emb = torch.randn(1, 16)

    # embedding_vendi_score
    for kernel in ("cosine", "linear", "rbf"):
        vs = embedding_vendi_score(emb, kernel=kernel)
        assert vs == 1.0, f"kernel={kernel}: single sample VS should be 1.0, got {vs}"

    # token_vendi_score
    vs = token_vendi_score([[1, 2, 3]])
    assert vs == 1.0, f"token single sample VS should be 1.0, got {vs}"

    # vendi_score with 1×1 kernel
    K = torch.ones(1, 1)
    vs = vendi_score(K)
    assert vs == 1.0, f"1×1 kernel VS should be 1.0, got {vs}"
