"""Unit tests for ColBERT late-interaction scorer."""

from __future__ import annotations

import math

import pytest
import torch

from src.retrieval.colbert_late_interaction import ColBERTConfig, ColBERTScorer


D = 8


def _scorer(normalize: bool = True, embed_dim: int = D) -> ColBERTScorer:
    return ColBERTScorer(ColBERTConfig(embed_dim=embed_dim, normalize=normalize))


def test_score_pair_returns_scalar_float() -> None:
    s = _scorer()
    q = torch.randn(3, D)
    d = torch.randn(5, D)
    out = s.score_pair(q, d)
    assert isinstance(out, float)
    assert math.isfinite(out)


def test_score_batch_returns_tensor_B() -> None:
    s = _scorer()
    B = 4
    q = torch.randn(B, 3, D)
    d = torch.randn(B, 5, D)
    out = s.score_batch(q, d)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (B,)


def test_maxsim_hand_computed() -> None:
    # 2-token query, 3-token doc, no normalization so dot-products are raw.
    s = _scorer(normalize=False, embed_dim=2)
    # Use embed_dim=2 for a readable hand-computed example.
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Nq=2
    d = torch.tensor(
        [
            [0.5, 0.0],  # dot with q0 = 0.5, dot with q1 = 0.0
            [2.0, 1.0],  # dot with q0 = 2.0, dot with q1 = 1.0
            [0.0, 3.0],  # dot with q0 = 0.0, dot with q1 = 3.0
        ]
    )
    # Max over doc tokens: q0 -> 2.0, q1 -> 3.0. Sum = 5.0.
    out = s.score_pair(q, d)
    assert out == pytest.approx(5.0, abs=1e-5)


def test_identical_q_and_d_high_score() -> None:
    s = _scorer(normalize=True)
    q = torch.randn(4, D)
    score_identical = s.score_pair(q, q.clone())
    score_random = s.score_pair(q, torch.randn(4, D))
    # With L2 norm, identical tokens yield per-token max = 1.0, so score == Nq.
    assert score_identical == pytest.approx(4.0, abs=1e-5)
    assert score_identical > score_random


def test_matched_doc_beats_random_doc() -> None:
    torch.manual_seed(0)
    s = _scorer(normalize=True)
    q = torch.randn(5, D)
    # Matched doc contains the query tokens plus extras.
    matched = torch.cat([q, torch.randn(3, D)], dim=0)
    random_doc = torch.randn(8, D)
    assert s.score_pair(q, matched) > s.score_pair(q, random_doc)


def test_score_query_against_corpus_length() -> None:
    s = _scorer()
    q = torch.randn(3, D)
    corpus = [torch.randn(n, D) for n in (1, 4, 7, 2)]
    out = s.score_query_against_corpus(q, corpus)
    assert isinstance(out, list)
    assert len(out) == len(corpus)
    assert all(isinstance(x, float) for x in out)


def test_normalize_bounds_score_range() -> None:
    s = _scorer(normalize=True)
    Nq = 6
    # Crank embeddings to large magnitudes — normalize should still bound.
    q = torch.randn(Nq, D) * 100.0
    d = torch.randn(10, D) * 100.0
    score = s.score_pair(q, d)
    assert -Nq - 1e-4 <= score <= Nq + 1e-4


def test_shape_mismatch_raises() -> None:
    s = _scorer()
    # Wrong embed_dim.
    with pytest.raises(ValueError):
        s.score_pair(torch.randn(2, D + 1), torch.randn(3, D))
    # Wrong rank for batch API.
    with pytest.raises(ValueError):
        s.score_batch(torch.randn(2, D), torch.randn(2, D))
    # Batch size mismatch.
    with pytest.raises(ValueError):
        s.score_batch(torch.randn(2, 3, D), torch.randn(3, 3, D))


def test_empty_doc_raises() -> None:
    s = _scorer()
    with pytest.raises(ValueError):
        s.score_pair(torch.randn(2, D), torch.empty(0, D))
    with pytest.raises(ValueError):
        s.score_batch(torch.randn(1, 2, D), torch.empty(1, 0, D))
    with pytest.raises(ValueError):
        s.score_query_against_corpus(torch.randn(2, D), [torch.empty(0, D)])


def test_empty_query_raises() -> None:
    s = _scorer()
    with pytest.raises(ValueError):
        s.score_pair(torch.empty(0, D), torch.randn(3, D))
    with pytest.raises(ValueError):
        s.score_batch(torch.empty(1, 0, D), torch.randn(1, 3, D))
    with pytest.raises(ValueError):
        s.score_query_against_corpus(torch.empty(0, D), [torch.randn(3, D)])


def test_batch_dim_preserved() -> None:
    s = _scorer()
    for B in (1, 2, 7):
        q = torch.randn(B, 3, D)
        d = torch.randn(B, 4, D)
        assert s.score_batch(q, d).shape == (B,)


def test_determinism() -> None:
    s = _scorer()
    torch.manual_seed(42)
    q = torch.randn(3, D)
    d = torch.randn(5, D)
    a = s.score_pair(q, d)
    b = s.score_pair(q, d)
    assert a == b


def test_gradient_flow() -> None:
    s = _scorer(normalize=True)
    q = torch.randn(3, D, requires_grad=True)
    d = torch.randn(4, D, requires_grad=True)
    # Use score_batch so output is a differentiable tensor.
    score = s.score_batch(q.unsqueeze(0), d.unsqueeze(0)).sum()
    score.backward()
    assert q.grad is not None
    assert d.grad is not None
    assert torch.isfinite(q.grad).all()
    assert q.grad.abs().sum() > 0


def test_embed_dim_1_degenerate() -> None:
    s = _scorer(normalize=False, embed_dim=1)
    q = torch.tensor([[1.0], [-1.0]])  # Nq=2
    d = torch.tensor([[2.0], [0.5], [-3.0]])  # Nd=3
    # q0=1.0 dots: [2.0, 0.5, -3.0] -> max 2.0
    # q1=-1.0 dots: [-2.0, -0.5, 3.0] -> max 3.0
    # Sum = 5.0
    assert s.score_pair(q, d) == pytest.approx(5.0, abs=1e-5)


def test_config_rejects_bad_embed_dim() -> None:
    with pytest.raises(ValueError):
        ColBERTScorer(ColBERTConfig(embed_dim=0))
