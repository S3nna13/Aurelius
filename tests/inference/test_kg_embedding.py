"""Tests for TransE and RotatE knowledge graph embedding models."""

from __future__ import annotations

import pytest
import torch

from src.inference.kg_embedding import (
    KGEConfig,
    TransE,
    RotatE,
    KGRetriever,
    negative_sample,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ENTITIES = 100
N_RELATIONS = 20
EMBED_DIM = 64
BATCH_SIZE = 4


@pytest.fixture()
def cfg():
    return KGEConfig(
        n_entities=N_ENTITIES,
        n_relations=N_RELATIONS,
        embed_dim=EMBED_DIM,
        margin=1.0,
        norm=1,
    )


@pytest.fixture()
def transe(cfg):
    return TransE(cfg)


@pytest.fixture()
def rotate(cfg):
    return RotatE(cfg)


def _batch(b: int = BATCH_SIZE):
    h = torch.randint(0, N_ENTITIES, (b,))
    r = torch.randint(0, N_RELATIONS, (b,))
    t = torch.randint(0, N_ENTITIES, (b,))
    t_neg = torch.randint(0, N_ENTITIES, (b,))
    return h, r, t, t_neg


# ---------------------------------------------------------------------------
# TransE tests
# ---------------------------------------------------------------------------


def test_transe_score_shape(transe):
    h, r, t, _ = _batch()
    scores = transe.score(h, r, t)
    assert scores.shape == (BATCH_SIZE,), f"Expected ({BATCH_SIZE},), got {scores.shape}"


def test_transe_loss_scalar(transe):
    h, r, t, t_neg = _batch()
    loss = transe.loss(h, r, t, t_neg)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() >= 0.0


def test_transe_loss_decreases(cfg):
    """Loss should decrease after a few gradient steps."""
    model = TransE(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    h, r, t, t_neg = _batch(16)

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = model.loss(h, r, t, t_neg)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# RotatE tests
# ---------------------------------------------------------------------------


def test_rotate_score_shape(rotate):
    h, r, t, _ = _batch()
    scores = rotate.score(h, r, t)
    assert scores.shape == (BATCH_SIZE,), f"Expected ({BATCH_SIZE},), got {scores.shape}"


def test_rotate_loss_scalar(rotate):
    h, r, t, t_neg = _batch()
    loss = rotate.loss(h, r, t, t_neg)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


def test_rotate_embed_dim_even_required():
    """Odd embed_dim should raise ValueError."""
    with pytest.raises(ValueError, match="even"):
        RotatE(KGEConfig(n_entities=10, n_relations=5, embed_dim=63))


# ---------------------------------------------------------------------------
# KGRetriever tests
# ---------------------------------------------------------------------------


def test_kg_retriever_build_index(transe):
    retriever = KGRetriever(transe)
    retriever.build_index()  # should not raise


def test_kg_retriever_retrieve_shape(transe):
    retriever = KGRetriever(transe)
    retriever.build_index()
    top_k = 5
    query = torch.randn(BATCH_SIZE, EMBED_DIM)
    entity_ids, scores = retriever.retrieve(query, top_k=top_k)
    assert entity_ids.shape == (BATCH_SIZE, top_k), f"entity_ids shape wrong: {entity_ids.shape}"
    assert scores.shape == (BATCH_SIZE, top_k), f"scores shape wrong: {scores.shape}"


def test_kg_retriever_top_k_valid(transe):
    retriever = KGRetriever(transe)
    retriever.build_index()
    query = torch.randn(BATCH_SIZE, EMBED_DIM)
    entity_ids, _ = retriever.retrieve(query, top_k=10)
    assert entity_ids.min().item() >= 0
    assert entity_ids.max().item() < N_ENTITIES


# ---------------------------------------------------------------------------
# negative_sample tests
# ---------------------------------------------------------------------------


def test_negative_sample_shape():
    b = 8
    n_neg = 3
    h = torch.randint(0, N_ENTITIES, (b,))
    t = torch.randint(0, N_ENTITIES, (b,))
    neg_h, neg_t = negative_sample(N_ENTITIES, h, t, n_neg=n_neg)
    assert neg_h.shape == (b * n_neg,), f"neg_h shape wrong: {neg_h.shape}"
    assert neg_t.shape == (b * n_neg,), f"neg_t shape wrong: {neg_t.shape}"


def test_negative_sample_excludes_true():
    """Negative samples should (mostly) differ from true pairs.

    With 100 entities and random sampling the probability of accidentally
    matching all 8*3=24 samples is astronomically low.
    """
    b = 8
    n_neg = 3
    h = torch.arange(b) % N_ENTITIES
    t = torch.arange(b) % N_ENTITIES
    neg_h, neg_t = negative_sample(N_ENTITIES, h, t, n_neg=n_neg)

    # At least some negatives should differ from original h or t
    h_rep = h.repeat_interleave(n_neg)
    t_rep = t.repeat_interleave(n_neg)
    h_differs = (neg_h != h_rep).any().item()
    t_differs = (neg_t != t_rep).any().item()
    assert h_differs or t_differs, "All negative samples matched true pairs — very unlikely"
